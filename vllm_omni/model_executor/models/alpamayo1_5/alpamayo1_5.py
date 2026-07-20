# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native vLLM Qwen3-VL rollout with the Alpamayo 1.5 action expert."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
from torch import nn
from transformers import AutoModel, DynamicCache
from vllm.config import VllmConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
    Qwen3VLProcessor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.processor import cached_get_processor

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.runner_context import RunnerKVCacheContext

from .action import PerWaypointActionInProjV2, UnicycleTrajectoryDecoder
from .processing import get_observation_history, observation_tensor


class Alpamayo1_5ProcessingInfo(Qwen3VLProcessingInfo):
    """Load visual preprocessing from Cosmos rather than the policy repo."""

    def get_hf_processor(self, **kwargs: object) -> Any:
        config = self.get_hf_config()
        # ``device`` is an invocation-only Qwen image-processor argument. If
        # forwarded to ``from_pretrained`` it is treated as constructor state
        # (and fails with newer Transformers backends).
        kwargs.pop("device", None)
        # Use vLLM's process-wide processor cache. Calling from_pretrained
        # directly here rebuilds and deep-copies the large tokenizer several
        # times during every multimodal request.
        return cached_get_processor(
            config.vlm_name_or_path,
            processor_cls=Qwen3VLProcessor,
            tokenizer=self.get_tokenizer(),
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )


def _observation_rotation(observation: Mapping[str, Any], history: torch.Tensor) -> torch.Tensor:
    value = observation.get("ego_history_rot")
    if value is None:
        identity = torch.eye(3, dtype=history.dtype)
        return identity.expand(16, 3, 3).clone()
    rotation = observation_tensor(value)
    while rotation.ndim > 3 and rotation.shape[0] == 1:
        rotation = rotation.squeeze(0)
    if rotation.shape != (16, 3, 3):
        raise ValueError(f"ego_history_rot must have shape (16, 3, 3), got {tuple(rotation.shape)}")
    return rotation


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Alpamayo1_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Alpamayo1_5ForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """Alpamayo 1.5 inference model for a single-GPU BF16 worker.

    The Qwen3-VL prefix is computed by vLLM. On the decode step containing
    ``<|traj_future_start|>``, its paged KV cache is gathered into a temporary
    Transformers cache and consumed by the smaller bidirectional action
    expert for ten flow-matching steps.
    """

    have_multimodal_outputs = True
    needs_runner_kv_cache = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.alpamayo_config = vllm_config.model_config.hf_config
        expert_config = copy.deepcopy(self.alpamayo_config.text_config)
        for key, value in self.alpamayo_config.expert_cfg.items():
            setattr(expert_config, key, value)
        # The policy config inherits ``flash_attention_2`` from the backbone,
        # but official vLLM images expose their own FlashAttention kernels and
        # do not necessarily install HF's separate ``flash-attn`` package.
        # SDPA supports the expert's non-causal 4D mask and avoids coupling the
        # native vLLM backbone to that optional Transformers dependency.
        expert_config._attn_implementation = "sdpa"
        self.expert = AutoModel.from_config(expert_config)
        if hasattr(self.expert, "embed_tokens"):
            del self.expert.embed_tokens
        input_config = self.alpamayo_config.action_in_proj_cfg
        self.action_in_proj = PerWaypointActionInProjV2(
            expert_config.hidden_size,
            num_enc_layers=int(input_config.get("num_enc_layers", 2)),
            hidden_size=int(input_config.get("hidden_size", 512)),
            max_freq=float(input_config.get("max_freq", 100.0)),
            num_fourier_feats=int(input_config.get("num_fourier_feats", 20)),
        )
        self.action_out_proj = nn.Linear(expert_config.hidden_size, 2)
        self.trajectory_decoder = UnicycleTrajectoryDecoder(self.alpamayo_config.action_space_cfg)
        self.expert_config = expert_config
        self._compiled_expert: nn.Module | None = None
        self._force_future_end = False

    @property
    def future_start_id(self) -> int:
        return int(self.alpamayo_config.traj_token_ids.get("future_start", 155681))

    @property
    def future_end_id(self) -> int:
        return int(self.alpamayo_config.traj_token_ids.get("future_end", 155683))

    def _policy_observation(self, sampling_extra_args: object) -> Mapping[str, Any] | None:
        if not isinstance(sampling_extra_args, list):
            return None
        policy_requests = [
            extra
            for extra in sampling_extra_args
            if isinstance(extra, Mapping) and isinstance(extra.get("robot_obs"), Mapping)
        ]
        if len(policy_requests) > 1:
            raise RuntimeError("Alpamayo policy inference currently supports max_num_seqs=1")
        if not policy_requests:
            return None
        if len(sampling_extra_args) != 1:
            raise RuntimeError("Alpamayo policy inference cannot share a batch with other requests")
        extra = policy_requests[0]
        if not isinstance(extra, Mapping):
            return None
        observation = extra.get("robot_obs")
        return observation if isinstance(observation, Mapping) else None

    def prepare_runner_inputs(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        req_ids: Sequence[str],
        num_computed_tokens: Sequence[int],
        num_scheduled_tokens: Sequence[int],
        input_ids_buffer: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Keep token IDs visible when vLLM supplies multimodal embeddings."""

        del req_ids, num_computed_tokens, num_scheduled_tokens
        if inputs_embeds is not None and input_ids is None and input_ids_buffer is not None:
            input_ids = input_ids_buffer
        return input_ids, positions

    @staticmethod
    def _gather_prefix_cache(caches: list[torch.Tensor], block_table: torch.Tensor, seq_len: int) -> DynamicCache:
        """Convert one request's paged vLLM cache to a Transformers cache."""

        dynamic_cache = DynamicCache()
        for layer_index, cache in enumerate(caches):
            if cache.ndim != 5 or (cache.shape[0] != 2 and cache.shape[1] != 2):
                raise RuntimeError("Alpamayo currently requires the standard vLLM paged-attention KV layout")
            block_size = cache.shape[2]
            num_blocks = (seq_len + block_size - 1) // block_size
            block_ids = block_table[:num_blocks].to(dtype=torch.long)
            if cache.shape[0] == 2:
                key = cache[0].index_select(0, block_ids).flatten(0, 1)[:seq_len]
                value = cache[1].index_select(0, block_ids).flatten(0, 1)[:seq_len]
            else:
                # FlashAttention's current logical layout is
                # (blocks, 2, block_size, kv_heads, head_dim).
                selected = cache.index_select(0, block_ids)
                key = selected[:, 0].flatten(0, 1)[:seq_len]
                value = selected[:, 1].flatten(0, 1)[:seq_len]
            # HF cache layout is (batch, kv_heads, sequence, head_dim).
            dynamic_cache.update(
                key.transpose(0, 1).unsqueeze(0).contiguous(),
                value.transpose(0, 1).unsqueeze(0).contiguous(),
                layer_index,
            )
        return dynamic_cache

    @staticmethod
    def _repeat_cache(cache: DynamicCache, count: int) -> DynamicCache:
        if count == 1:
            return cache
        repeated = DynamicCache()
        for layer_index, layer in enumerate(cache.layers):
            repeated.update(
                layer.keys.repeat(count, 1, 1, 1),
                layer.values.repeat(count, 1, 1, 1),
                layer_index,
            )
        return repeated

    def _sample_actions(
        self,
        *,
        caches: list[torch.Tensor],
        block_table: torch.Tensor,
        seq_len: int,
        positions: torch.Tensor,
        observation: Mapping[str, Any],
        extra_args: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        profile = bool(extra_args.get("_profile_timings", False))
        timing_events = [torch.cuda.Event(enable_timing=True) for _ in range(5)] if profile else []
        if profile:
            timing_events[0].record()
        history = get_observation_history(observation).to(device=caches[0].device, dtype=torch.float32)
        history_rotation = _observation_rotation(observation, history).to(device=history.device, dtype=torch.float32)
        sample_count = int(extra_args.get("num_traj_samples", 6))
        inference_steps = int(extra_args.get("diffusion_steps", 10))
        temperature = float(extra_args.get("action_temperature", 1.0))
        prefix = self._repeat_cache(self._gather_prefix_cache(caches, block_table, seq_len), sample_count)
        prefix_len = prefix.get_seq_length()
        if profile:
            timing_events[1].record()
        generator = None
        sampling_seed = extra_args.get("_sampling_seed")
        if sampling_seed is not None:
            generator = torch.Generator(device=history.device)
            generator.manual_seed(int(sampling_seed))
        action = (
            torch.randn(
                sample_count,
                64,
                2,
                device=history.device,
                # The reference flow sampler keeps the integration state in
                # FP32 even when the model weights and expert run in BF16.
                dtype=torch.float32,
                generator=generator,
            )
            * temperature
        )
        if positions.ndim == 2 and positions.shape[0] == 3:
            last_position = positions[:, -1:].to(history.device)
        else:
            last_position = positions.reshape(-1)[-1:].repeat(3, 1).to(history.device)
        expert_positions = last_position[:, None, :] + 1 + torch.arange(64, device=history.device)[None, None, :]
        expert_positions = expert_positions.expand(3, sample_count, 64)
        attention_mask = torch.zeros(
            sample_count,
            1,
            64,
            prefix_len + 64,
            device=history.device,
            dtype=self.action_out_proj.weight.dtype,
        )
        if profile:
            timing_events[2].record()
        expert = self.expert
        if bool(extra_args.get("_compile_expert", False)):
            if self._compiled_expert is None:
                self._compiled_expert = torch.compile(
                    self.expert,
                    fullgraph=False,
                    dynamic=False,
                    options={"triton.cudagraphs": False},
                )
            expert = self._compiled_expert
        for step in range(inference_steps):
            timestep = action.new_full((sample_count, 1, 1), step / inference_steps)
            with torch.autocast(device_type="cuda", dtype=self.action_out_proj.weight.dtype):
                embeddings = self.action_in_proj(action, timestep)
            embeddings = embeddings.to(dtype=self.action_out_proj.weight.dtype)
            output = expert(
                inputs_embeds=embeddings,
                position_ids=expert_positions,
                past_key_values=prefix,
                attention_mask=attention_mask,
                use_cache=True,
                is_causal=not bool(self.alpamayo_config.expert_non_causal_attention),
            )
            prefix.crop(prefix_len)
            velocity = self.action_out_proj(output.last_hidden_state[:, -64:])
            action = action + velocity / inference_steps
        if profile:
            timing_events[3].record()
        repeated_history = history.unsqueeze(0).expand(sample_count, -1, -1)
        repeated_rotation = history_rotation.unsqueeze(0).expand(sample_count, -1, -1, -1)
        xyz, rotation = self.trajectory_decoder(action, repeated_history, repeated_rotation)
        if profile:
            timing_events[4].record()
            timing_events[4].synchronize()
        result = {
            "actions": xyz,
            "rotations": rotation,
            "normalized_controls": action,
        }
        if profile:
            # KV extraction, sampler setup, expert integration, trajectory decode.
            result["profile_timings_ms"] = torch.tensor(
                [timing_events[index].elapsed_time(timing_events[index + 1]) for index in range(4)],
                device=history.device,
                dtype=torch.float32,
            )
        return result

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors | OmniOutput:
        # Runner-only state is consumed by ``make_omni_output`` after the
        # graphable Qwen forward returns. Keeping request-dependent Python
        # branches here prevents vLLM from compiling/capturing the backbone.
        kwargs.pop("sampling_extra_args", None)
        kwargs.pop("runner_kv_cache_context", None)
        return super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def make_omni_output(
        self,
        hidden_states: torch.Tensor | IntermediateTensors,
        *,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor,
        sampling_extra_args: object = None,
        runner_kv_cache_context: RunnerKVCacheContext | None = None,
        **_: Any,
    ) -> IntermediateTensors | OmniOutput:
        """Run the request-dependent action hook outside the compiled VLM."""

        if isinstance(hidden_states, IntermediateTensors):
            return hidden_states
        observation = self._policy_observation(sampling_extra_args)
        trigger = observation is not None and input_ids is not None and bool(
            torch.any(input_ids == self.future_start_id)
        )
        multimodal_outputs: dict[str, torch.Tensor] = {}
        if trigger:
            if runner_kv_cache_context is None:
                raise RuntimeError("Alpamayo action generation requires runner KV-cache access")
            if len(runner_kv_cache_context.request_ids) != 1:
                raise RuntimeError("Alpamayo action generation currently supports one request")
            extra = sampling_extra_args[0] if isinstance(sampling_extra_args, list) else {}
            multimodal_outputs = self._sample_actions(
                caches=runner_kv_cache_context.caches,
                block_table=runner_kv_cache_context.block_table[0],
                seq_len=runner_kv_cache_context.sequence_lengths[0],
                positions=positions,
                observation=observation,
                extra_args=extra,
            )
            self._force_future_end = True
        text_hidden_states = hidden_states[0] if isinstance(hidden_states, (list, tuple)) else hidden_states
        return OmniOutput(text_hidden_states=text_hidden_states, multimodal_outputs=multimodal_outputs)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput) -> torch.Tensor:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        logits = super().compute_logits(hidden_states)
        if self._force_future_end:
            self._force_future_end = False
            selected = logits[..., self.future_end_id].clone()
            logits.fill_(-torch.inf)
            logits[..., self.future_end_id] = selected
        else:
            start = int(self.alpamayo_config.traj_token_start_idx)
            size = int(self.alpamayo_config.traj_vocab_size)
            logits[..., start : start + size] = -torch.inf
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        vlm_weights: list[tuple[str, torch.Tensor]] = []
        local_weights: list[tuple[str, torch.Tensor]] = []
        for name, weight in weights:
            if name.startswith("vlm."):
                vlm_weights.append((name.removeprefix("vlm."), weight))
            elif name.startswith(("expert.", "action_in_proj.", "action_out_proj.")):
                local_weights.append((name, weight))
        loaded = set(super().load_weights(vlm_weights))
        parameters = dict(self.named_parameters())
        for name, weight in local_weights:
            parameter = parameters.get(name)
            if parameter is None:
                continue
            loader = getattr(parameter, "weight_loader", default_weight_loader)
            loader(parameter, weight)
            loaded.add(name)
        return loaded
