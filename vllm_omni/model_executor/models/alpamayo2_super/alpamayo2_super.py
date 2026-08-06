# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native Qwen3-VL rollout with the released Alpamayo 2 Super expert."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
from transformers import DynamicCache
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


class Alpamayo2SuperProcessingInfo(Qwen3VLProcessingInfo):
    """Use the processor and extended tokenizer shipped with Super."""

    def get_hf_processor(self, **kwargs: object) -> Any:
        config = self.get_hf_config()
        kwargs.pop("device", None)
        return cached_get_processor(
            config._name_or_path,
            processor_cls=Qwen3VLProcessor,
            tokenizer=self.get_tokenizer(),
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Alpamayo2SuperProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Alpamayo2SuperForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """Run Super's VLM in vLLM and its terminal diffusion expert in-model.

    The autoregressive Qwen3-VL prefix stays entirely native to vLLM. Once
    ``<|traj_future_start|>`` is emitted, the model gathers that request's
    paged KV prefix and gives it to the released, non-causal action expert for
    fixed-step flow matching. Policy inference is intentionally single-request
    for now; ordinary VQA requests retain the normal vLLM batching path.
    """

    have_multimodal_outputs = True
    needs_runner_kv_cache = True
    # This is a terminal single-stage policy. Only the structured action
    # payload is client-facing; no downstream stage consumes VLM hidden states.
    requires_full_prefix_cached_hidden_states = False
    omni_pooler_payload_include_hidden = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.alpamayo_config = vllm_config.model_config.hf_config

        # The released package owns the action-space math and expert modules.
        # Keeping it an image/model dependency avoids duplicating architecture
        # code in the engine while leaving the VLM on vLLM's native Qwen path.
        from alpamayo2_super.models.expert import ExpertModel, ExpertModelConfig

        expert_config = ExpertModelConfig(**copy.deepcopy(self.alpamayo_config.expert_config))
        # A non-causal 4D mask is required by the diffusion suffix. HF SDPA
        # supports it without requiring the separately packaged flash-attn.
        expert_config.llm_config._attn_implementation = "sdpa"
        self.expert = ExpertModel(
            expert_config,
            dtype=vllm_config.model_config.dtype,
        )
        self._force_future_end = False

    @property
    def future_start_id(self) -> int:
        return int(self.alpamayo_config.traj_ids["future_start"])

    @property
    def future_end_id(self) -> int:
        return int(self.alpamayo_config.traj_ids["future_end"])

    def _policy_observation(self, sampling_extra_args: object) -> Mapping[str, Any] | None:
        if not isinstance(sampling_extra_args, list):
            return None
        policy_requests = [
            extra
            for extra in sampling_extra_args
            if isinstance(extra, Mapping) and isinstance(extra.get("robot_obs"), Mapping)
        ]
        if len(policy_requests) > 1:
            raise RuntimeError("Alpamayo 2 Super policy inference currently supports max_num_seqs=1")
        if not policy_requests:
            return None
        if len(sampling_extra_args) != 1:
            raise RuntimeError("Alpamayo 2 Super policy inference cannot share a batch")
        observation = policy_requests[0].get("robot_obs")
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
        """Keep terminal token IDs visible with multimodal embeddings."""

        del req_ids, num_computed_tokens, num_scheduled_tokens
        if inputs_embeds is not None and input_ids is None and input_ids_buffer is not None:
            input_ids = input_ids_buffer
        return input_ids, positions

    @staticmethod
    def _gather_prefix_cache(
        caches: list[torch.Tensor],
        block_table: torch.Tensor,
        seq_len: int,
    ) -> DynamicCache:
        """Convert one request's standard paged cache to HF cache layout."""

        dynamic_cache = DynamicCache()
        for layer_index, cache in enumerate(caches):
            if cache.ndim != 5 or (cache.shape[0] != 2 and cache.shape[1] != 2):
                raise RuntimeError(
                    "Alpamayo 2 Super requires the standard vLLM paged KV layout"
                )
            block_size = cache.shape[2]
            num_blocks = (seq_len + block_size - 1) // block_size
            block_ids = block_table[:num_blocks].to(dtype=torch.long)
            if cache.shape[0] == 2:
                key = cache[0].index_select(0, block_ids).flatten(0, 1)[:seq_len]
                value = cache[1].index_select(0, block_ids).flatten(0, 1)[:seq_len]
            else:
                selected = cache.index_select(0, block_ids)
                key = selected[:, 0].flatten(0, 1)[:seq_len]
                value = selected[:, 1].flatten(0, 1)[:seq_len]
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

    @staticmethod
    def _observation_tensor(value: Any, *, device: torch.device) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=device, dtype=torch.float32)
        while tensor.ndim > 1 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        return tensor

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
        device = caches[0].device
        sample_count = int(extra_args.get("num_traj_samples", 1))
        inference_steps = int(extra_args.get("diffusion_steps", 10))
        temperature = float(extra_args.get("action_temperature", 1.0))
        if sample_count < 1 or inference_steps < 1:
            raise ValueError("num_traj_samples and diffusion_steps must be positive")

        prefix = self._repeat_cache(
            self._gather_prefix_cache(caches, block_table, seq_len),
            sample_count,
        )
        prefix_len = prefix.get_seq_length()
        action_dims = tuple(int(dim) for dim in self.expert.action_space.get_action_space_dims())

        generator = None
        if (sampling_seed := extra_args.get("_sampling_seed")) is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(sampling_seed))
        action = torch.randn(
            sample_count,
            *action_dims,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        initial_noise = action.clone() if bool(extra_args.get("_return_action_noise", False)) else None
        action = action * temperature

        if positions.ndim == 2 and positions.shape[0] == 3:
            last_position = positions[:, -1:].to(device)
        else:
            last_position = positions.reshape(-1)[-1:].repeat(3, 1).to(device)
        suffix_length = action_dims[0]
        expert_positions = (
            last_position[:, None, :]
            + 1
            + torch.arange(suffix_length, device=device)[None, None, :]
        ).expand(3, sample_count, suffix_length)
        weight_dtype = self.expert.action_out_proj.weight.dtype
        attention_mask = torch.zeros(
            (sample_count, 1, suffix_length, prefix_len + suffix_length),
            device=device,
            # Transformers SDPA requires additive-mask and query dtypes to
            # match. With a compact paged prefix this mask contains no padded
            # tail, so BF16 zeros are exactly equivalent to reference FP32.
            dtype=weight_dtype,
        )

        for step in range(inference_steps):
            timestep = action.new_full(
                (sample_count,) + (1,) * len(action_dims),
                step / inference_steps,
            )
            with torch.autocast(device_type=device.type, dtype=weight_dtype):
                embeddings = self.expert.action_in_proj(action, timestep)
            outputs = self.expert.expert(
                inputs_embeds=embeddings.to(weight_dtype),
                position_ids=expert_positions,
                past_key_values=prefix,
                attention_mask=attention_mask,
                use_cache=True,
                is_causal=not bool(self.expert.config.expert_non_causal_attention),
            )
            prefix.crop(prefix_len)
            velocity = self.expert.action_out_proj(outputs.last_hidden_state)
            action = action + velocity.float().view(sample_count, *action_dims) / inference_steps

        history_xyz = self._observation_tensor(observation["ego_history_xyz"], device=device)
        history_rot = self._observation_tensor(observation["ego_history_rot"], device=device)
        if history_xyz.ndim != 2 or history_xyz.shape[-1] != 3:
            raise ValueError(f"ego_history_xyz must have shape (T, 3), got {tuple(history_xyz.shape)}")
        if history_rot.ndim != 3 or history_rot.shape[-2:] != (3, 3):
            raise ValueError(
                f"ego_history_rot must have shape (T, 3, 3), got {tuple(history_rot.shape)}"
            )
        repeated_history_xyz = history_xyz.unsqueeze(0).expand(sample_count, -1, -1)
        repeated_history_rot = history_rot.unsqueeze(0).expand(sample_count, -1, -1, -1)
        pred_xyz, pred_rot = self.expert.action_space.action_to_traj(
            action,
            repeated_history_xyz,
            repeated_history_rot,
        )
        result = {
            "pred_trajectories": pred_xyz,
            "pred_rotations": pred_rot,
            "actions": pred_xyz,
            "rotations": pred_rot,
            "normalized_controls": action,
        }
        if initial_noise is not None:
            result["action_noise"] = initial_noise
        return result

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors | OmniOutput:
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
        if isinstance(hidden_states, IntermediateTensors):
            return hidden_states
        observation = self._policy_observation(sampling_extra_args)
        trigger = (
            observation is not None
            and input_ids is not None
            and bool(torch.any(input_ids == self.future_start_id))
        )
        multimodal_outputs: dict[str, torch.Tensor] = {}
        if trigger:
            if runner_kv_cache_context is None:
                raise RuntimeError("Super action generation requires runner KV-cache access")
            if len(runner_kv_cache_context.request_ids) != 1:
                raise RuntimeError("Super action generation currently supports one request")
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
        return OmniOutput(
            text_hidden_states=text_hidden_states,
            multimodal_outputs=multimodal_outputs,
        )

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
            traj_ids = self.alpamayo_config.traj_ids
            start = min(int(traj_ids["history_id0"]), int(traj_ids["future_id0"]))
            logits[..., start : start + int(self.alpamayo_config.traj_vocab_size)] = -torch.inf
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        vlm_weights: list[tuple[str, torch.Tensor]] = []
        local_weights: list[tuple[str, torch.Tensor]] = []
        for name, weight in weights:
            if name.startswith("vlm."):
                vlm_weights.append((name.removeprefix("vlm."), weight))
            elif name.startswith("expert."):
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
