# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native Qwen3-VL rollout with the released Alpamayo 2 Super expert."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
from torch import nn
from transformers import DynamicCache, StaticCache
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

_BATCHED_POLICY_OUTPUT_KEYS = frozenset(
    {
        "pred_trajectories",
        "pred_rotations",
        "actions",
        "rotations",
        "normalized_controls",
        "action_noise",
        "action_expert_invocation_batch_size",
    }
)


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
    fixed-step flow matching. Parallel samples are separate vLLM requests so
    every trajectory is conditioned on its own sampled VLM reasoning.
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
        self._compiled_expert: nn.Module | None = None
        self._action_graphs: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._force_future_end_indices: tuple[int, ...] = ()
        self._force_future_start_indices: tuple[int, ...] = ()
        self._mask_text_eos_indices: tuple[int, ...] = ()
        self._pending_policy_groups: dict[str, dict[str, dict[str, Any]]] = {}
        generation_config = vllm_config.model_config.try_get_generation_config()
        text_eos_ids = generation_config.get("eos_token_id")
        if text_eos_ids is None:
            text_eos_ids = getattr(self.alpamayo_config.text_config, "eos_token_id", None)
        if isinstance(text_eos_ids, int):
            text_eos_ids = [text_eos_ids]
        self._text_eos_token_ids = tuple(
            int(token_id) for token_id in (text_eos_ids or []) if int(token_id) != self.future_start_id
        )

    @property
    def future_start_id(self) -> int:
        return int(self.alpamayo_config.traj_ids["future_start"])

    @property
    def speculation_terminal_token_id(self) -> int:
        """Token after which speculative decoding yields to the action step."""
        return self.future_start_id

    @property
    def future_end_id(self) -> int:
        return int(self.alpamayo_config.traj_ids["future_end"])

    def _policy_observations(
        self,
        sampling_extra_args: object,
    ) -> list[Mapping[str, Any] | None]:
        if not isinstance(sampling_extra_args, list):
            return []
        observations: list[Mapping[str, Any] | None] = []
        for extra in sampling_extra_args:
            observation = extra.get("robot_obs") if isinstance(extra, Mapping) else None
            observations.append(observation if isinstance(observation, Mapping) else None)
        return observations

    @staticmethod
    def _request_positions(
        positions: torch.Tensor,
        start: int,
        end: int,
    ) -> torch.Tensor:
        """Select one request's position rows from a flattened runner batch."""
        if positions.ndim == 2 and positions.shape[0] == 3:
            return positions[:, start:end]
        return positions.reshape(-1)[start:end]

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
        """Convert one request's paged cache to HF cache layout.

        vLLM has used both a rank-5 layout with an explicit K/V axis and a
        rank-4 layout with K and V concatenated in the final dimension.
        """

        dynamic_cache = DynamicCache()
        for layer_index, cache in enumerate(caches):
            if cache.ndim == 4 and cache.shape[-1] % 2 == 0:
                # FlashAttention: [blocks, kv_heads, block_size, 2 * head_dim].
                block_size = cache.shape[2]
                num_blocks = (seq_len + block_size - 1) // block_size
                block_ids = block_table[:num_blocks].to(dtype=torch.long)
                key_blocks, value_blocks = cache.chunk(2, dim=-1)
                key = key_blocks.index_select(0, block_ids)
                value = value_blocks.index_select(0, block_ids)
                key = key.permute(1, 0, 2, 3).flatten(1, 2)[:, :seq_len]
                value = value.permute(1, 0, 2, 3).flatten(1, 2)[:, :seq_len]
            elif cache.ndim == 5 and (cache.shape[0] == 2 or cache.shape[1] == 2):
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
                key = key.transpose(0, 1)
                value = value.transpose(0, 1)
            else:
                raise RuntimeError(
                    "Alpamayo 2 Super requires a supported vLLM paged KV layout; "
                    f"layer {layer_index} has shape {tuple(cache.shape)}"
                )
            dynamic_cache.update(
                key.unsqueeze(0).contiguous(),
                value.unsqueeze(0).contiguous(),
                layer_index,
            )
        return dynamic_cache

    @staticmethod
    def _repeat_cache(cache: DynamicCache, count: int) -> DynamicCache:
        if count == 1:
            return cache
        # Repeat each layer in place so the one-sample source layer can be
        # released before the next layer is expanded. Building a second cache
        # retained the complete source cache until all repeated layers existed,
        # adding roughly one full dense prefix to the K-sample peak.
        cache.batch_repeat_interleave(count)
        return cache

    @classmethod
    def _gather_prefix_cache_batch(
        cls,
        caches: list[torch.Tensor],
        block_tables: Sequence[torch.Tensor],
        seq_lens: Sequence[int],
    ) -> DynamicCache:
        """Gather distinct paged prefixes directly into one dense batch.

        Gathering every branch into a complete ``DynamicCache`` before
        combining them retains both the six branch caches and the combined
        cache. Constructing the batch one layer at a time keeps only one
        layer's temporary tensors live in addition to the final cache.
        """
        if not block_tables or len(block_tables) != len(seq_lens):
            raise ValueError("Batched prefix block tables and lengths must align")
        max_length = max(int(seq_len) for seq_len in seq_lens)
        combined = DynamicCache()
        for layer_index, paged_layer in enumerate(caches):
            keys: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            for block_table, seq_len in zip(block_tables, seq_lens, strict=True):
                gathered = cls._gather_prefix_cache(
                    [paged_layer],
                    block_table,
                    int(seq_len),
                )
                layer = gathered.layers[0]
                pad_length = max_length - int(seq_len)
                keys.append(torch.nn.functional.pad(layer.keys, (0, 0, 0, pad_length)))
                values.append(torch.nn.functional.pad(layer.values, (0, 0, 0, pad_length)))
            combined.update(
                torch.cat(keys, dim=0),
                torch.cat(values, dim=0),
                layer_index,
            )
        return combined

    @staticmethod
    def _combine_prefix_caches(caches: Sequence[DynamicCache]) -> DynamicCache:
        """Right-pad distinct VLM prefixes into one expert batch."""
        if not caches:
            raise ValueError("At least one VLM prefix is required")
        max_length = max(cache.get_seq_length() for cache in caches)
        combined = DynamicCache()
        for layer_index in range(len(caches[0].layers)):
            keys: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            for cache in caches:
                layer = cache.layers[layer_index]
                pad_length = max_length - layer.keys.shape[-2]
                keys.append(torch.nn.functional.pad(layer.keys, (0, 0, 0, pad_length)))
                values.append(torch.nn.functional.pad(layer.values, (0, 0, 0, pad_length)))
            combined.update(
                torch.cat(keys, dim=0).contiguous(),
                torch.cat(values, dim=0).contiguous(),
                layer_index,
            )
        return combined

    @staticmethod
    def _observation_tensor(value: Any, *, device: torch.device) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=device, dtype=torch.float32)
        while tensor.ndim > 1 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        return tensor

    @staticmethod
    def _action_boundary_position(
        positions: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Return the real terminal position, excluding graph padding."""
        if positions.ndim == 2 and positions.shape[0] == 3:
            return positions[:, :1].to(device)
        return positions.reshape(-1)[:1].repeat(3, 1).to(device)

    def _make_static_action_cache(
        self,
        cache: DynamicCache,
        *,
        suffix_length: int,
        max_cache_len: int | None,
    ) -> StaticCache:
        prefix_length = cache.get_seq_length()
        minimum_length = prefix_length + suffix_length
        effective_max_len = max_cache_len or minimum_length
        if effective_max_len < minimum_length:
            raise ValueError(
                f"static action cache is too short: need at least {minimum_length} tokens, got {effective_max_len}"
            )
        static_cache = StaticCache(
            config=self.expert.config.llm_config,
            max_cache_len=effective_max_len,
        )
        cache_position = torch.arange(
            prefix_length,
            device=cache.layers[0].keys.device,
            dtype=torch.long,
        )
        for layer_index, layer in enumerate(cache.layers):
            static_cache.update(
                layer.keys,
                layer.values,
                layer_index,
                cache_kwargs={"cache_position": cache_position},
            )
        return static_cache

    @staticmethod
    def _copy_action_prefix(
        dst: StaticCache,
        src: DynamicCache,
        prefix_length: int,
    ) -> None:
        for dst_layer, src_layer in zip(dst.layers, src.layers, strict=True):
            dst_layer.keys[:, :, :prefix_length].copy_(src_layer.keys)
            dst_layer.values[:, :, :prefix_length].copy_(src_layer.values)
            dst_layer.cumulative_length.fill_(prefix_length)

    @staticmethod
    def _rewind_static_action_cache(
        cache: StaticCache,
        prefix_length: int | torch.Tensor,
    ) -> None:
        # Each expert call overwrites the complete action suffix, so only the
        # write cursor needs to be reset between flow-matching steps.
        for layer in cache.layers:
            if isinstance(prefix_length, torch.Tensor):
                layer.cumulative_length.copy_(prefix_length)
            else:
                layer.cumulative_length.fill_(prefix_length)

    @staticmethod
    def _make_action_attention_mask(
        *,
        sample_count: int,
        prefix_length: int,
        suffix_length: int,
        max_cache_len: int | None,
        device: torch.device,
        dtype: torch.dtype,
        prefix_lengths: Sequence[int] | None = None,
    ) -> torch.Tensor:
        minimum_length = prefix_length + suffix_length
        mask_length = max_cache_len or minimum_length
        if mask_length < minimum_length:
            raise ValueError(
                f"static action cache is too short: need at least {minimum_length} tokens, got {mask_length}"
            )
        attention_mask = torch.full(
            (sample_count, 1, suffix_length, mask_length),
            torch.finfo(dtype).min,
            device=device,
            dtype=dtype,
        )
        if prefix_lengths is None:
            attention_mask[..., :minimum_length] = 0
        else:
            if len(prefix_lengths) != sample_count:
                raise ValueError("Expert prefix lengths must align with the action batch")
            for index, valid_prefix_length in enumerate(prefix_lengths):
                attention_mask[index, ..., :valid_prefix_length] = 0
                attention_mask[
                    index,
                    ...,
                    prefix_length : prefix_length + suffix_length,
                ] = 0
        return attention_mask

    def _run_manual_action_graph(
        self,
        *,
        expert: nn.Module,
        prefix: DynamicCache,
        action: torch.Tensor,
        expert_positions: torch.Tensor,
        attention_mask: torch.Tensor,
        inference_steps: int,
        suffix_length: int,
        static_cache_max_len: int | None,
    ) -> torch.Tensor:
        """Replay the fixed-shape ten-step expert integration in one graph."""

        prefix_length = prefix.get_seq_length()
        key = (
            tuple(action.shape),
            action.dtype,
            action.device,
            tuple(expert_positions.shape),
            tuple(attention_mask.shape),
            inference_steps,
        )
        state = self._action_graphs.get(key)
        if state is None:
            static_cache = self._make_static_action_cache(
                prefix,
                suffix_length=suffix_length,
                max_cache_len=static_cache_max_len,
            )
            state = {
                "cache": static_cache,
                "action": action.clone(),
                "positions": expert_positions.clone(),
                "attention_mask": attention_mask.clone(),
                "cache_position": torch.arange(
                    prefix_length,
                    prefix_length + suffix_length,
                    device=action.device,
                    dtype=torch.long,
                ),
            }

            def graph_body() -> torch.Tensor:
                graph_action = state["action"]
                for step in range(inference_steps):
                    timestep = graph_action.new_full(
                        (graph_action.shape[0],) + (1,) * (graph_action.ndim - 1),
                        step / inference_steps,
                    )
                    with torch.autocast(
                        device_type="cuda",
                        dtype=self.expert.action_out_proj.weight.dtype,
                    ):
                        embeddings = self.expert.action_in_proj(graph_action, timestep)
                    output = expert(
                        inputs_embeds=embeddings.to(self.expert.action_out_proj.weight.dtype),
                        position_ids=state["positions"],
                        past_key_values=static_cache,
                        attention_mask=state["attention_mask"],
                        cache_position=state["cache_position"],
                        use_cache=True,
                        is_causal=not bool(self.expert.config.expert_non_causal_attention),
                    )
                    self._rewind_static_action_cache(
                        static_cache,
                        state["cache_position"][0],
                    )
                    velocity = self.expert.action_out_proj(output.last_hidden_state[:, -suffix_length:])
                    graph_action = graph_action + velocity.float().view_as(graph_action) / inference_steps
                return graph_action

            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                for _ in range(2):
                    graph_body()
                    self._rewind_static_action_cache(
                        static_cache,
                        state["cache_position"][0],
                    )
            torch.cuda.current_stream().wait_stream(warmup_stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_output = graph_body()
            state["graph"] = graph
            state["output"] = graph_output
            self._action_graphs[key] = state

        state["action"].copy_(action)
        state["positions"].copy_(expert_positions)
        state["attention_mask"].copy_(attention_mask)
        state["cache_position"].copy_(
            torch.arange(
                prefix_length,
                prefix_length + suffix_length,
                device=action.device,
                dtype=torch.long,
            )
        )
        self._copy_action_prefix(state["cache"], prefix, prefix_length)
        state["graph"].replay()
        return state["output"].clone()

    def _sample_actions_batch(
        self,
        *,
        caches: list[torch.Tensor],
        block_tables: Sequence[torch.Tensor],
        seq_lens: Sequence[int],
        positions: Sequence[torch.Tensor],
        observations: Sequence[Mapping[str, Any]],
        extra_args: Sequence[Mapping[str, Any]],
    ) -> dict[str, torch.Tensor]:
        target_cache_layers = int(self.expert.config.llm_config.num_hidden_layers)
        if len(caches) < target_cache_layers:
            raise RuntimeError(
                "Alpamayo action generation received fewer target KV-cache "
                f"layers than expected: got {len(caches)}, expected {target_cache_layers}"
            )
        # vLLM 0.28 appends speculative-draft caches after the target model's
        # layers. The action expert is conditioned only on the target prefix.
        caches = caches[:target_cache_layers]
        device = caches[0].device
        sample_count = len(observations)
        if not (sample_count == len(block_tables) == len(seq_lens) == len(positions) == len(extra_args)):
            raise ValueError("Batched expert inputs must have matching lengths")
        if sample_count < 1:
            raise ValueError("The action expert batch must not be empty")
        first_extra = extra_args[0]
        profile_action = bool(first_extra.get("_profile_action", False)) and device.type == "cuda"
        profile_events: list[torch.cuda.Event] = []

        def record_profile_event() -> None:
            if profile_action:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                profile_events.append(event)

        record_profile_event()
        if any(int(extra.get("num_traj_samples", 1)) != 1 for extra in extra_args):
            raise ValueError("Each parallel VLM branch must request one expert trajectory")
        inference_steps = int(first_extra.get("diffusion_steps", 10))
        temperature = float(first_extra.get("action_temperature", 1.0))
        if inference_steps < 1:
            raise ValueError("num_traj_samples and diffusion_steps must be positive")
        for extra in extra_args[1:]:
            if (
                int(extra.get("diffusion_steps", 10)) != inference_steps
                or float(extra.get("action_temperature", 1.0)) != temperature
            ):
                raise ValueError("Batched expert branches must use identical diffusion settings")

        prefix_lengths = [int(seq_len) for seq_len in seq_lens]
        prefix = self._gather_prefix_cache_batch(caches, block_tables, seq_lens)
        if bool(first_extra.get("_batch_action_expert", True)) is False and sample_count > 1:
            raise ValueError("Sequential expert mode must submit one VLM branch per expert call")
        record_profile_event()
        prefix_len = prefix.get_seq_length()
        action_dims = tuple(int(dim) for dim in self.expert.action_space.get_action_space_dims())
        suffix_length = action_dims[0]
        static_cache_max_len_value = first_extra.get("_static_expert_cache_max_len")
        static_cache_max_len = int(static_cache_max_len_value) if static_cache_max_len_value is not None else None
        manual_action_cudagraph = bool(first_extra.get("_manual_action_cudagraph", False))
        static_expert_cache = bool(first_extra.get("_static_expert_cache", manual_action_cudagraph))
        if static_expert_cache and static_cache_max_len is not None:
            # The configured length is the reusable latency profile, not a
            # hard request limit. Longer prompts remain correct and simply
            # create a second graph shape.
            static_cache_max_len = max(
                static_cache_max_len,
                prefix_len + suffix_length,
            )

        noise_rows: list[torch.Tensor] = []
        for extra in extra_args:
            generator = None
            if (sampling_seed := extra.get("_sampling_seed")) is not None:
                generator = torch.Generator(device=device)
                generator.manual_seed(int(sampling_seed))
            noise_rows.append(
                torch.randn(
                    1,
                    *action_dims,
                    device=device,
                    dtype=torch.float32,
                    generator=generator,
                )
            )
        action = torch.cat(noise_rows, dim=0)
        initial_noise = (
            action.clone() if any(bool(extra.get("_return_action_noise", False)) for extra in extra_args) else None
        )
        action = action * temperature

        last_position = torch.cat(
            [self._action_boundary_position(value, device=device) for value in positions],
            dim=1,
        )
        expert_positions = (
            last_position[:, :, None] + 1 + torch.arange(suffix_length, device=device)[None, None, :]
        ).expand(3, sample_count, suffix_length)
        weight_dtype = self.expert.action_out_proj.weight.dtype
        attention_mask = self._make_action_attention_mask(
            sample_count=sample_count,
            prefix_length=prefix_len,
            suffix_length=suffix_length,
            max_cache_len=(static_cache_max_len if static_expert_cache else None),
            device=device,
            dtype=weight_dtype,
            prefix_lengths=prefix_lengths,
        )
        expert_cache: DynamicCache | StaticCache = prefix
        expert_cache_position = None
        if static_expert_cache and not manual_action_cudagraph:
            expert_cache = self._make_static_action_cache(
                prefix,
                suffix_length=suffix_length,
                max_cache_len=static_cache_max_len,
            )
            expert_cache_position = torch.arange(
                prefix_len,
                prefix_len + suffix_length,
                device=device,
                dtype=torch.long,
            )
        record_profile_event()

        expert = self.expert.expert
        if bool(first_extra.get("_compile_expert", False)):
            if self._compiled_expert is None:
                self._compiled_expert = torch.compile(
                    self.expert.expert,
                    fullgraph=False,
                    dynamic=False,
                    options={"triton.cudagraphs": False},
                )
            expert = self._compiled_expert
        if manual_action_cudagraph:
            action = self._run_manual_action_graph(
                expert=expert,
                prefix=prefix,
                action=action,
                expert_positions=expert_positions,
                attention_mask=attention_mask,
                inference_steps=inference_steps,
                suffix_length=suffix_length,
                static_cache_max_len=static_cache_max_len,
            )
        else:
            for step in range(inference_steps):
                timestep = action.new_full(
                    (sample_count,) + (1,) * len(action_dims),
                    step / inference_steps,
                )
                with torch.autocast(device_type=device.type, dtype=weight_dtype):
                    embeddings = self.expert.action_in_proj(action, timestep)
                outputs = expert(
                    inputs_embeds=embeddings.to(weight_dtype),
                    position_ids=expert_positions,
                    past_key_values=expert_cache,
                    attention_mask=attention_mask,
                    cache_position=expert_cache_position,
                    use_cache=True,
                    is_causal=not bool(self.expert.config.expert_non_causal_attention),
                )
                if isinstance(expert_cache, StaticCache):
                    self._rewind_static_action_cache(expert_cache, prefix_len)
                else:
                    expert_cache.crop(prefix_len)
                velocity = self.expert.action_out_proj(outputs.last_hidden_state[:, -suffix_length:])
                action = action + velocity.float().view(sample_count, *action_dims) / inference_steps
        record_profile_event()

        history_xyz_rows = [
            self._observation_tensor(observation["ego_history_xyz"], device=device) for observation in observations
        ]
        history_rot_rows = [
            self._observation_tensor(observation["ego_history_rot"], device=device) for observation in observations
        ]
        if any(value.ndim != 2 or value.shape[-1] != 3 for value in history_xyz_rows):
            raise ValueError("Every ego_history_xyz must have shape (T, 3)")
        if any(value.ndim != 3 or value.shape[-2:] != (3, 3) for value in history_rot_rows):
            raise ValueError("Every ego_history_rot must have shape (T, 3, 3)")
        repeated_history_xyz = torch.stack(history_xyz_rows)
        repeated_history_rot = torch.stack(history_rot_rows)
        pred_xyz, pred_rot = self.expert.action_space.action_to_traj(
            action,
            repeated_history_xyz,
            repeated_history_rot,
        )
        record_profile_event()
        result = {
            "pred_trajectories": pred_xyz,
            "pred_rotations": pred_rot,
            "actions": pred_xyz,
            "rotations": pred_rot,
            "normalized_controls": action,
        }
        if profile_action:
            # Synchronize only for explicitly profiled requests. The four
            # intervals are KV materialization, action setup, diffusion, and
            # action-space trajectory conversion respectively.
            profile_events[-1].synchronize()
            result["action_profile_ms"] = torch.tensor(
                [
                    profile_events[index].elapsed_time(profile_events[index + 1])
                    for index in range(len(profile_events) - 1)
                ],
                device=device,
                dtype=torch.float32,
            )
        if initial_noise is not None:
            result["action_noise"] = initial_noise
        result["action_expert_invocation_batch_size"] = torch.full(
            (sample_count,),
            sample_count,
            device=device,
            dtype=torch.int32,
        )
        return result

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
        """Compatibility wrapper for one sequential expert invocation."""
        return self._sample_actions_batch(
            caches=caches,
            block_tables=[block_table],
            seq_lens=[seq_len],
            positions=[positions],
            observations=[observation],
            extra_args=[extra_args],
        )

    @staticmethod
    def _parallel_group_info(
        request_id: str,
        extra_args: Mapping[str, Any],
    ) -> tuple[str, int, int]:
        expected = int(extra_args.get("_parallel_sample_count", 1))
        if expected <= 1:
            return request_id, 0, 1
        child_index, separator, parent_id = request_id.partition("_")
        if not separator or not child_index.isdigit() or not parent_id:
            raise RuntimeError("Parallel Alpamayo request IDs must use vLLM's '<index>_<parent>' form")
        index = int(child_index)
        if index >= expected:
            raise RuntimeError("Parallel Alpamayo child index exceeds its sample count")
        return parent_id, index, expected

    @staticmethod
    def _split_batched_policy_output(
        output: Mapping[str, torch.Tensor],
        index: int,
    ) -> dict[str, torch.Tensor]:
        return {
            key: value[index : index + 1] if key in _BATCHED_POLICY_OUTPUT_KEYS else value
            for key, value in output.items()
        }

    @staticmethod
    def _concat_batched_policy_outputs(
        outputs: Sequence[Mapping[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Join independently evaluated expert microbatches in sample order."""
        if not outputs:
            raise ValueError("At least one action-expert output is required")
        keys = set(outputs[0])
        if any(set(output) != keys for output in outputs[1:]):
            raise RuntimeError("Action-expert microbatches returned different fields")
        combined: dict[str, torch.Tensor] = {}
        for key in keys:
            values = [output[key] for output in outputs]
            if key in _BATCHED_POLICY_OUTPUT_KEYS:
                combined[key] = torch.cat(values, dim=0)
            elif key == "action_profile_ms":
                combined[key] = torch.stack(values).sum(dim=0)
            else:
                combined[key] = values[0]
        return combined

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
        hidden_states: torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]],
        *,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor,
        sampling_extra_args: object = None,
        runner_kv_cache_context: RunnerKVCacheContext | None = None,
        request_token_spans: Sequence[tuple[int, int]] | None = None,
        **_: Any,
    ) -> IntermediateTensors | OmniOutput:
        if isinstance(hidden_states, IntermediateTensors):
            return hidden_states
        observations = self._policy_observations(sampling_extra_args)
        # The released OSS model does not allow an ordinary text EOS to end
        # trajectory reasoning. Generation must continue until future_start,
        # which transfers control to the action expert. Keep this per request
        # so text-only tasks retain their normal EOS behavior in mixed batches.
        self._mask_text_eos_indices = tuple(
            index for index, observation in enumerate(observations) if observation is not None
        )
        multimodal_outputs: dict[str, list[torch.Tensor | None]] = {}
        triggered_indices: list[int] = []
        if observations and input_ids is not None:
            request_count = len(observations)
            if request_token_spans is None:
                if request_count != 1:
                    raise RuntimeError("Super batched policy inference requires request_token_spans")
                request_token_spans = [(0, input_ids.numel())]
            if len(request_token_spans) != request_count:
                raise RuntimeError("Super request token spans and runner requests must align")

            flat_input_ids = input_ids.reshape(-1)
            extras = sampling_extra_args if isinstance(sampling_extra_args, list) else []
            trigger_mask = [
                observation is not None and end > start and int(flat_input_ids[start]) == self.future_start_id
                for observation, (start, end) in zip(observations, request_token_spans, strict=True)
            ]
            if any(trigger_mask):
                if runner_kv_cache_context is None:
                    raise RuntimeError("Super action generation requires runner KV-cache access")
                if len(runner_kv_cache_context.request_ids) != request_count:
                    raise RuntimeError("Super policy arguments and runner requests must align")
            per_request_outputs: list[dict[str, torch.Tensor] | None] = [None] * request_count
            touched_groups: set[str] = set()
            for index, (observation, span) in enumerate(zip(observations, request_token_spans, strict=True)):
                start, end = span
                # A speculative verification block can contain an unaccepted
                # future_start after its first token. The accepted boundary is
                # fed back as the first token of a draft-free request span.
                if not trigger_mask[index]:
                    continue
                assert observation is not None
                assert runner_kv_cache_context is not None
                extra = extras[index]
                if not isinstance(extra, Mapping):
                    raise RuntimeError("Super policy sampling arguments must be mappings")
                request_id = runner_kv_cache_context.request_ids[index]
                group_id, child_index, expected = self._parallel_group_info(
                    request_id,
                    extra,
                )
                if expected == 1 or not bool(extra.get("_batch_action_expert", True)):
                    per_request_outputs[index] = self._sample_actions(
                        caches=runner_kv_cache_context.caches,
                        block_table=runner_kv_cache_context.block_table[index],
                        seq_len=runner_kv_cache_context.sequence_lengths[index],
                        positions=self._request_positions(positions, start, end),
                        observation=observation,
                        extra_args=extra,
                    )
                    triggered_indices.append(index)
                    continue

                group = self._pending_policy_groups.setdefault(group_id, {})
                group.setdefault(
                    request_id,
                    {
                        "child_index": child_index,
                        "expected": expected,
                        "block_table": runner_kv_cache_context.block_table[index].clone(),
                        "seq_len": runner_kv_cache_context.sequence_lengths[index],
                        "positions": self._request_positions(positions, start, end).clone(),
                        "observation": observation,
                        "extra_args": extra,
                    },
                )
                touched_groups.add(group_id)

            request_index_by_id = (
                {request_id: index for index, request_id in enumerate(runner_kv_cache_context.request_ids)}
                if runner_kv_cache_context is not None
                else {}
            )
            waiting_indices: set[int] = set()
            for group_id in touched_groups:
                group = self._pending_policy_groups[group_id]
                expected_values = {int(entry["expected"]) for entry in group.values()}
                if len(expected_values) != 1:
                    raise RuntimeError("Parallel Alpamayo children disagree on sample count")
                expected = expected_values.pop()
                all_members_scheduled = all(request_id in request_index_by_id for request_id in group)
                if len(group) < expected or not all_members_scheduled:
                    waiting_indices.update(
                        request_index_by_id[request_id] for request_id in group if request_id in request_index_by_id
                    )
                    continue

                ordered = sorted(group.items(), key=lambda item: int(item[1]["child_index"]))
                configured_batch_sizes = {
                    int(entry["extra_args"].get("_action_expert_max_batch_size", expected)) for _, entry in ordered
                }
                if len(configured_batch_sizes) != 1:
                    raise RuntimeError("Parallel Alpamayo children disagree on expert batch size")
                max_batch_size = configured_batch_sizes.pop()
                if max_batch_size < 1:
                    raise ValueError("action_expert_max_batch_size must be positive")
                microbatch_outputs = []
                for begin in range(0, len(ordered), max_batch_size):
                    chunk = ordered[begin : begin + max_batch_size]
                    microbatch_outputs.append(
                        self._sample_actions_batch(
                            caches=runner_kv_cache_context.caches,
                            block_tables=[entry["block_table"] for _, entry in chunk],
                            seq_lens=[int(entry["seq_len"]) for _, entry in chunk],
                            positions=[entry["positions"] for _, entry in chunk],
                            observations=[entry["observation"] for _, entry in chunk],
                            extra_args=[entry["extra_args"] for _, entry in chunk],
                        )
                    )
                batched_output = self._concat_batched_policy_outputs(microbatch_outputs)
                for batch_index, (request_id, _) in enumerate(ordered):
                    request_index = request_index_by_id[request_id]
                    per_request_outputs[request_index] = self._split_batched_policy_output(
                        batched_output,
                        batch_index,
                    )
                    triggered_indices.append(request_index)
                del self._pending_policy_groups[group_id]

            output_keys = {key for output in per_request_outputs if output is not None for key in output}
            multimodal_outputs = {
                key: [output.get(key) if output is not None else None for output in per_request_outputs]
                for key in output_keys
            }
            self._force_future_end_indices = tuple(triggered_indices)
            self._force_future_start_indices = tuple(sorted(waiting_indices))
        if isinstance(hidden_states, list | tuple):
            text_hidden_states = hidden_states[0]
            aux_hidden_states = hidden_states[1]
        else:
            text_hidden_states = hidden_states
            aux_hidden_states = None
        return OmniOutput(
            text_hidden_states=text_hidden_states,
            multimodal_outputs=multimodal_outputs,
            aux_hidden_states=aux_hidden_states,
        )

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput) -> torch.Tensor:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        logits = super().compute_logits(hidden_states)
        force_end_indices = self._force_future_end_indices
        force_start_indices = self._force_future_start_indices
        mask_text_eos_indices = self._mask_text_eos_indices
        self._force_future_end_indices = ()
        self._force_future_start_indices = ()
        self._mask_text_eos_indices = ()
        traj_ids = self.alpamayo_config.traj_ids
        start = min(int(traj_ids["history_id0"]), int(traj_ids["future_id0"]))
        logits[..., start : start + int(self.alpamayo_config.traj_vocab_size)] = -torch.inf
        if mask_text_eos_indices and self._text_eos_token_ids:
            indices = torch.as_tensor(mask_text_eos_indices, device=logits.device, dtype=torch.long)
            logits[
                indices[:, None],
                torch.as_tensor(self._text_eos_token_ids, device=logits.device, dtype=torch.long),
            ] = -torch.inf
        for force_indices, token_id in (
            (force_start_indices, self.future_start_id),
            (force_end_indices, self.future_end_id),
        ):
            if not force_indices:
                continue
            indices = torch.as_tensor(force_indices, device=logits.device, dtype=torch.long)
            logits[indices] = -torch.inf
            logits[indices, token_id] = 0
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
