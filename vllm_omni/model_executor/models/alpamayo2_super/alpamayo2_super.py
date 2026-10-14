# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Alpamayo 2 Super: Qwen3-VL reasoning followed by a flow-matching trajectory expert.

The Qwen3-VL decoder runs natively in vLLM and generates chain-of-thought
reasoning. When it emits ``<|traj_future_start|>``, the next forward writes
that token's KV and this model gathers the request's paged KV prefix, runs the
non-causal action expert over it, and returns the decoded ego trajectory as
multimodal output. ``compute_logits`` then forces ``<|traj_future_end|>`` so
the request stops.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from transformers import DynamicCache
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
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

from vllm_omni.model_executor.models.alpamayo2_super.action_expert import Alpamayo2SuperActionExpert
from vllm_omni.model_executor.models.output_templates import OmniOutput

# Multimodal output keys: "actions" holds the (64, 3) ego positions, under the
# key robot policies use for their actions, and "rotations" the matching
# (64, 3, 3) rotation matrices.
ACTIONS_KEY = "actions"
ROTATIONS_KEY = "rotations"


class Alpamayo2SuperProcessingInfo(Qwen3VLProcessingInfo):
    """Resize camera frames to the pixel budget the checkpoint was trained with."""

    def get_hf_processor(self, **kwargs: object) -> Qwen3VLProcessor:
        config = self.get_hf_config()
        return self.ctx.get_hf_processor(
            Qwen3VLProcessor,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            **kwargs,
        )


@dataclass
class _StepInputs:
    """Per-step runner inputs captured before the forward."""

    input_ids: torch.Tensor
    positions: torch.Tensor
    num_computed_tokens: list[int]
    num_scheduled_tokens: list[int]
    extra_args: list[Mapping[str, Any]]


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Alpamayo2SuperProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Alpamayo2SuperForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """Alpamayo 2 Super policy.

    A request opts into trajectory generation by passing ``robot_obs`` in
    ``SamplingParams.extra_args`` with ``ego_history_xyz`` ``(T, 3)`` and
    ``ego_history_rot`` ``(T, 3, 3)``. Optional ``diffusion_steps`` (default
    10) and ``action_seed`` control the expert's sampling. The request must
    stop on ``<|traj_future_end|>``.

    The expert reads the request's KV from the paged cache using the
    attention block table, so the stage must use ``FLASH_ATTN`` or ``TRITON_ATTN``.
    """

    have_multimodal_outputs = True
    accepts_runner_sampling_extra_args = True
    omni_client_multimodal_output_keys = (ACTIONS_KEY, ROTATIONS_KEY)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        self.future_start_id = int(config.traj_ids["future_start"])
        self.future_end_id = int(config.traj_ids["future_end"])
        self.traj_vocab_start = min(int(config.traj_ids["history_id0"]), int(config.traj_ids["future_id0"]))
        self.traj_vocab_size = int(config.traj_vocab_size)
        self.expert = Alpamayo2SuperActionExpert(config.expert_config)

        # Trajectory reasoning must end with future_start, never a text EOS.
        eos_ids = vllm_config.model_config.try_get_generation_config().get("eos_token_id")
        if eos_ids is None:
            eos_ids = config.text_config.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        self.text_eos_ids = [int(token) for token in eos_ids or () if int(token) != self.future_start_id]

        self._step: _StepInputs | None = None
        self._policy_rows: list[int] = []
        self._finished_rows: list[int] = []

    def prepare_runner_inputs(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        req_ids: Sequence[str],
        num_computed_tokens: Sequence[int],
        num_scheduled_tokens: Sequence[int],
        input_ids_buffer: torch.Tensor,
        sampling_extra_args: Sequence[Mapping[str, Any]],
        **_: Any,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        # Multimodal steps pass embeddings; the token ids stay in the buffer.
        self._step = _StepInputs(
            input_ids=input_ids if input_ids is not None else input_ids_buffer,
            positions=positions,
            num_computed_tokens=[int(n) for n in num_computed_tokens],
            num_scheduled_tokens=[int(n) for n in num_scheduled_tokens],
            extra_args=list(sampling_extra_args),
        )
        return input_ids, positions

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | IntermediateTensors,
        **_: Any,
    ) -> OmniOutput | IntermediateTensors:
        if isinstance(model_outputs, IntermediateTensors):
            return model_outputs
        step, self._step = self._step, None
        self._policy_rows = []
        self._finished_rows = []
        outputs: dict[str, list[torch.Tensor | None]] = {}
        if step is not None:
            num_requests = len(step.num_scheduled_tokens)
            xyz: list[torch.Tensor | None] = [None] * num_requests
            rot: list[torch.Tensor | None] = [None] * num_requests
            start = 0
            for index, (count, extra) in enumerate(zip(step.num_scheduled_tokens, step.extra_args, strict=True)):
                observation = extra.get("robot_obs") if extra else None
                if observation is not None:
                    self._policy_rows.append(index)
                    # future_start is sampled at the end of one step and fed as
                    # the first input of the next, which writes its KV.
                    if count > 0 and int(step.input_ids[start]) == self.future_start_id:
                        xyz[index], rot[index] = self._sample_trajectory(
                            request_index=index,
                            seq_len=step.num_computed_tokens[index] + count,
                            trigger_position=self._position_at(step.positions, start),
                            observation=observation,
                            extra=extra,
                        )
                        self._finished_rows.append(index)
                start += count
            if self._finished_rows:
                outputs = {ACTIONS_KEY: xyz, ROTATIONS_KEY: rot}
        return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs=outputs)

    @staticmethod
    def _position_at(positions: torch.Tensor, index: int) -> torch.Tensor:
        """M-RoPE ``(3,)`` position of one flattened token."""
        if positions.ndim == 2:
            return positions[:, index]
        return positions[index].repeat(3)

    def _gather_prefix(self, request_index: int, seq_len: int) -> DynamicCache:
        """Copy one request's paged KV into an HF cache, ``[1, kv_heads, seq_len, head_dim]`` per layer."""
        attn_metadata = get_forward_context().attn_metadata
        cache = DynamicCache()
        for layer_index, layer in enumerate(self.language_model.model.layers):
            attn = layer.self_attn.attn
            metadata = attn_metadata[attn.layer_name] if isinstance(attn_metadata, dict) else None
            block_table = getattr(metadata, "block_table", None)
            if block_table is None:
                raise RuntimeError("Alpamayo 2 Super requires the FLASH_ATTN or TRITON_ATTN attention backend")
            # FLASH_ATTN and TRITON_ATTN layout: [num_blocks, kv_heads, block_size, 2 * head_dim].
            kv_cache = attn.kv_cache
            if kv_cache.ndim != 4:
                raise RuntimeError(f"Unexpected KV cache shape {tuple(kv_cache.shape)}")
            block_size = kv_cache.shape[2]
            block_ids = block_table[request_index, : (seq_len + block_size - 1) // block_size].long()
            # -> [kv_heads, seq_len, 2 * head_dim]
            kv = kv_cache.index_select(0, block_ids).transpose(0, 1).flatten(1, 2)[:, :seq_len]
            key, value = kv.chunk(2, dim=-1)
            cache.update(key.unsqueeze(0).contiguous(), value.unsqueeze(0).contiguous(), layer_index)
        return cache

    def _sample_trajectory(
        self,
        *,
        request_index: int,
        seq_len: int,
        trigger_position: torch.Tensor,
        observation: Mapping[str, Any],
        extra: Mapping[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Flow-matching (Euler) sampling of one trajectory conditioned on the VLM prefix."""
        expert = self.expert
        device = trigger_position.device
        num_waypoints, action_dim = expert.action_space.get_action_space_dims()
        num_steps = int(extra.get("diffusion_steps", 10))
        prefix = self._gather_prefix(request_index, seq_len)

        position_ids = (trigger_position[:, None, None] + 1 + torch.arange(num_waypoints, device=device)).to(torch.long)
        dtype = expert.action_out_proj.weight.dtype
        attention_mask = torch.zeros((1, 1, num_waypoints, seq_len + num_waypoints), device=device, dtype=dtype)

        generator = None
        if extra.get("action_seed") is not None:
            generator = torch.Generator(device=device).manual_seed(int(extra["action_seed"]))
        action = torch.randn(1, num_waypoints, action_dim, device=device, generator=generator)
        time_steps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        for i in range(num_steps):
            hidden = expert.expert(
                inputs_embeds=expert.action_in_proj(action, time_steps[i].view(1, 1, 1)),
                position_ids=position_ids,
                past_key_values=prefix,
                attention_mask=attention_mask,
                use_cache=True,
                is_causal=not expert.non_causal,
            ).last_hidden_state
            prefix.crop(seq_len)
            velocity = expert.action_out_proj(hidden).view(1, num_waypoints, action_dim)
            # A (1, 1, 1) step keeps the update in float32, as in the reference.
            action = action + (time_steps[i + 1] - time_steps[i]).view(1, 1, 1) * velocity

        history_xyz = torch.as_tensor(observation["ego_history_xyz"], device=device, dtype=torch.float32)
        history_rot = torch.as_tensor(observation["ego_history_rot"], device=device, dtype=torch.float32)
        trajectory_xyz, trajectory_rot = expert.action_space.action_to_traj(
            action, history_xyz[None], history_rot[None]
        )
        return trajectory_xyz[0], trajectory_rot[0]

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        logits = super().compute_logits(hidden_states)
        if logits is None:
            return None
        # Trajectory tokens are produced by the expert, never sampled.
        logits[..., self.traj_vocab_start : self.traj_vocab_start + self.traj_vocab_size] = -torch.inf
        policy_rows, self._policy_rows = self._policy_rows, []
        finished_rows, self._finished_rows = self._finished_rows, []
        if policy_rows and self.text_eos_ids:
            rows = torch.tensor(policy_rows, device=logits.device)
            logits[rows[:, None], torch.tensor(self.text_eos_ids, device=logits.device)] = -torch.inf
        if finished_rows:
            rows = torch.tensor(finished_rows, device=logits.device)
            logits[rows] = -torch.inf
            logits[rows, self.future_end_id] = 0
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        vlm_weights: list[tuple[str, torch.Tensor]] = []
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, weight in weights:
            if name.startswith("vlm."):
                vlm_weights.append((name.removeprefix("vlm."), weight))
            elif name.startswith("expert."):
                param = params[name]
                getattr(param, "weight_loader", default_weight_loader)(param, weight)
                loaded.add(name)
        return loaded | super().load_weights(vlm_weights)
