# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Alpamayo 2 Super reasoning stage: Qwen3-VL chain-of-thought up to the trajectory.

The Qwen3-VL decoder runs natively in vLLM and generates the driving
reasoning. When it emits ``<|traj_future_start|>``, the next forward writes
that token's KV and ``compute_logits`` forces ``<|traj_future_end|>``, so the
request stops with its prefix complete. The request's KV is then handed to the
expert stage (``Alpamayo2SuperPipeline``) with the prefix length and the
trigger token's M-RoPE position as KV-transfer metadata.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
    Qwen3VLProcessor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY


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


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Alpamayo2SuperProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Alpamayo2SuperForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """Reasoning stage of the Alpamayo 2 Super policy.

    Every request is a trajectory request: it must end with
    ``<|traj_future_start|>``, so text EOS and the trajectory vocabulary are
    never sampled. The request stops on ``<|traj_future_end|>``.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        self.future_start_id = int(config.traj_ids["future_start"])
        self.future_end_id = int(config.traj_ids["future_end"])
        self.traj_vocab_start = min(int(config.traj_ids["history_id0"]), int(config.traj_ids["future_id0"]))
        self.traj_vocab_size = int(config.traj_vocab_size)

        eos_ids = vllm_config.model_config.try_get_generation_config().get("eos_token_id")
        if eos_ids is None:
            eos_ids = config.text_config.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        self.text_eos_ids = torch.tensor(
            [int(token) for token in eos_ids or () if int(token) != self.future_start_id], dtype=torch.long
        )

        # Rows of this step whose request fed back future_start; compute_logits ends them.
        self._finished_rows: list[int] = []
        # req_id -> KV-transfer metadata, published once the request's prefix is complete.
        self._kv_transfer_metadata: dict[str, dict[str, Any]] = {}

    def prepare_runner_inputs(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        req_ids: Sequence[str],
        num_computed_tokens: Sequence[int],
        num_scheduled_tokens: Sequence[int],
        input_ids_buffer: torch.Tensor,
        **_: Any,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        # future_start is sampled at the end of one step and fed as the first
        # input of the next, whose forward writes its KV.
        self._finished_rows = []
        token_ids = input_ids if input_ids is not None else input_ids_buffer
        counts = [int(n) for n in num_scheduled_tokens]
        starts = [sum(counts[:index]) for index in range(len(counts))]
        rows = [index for index, count in enumerate(counts) if count > 0]
        if rows:
            first_ids = token_ids[torch.tensor([starts[row] for row in rows], device=token_ids.device)].tolist()
            for row, token_id in zip(rows, first_ids, strict=True):
                if token_id != self.future_start_id:
                    continue
                self._finished_rows.append(row)
                self._kv_transfer_metadata[req_ids[row]] = {
                    "prefix_len": int(num_computed_tokens[row]) + counts[row],
                    "trigger_position": self._position_at(positions, starts[row]),
                }
        return input_ids, positions

    @staticmethod
    def _position_at(positions: torch.Tensor, index: int) -> list[int]:
        """M-RoPE ``(3,)`` position of one flattened token."""
        if positions.ndim == 2:
            return [int(p) for p in positions[:, index].tolist()]
        return [int(positions[index])] * 3

    def get_kv_transfer_metadata(self, req_id: str, *, num_computed_tokens: int | None = None) -> dict[str, Any] | None:
        """Prefix length and ``<|traj_future_start|>`` position, sent with the request's KV."""
        return self._kv_transfer_metadata.pop(req_id, None)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = super().compute_logits(hidden_states)
        if logits is None:
            return None
        # Trajectory tokens are produced by the expert stage, never sampled,
        # and reasoning must end with future_start rather than a text EOS.
        logits[..., self.traj_vocab_start : self.traj_vocab_start + self.traj_vocab_size] = -torch.inf
        if self.text_eos_ids.numel():
            logits[..., self.text_eos_ids.to(logits.device, non_blocking=True)] = -torch.inf
        finished_rows, self._finished_rows = self._finished_rows, []
        if finished_rows:
            rows = torch.tensor(finished_rows, device=logits.device)
            logits[rows] = -torch.inf
            logits[rows, self.future_end_id] = 0
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        vlm_weights: list[tuple[str, torch.Tensor]] = []
        for name, weight in weights:
            if name.startswith("vlm."):
                vlm_weights.append((name.removeprefix("vlm."), weight))
            elif not name.startswith("expert."):  # the expert stage loads expert.*
                raise ValueError(f"Unexpected Alpamayo 2 Super weight {name!r}: expected a 'vlm.' or 'expert.' prefix")
        return super().load_weights(vlm_weights)
