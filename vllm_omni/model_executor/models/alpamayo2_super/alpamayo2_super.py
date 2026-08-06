# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native Qwen3-VL portion of the Alpamayo 2 Super integration."""

from __future__ import annotations

from collections.abc import Iterable
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
from vllm.transformers_utils.processor import cached_get_processor


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
    """Load and execute the nested Super Qwen3-VL backbone natively.

    This first checkpoint makes VQA and policy-prefix execution available and
    establishes the correct nested-weight mapping. The terminal action-expert
    hook will be added next; until then policy calls intentionally return no
    structured trajectory instead of silently using the incompatible 1.5
    expert implementation.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.alpamayo_config = vllm_config.model_config.hf_config

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = super().compute_logits(hidden_states)
        traj_ids = self.alpamayo_config.traj_ids
        start = min(int(traj_ids["history_id0"]), int(traj_ids["future_id0"]))
        logits[..., start : start + int(self.alpamayo_config.traj_vocab_size)] = -torch.inf
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Super wraps the backbone under ``vlm``. Expert weights are left for
        # the model-owned terminal hook rather than loaded into an HF duplicate.
        return set(
            super().load_weights(
                (name.removeprefix("vlm."), weight)
                for name, weight in weights
                if name.startswith("vlm.")
            )
        )
