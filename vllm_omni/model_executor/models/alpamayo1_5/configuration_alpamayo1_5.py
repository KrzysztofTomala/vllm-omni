# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transformers configuration adapter for NVIDIA Alpamayo 1.5."""

from __future__ import annotations

from typing import Any

from transformers import Qwen3VLConfig

from .tokenizer import resolve_backbone_path


class Alpamayo1_5Config(Qwen3VLConfig):
    """Expose Alpamayo's external Qwen3-VL backbone to vLLM.

    Alpamayo checkpoints intentionally keep the backbone configuration in
    ``vlm_name_or_path`` instead of duplicating ``text_config`` and
    ``vision_config`` in their own ``config.json``.  vLLM needs those nested
    configs while constructing its cache and multimodal model, so this class
    resolves them before delegating to :class:`Qwen3VLConfig`.
    """

    model_type = "alpamayo1_5"

    def __init__(
        self,
        *,
        vlm_name_or_path: str = "nvidia/Cosmos-Reason2-8B",
        text_config: dict[str, Any] | None = None,
        vision_config: dict[str, Any] | None = None,
        expert_cfg: dict[str, Any] | None = None,
        action_in_proj_cfg: dict[str, Any] | None = None,
        action_out_proj_cfg: dict[str, Any] | None = None,
        action_space_cfg: dict[str, Any] | None = None,
        diffusion_cfg: dict[str, Any] | None = None,
        hist_traj_tokenizer_cfg: dict[str, Any] | None = None,
        traj_token_ids: dict[str, int] | None = None,
        traj_token_start_idx: int = 151669,
        traj_vocab_size: int = 4000,
        tokens_per_history_traj: int = 48,
        tokens_per_future_traj: int = 128,
        expert_non_causal_attention: bool = True,
        include_camera_ids: bool = True,
        include_frame_nums: bool = True,
        min_pixels: int = 163840,
        max_pixels: int = 196608,
        **kwargs: Any,
    ) -> None:
        vlm_name_or_path = resolve_backbone_path(vlm_name_or_path)
        if text_config is None or vision_config is None:
            backbone = Qwen3VLConfig.from_pretrained(vlm_name_or_path)
            text_config = text_config or backbone.text_config.to_dict()
            vision_config = vision_config or backbone.vision_config.to_dict()
            for field in (
                "image_token_id",
                "video_token_id",
                "vision_start_token_id",
                "vision_end_token_id",
            ):
                kwargs.setdefault(field, getattr(backbone, field))

        # The checkpoint extends the Cosmos-Reason2 tokenizer vocabulary.
        text_config = dict(text_config)
        text_config["vocab_size"] = int(kwargs.get("vocab_size", 155697))
        super().__init__(
            text_config=text_config,
            vision_config=vision_config,
            **kwargs,
        )

        self.vlm_name_or_path = vlm_name_or_path
        self.expert_cfg = expert_cfg or {}
        self.action_in_proj_cfg = action_in_proj_cfg or {}
        self.action_out_proj_cfg = action_out_proj_cfg or {}
        self.action_space_cfg = action_space_cfg or {}
        self.diffusion_cfg = diffusion_cfg or {}
        self.hist_traj_tokenizer_cfg = hist_traj_tokenizer_cfg or {}
        self.traj_token_ids = traj_token_ids or {}
        self.traj_token_start_idx = traj_token_start_idx
        self.traj_vocab_size = traj_vocab_size
        self.tokens_per_history_traj = tokens_per_history_traj
        self.tokens_per_future_traj = tokens_per_future_traj
        self.expert_non_causal_attention = expert_non_causal_attention
        self.include_camera_ids = include_camera_ids
        self.include_frame_nums = include_frame_nums
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
