# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""OpenPI request adapter for Alpamayo 2 Super."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from vllm import SamplingParams
from vllm.config import ModelConfig

from vllm_omni.entrypoints.openpi.serving import OpenPIEngineRequest
from vllm_omni.model_executor.models.alpamayo2_super.processing import (
    build_alpamayo2_super_tokenizer,
    build_trajectory_prompt,
)

# Sampling settings of the reference trajectory inference.
_TEMPERATURE = 0.6
_TOP_P = 0.98
_MAX_TOKENS = 256


class Alpamayo2SuperOpenPIAdapter:
    """Build one trajectory request per OpenPI observation.

    Observation keys: ``image_frames`` ``(cameras, frames, 3, H, W)`` uint8,
    sorted ``camera_indices``, ``ego_history_xyz`` ``(16, 3)`` and
    ``ego_history_rot`` ``(16, 3, 3)``; optional ``seed`` and
    ``sampling_params.diffusion_steps``. The reply is the model's ``actions``
    output: the ``(64, 3)`` ego trajectory.
    """

    def __init__(self, model_config: ModelConfig) -> None:
        self.config = model_config.hf_config
        self.tokenizer = build_alpamayo2_super_tokenizer(
            model_config.tokenizer,
            self.config.history_vocab_size,
            self.config.future_vocab_size,
        )

    def build_request(
        self,
        obs: Mapping[str, Any],
        *,
        request_id: str,
        session_id: str,
        reset: bool,
    ) -> OpenPIEngineRequest:
        del session_id, reset  # The policy is stateless across observations.
        frames = torch.as_tensor(np.asarray(obs["image_frames"]))
        if frames.ndim != 5:
            raise ValueError(f"image_frames must be (cameras, frames, 3, H, W), got {tuple(frames.shape)}")
        history_xyz = np.asarray(obs["ego_history_xyz"], dtype=np.float32).reshape(16, 3)
        history_rot = np.asarray(obs["ego_history_rot"], dtype=np.float32).reshape(16, 3, 3)
        prompt_ids = build_trajectory_prompt(
            self.tokenizer,
            self.config,
            frames=frames,
            camera_indices=np.asarray(obs["camera_indices"]).tolist(),
            ego_history_xyz=torch.from_numpy(history_xyz),
        )
        seed = obs.get("seed")
        extra_args: dict[str, Any] = {
            "robot_obs": {"ego_history_xyz": history_xyz.tolist(), "ego_history_rot": history_rot.tolist()},
        }
        sampling = obs.get("sampling_params") or {}
        if "diffusion_steps" in sampling:
            extra_args["diffusion_steps"] = int(sampling["diffusion_steps"])
        if seed is not None:
            extra_args["action_seed"] = int(seed)
        return OpenPIEngineRequest(
            prompt={"prompt_token_ids": prompt_ids, "multi_modal_data": {"image": list(frames.flatten(0, 1))}},
            sampling_params=SamplingParams(
                temperature=_TEMPERATURE,
                top_p=_TOP_P,
                max_tokens=_MAX_TOKENS,
                seed=None if seed is None else int(seed),
                # Trajectory tokens are outside the stage tokenizer.
                detokenize=False,
                stop_token_ids=[int(self.config.traj_ids["future_end"])],
                extra_args=extra_args,
            ),
            request_id=request_id,
        )
