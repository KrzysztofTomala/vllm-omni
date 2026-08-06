# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request adaptation for the released Alpamayo 2 Super checkpoint."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig
from vllm import SamplingParams

from vllm_omni.entrypoints.openpi.request_adapters import OpenPIEngineRequest


def _wire_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _wire_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_wire_value(item) for item in value]
    return value


class Alpamayo2SuperOpenPIRequestAdapter:
    """Build Super prompts while keeping policy details out of the NIM."""

    def __init__(self, policy_config: dict[str, Any]) -> None:
        # Imports are lazy because the released Alpamayo package is supplied by
        # the model/NIM image rather than becoming a core vLLM dependency.
        import hydra.utils as hyu
        from alpamayo2_super.config import build_alpamayo2_super_tokenizer

        self.policy_config = policy_config
        self.model_path = str(
            policy_config.get("model_path")
            or os.getenv("NIM_ALPAMAYO_MODEL")
            or os.getenv("ALPAMAYO_MODEL_DIR")
            or "nvidia/Alpamayo2-Super"
        )
        self.model_config = AutoConfig.from_pretrained(self.model_path)
        self.tokenizer = build_alpamayo2_super_tokenizer(
            self.model_path,
            int(self.model_config.history_vocab_size),
            int(self.model_config.future_vocab_size),
        )
        self.history_tokenizer = hyu.instantiate(
            self.model_config.hist_traj_tokenizer_cfg,
            load_weights=False,
        )
        self.future_tokenizer = hyu.instantiate(
            self.model_config.future_traj_tokenizer_cfg,
            load_weights=False,
        )

    @staticmethod
    def _frames(observation: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, int]:
        frames = observation.get("image_frames", observation.get("images"))
        if frames is None:
            raise KeyError("Super observations require image_frames or images")
        if isinstance(frames, torch.Tensor):
            frame_tensor = frames
        else:
            frame_tensor = torch.stack([torch.as_tensor(frame) for frame in frames])
        camera_indices = torch.as_tensor(observation["camera_indices"], dtype=torch.int64)
        frames_per_camera = int(observation.get("num_frames_per_camera", 4))
        if frame_tensor.ndim == 4:
            frame_tensor = frame_tensor.unflatten(
                0, (camera_indices.numel(), frames_per_camera)
            )
        return frame_tensor, camera_indices, frames_per_camera

    def _policy_prompt(self, observation: Mapping[str, Any]) -> tuple[list[int], list[torch.Tensor]]:
        from alpamayo2_super.helper import create_messages
        from alpamayo2_super.models.utils import fuse_traj_tokens

        frames, camera_indices, _ = self._frames(observation)
        data = {
            "image_frames": frames,
            "camera_indices": camera_indices,
        }
        navigation = observation.get("nav_text", observation.get("navigation"))
        if navigation:
            data["nav_text"] = [navigation]
        messages = create_messages(data, self.model_config)
        has_assistant = messages[-1]["role"] == "assistant" and bool(messages[-1]["content"])
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not has_assistant,
            continue_final_message=has_assistant,
        )
        prompt_ids = torch.tensor([self.tokenizer.encode(prompt)])
        prompt_ids = fuse_traj_tokens(
            self.history_tokenizer,
            self.future_tokenizer,
            prompt_ids,
            {
                "ego_history_xyz": torch.as_tensor(observation["ego_history_xyz"]),
                "ego_history_rot": torch.as_tensor(observation["ego_history_rot"]),
            },
            self.model_config.traj_ids,
        )[0].tolist()
        return prompt_ids, list(frames.flatten(0, 1))

    def build_request(
        self,
        observation: dict[str, Any],
        *,
        request_id: str,
        session_id: str,
        reset: bool,
    ) -> OpenPIEngineRequest:
        prompt_ids, images = self._policy_prompt(observation)
        extra_args = {
            "reset": reset,
            "session_id": session_id,
            "robot_obs": _wire_value(
                {
                    "ego_history_xyz": observation["ego_history_xyz"],
                    "ego_history_rot": observation["ego_history_rot"],
                }
            ),
            "num_traj_samples": int(self.policy_config.get("num_trajectory_samples", 1)),
            "diffusion_steps": int(self.policy_config.get("diffusion_steps", 10)),
            "_sampling_seed": int(self.policy_config.get("seed", 42)),
        }
        return OpenPIEngineRequest(
            prompt={
                "prompt_token_ids": prompt_ids,
                "multi_modal_data": {"image": images},
                "multi_modal_uuids": {
                    "image": [f"{request_id}:image:{index}" for index in range(len(images))]
                },
                "mm_processor_kwargs": {"device": "cuda"},
            },
            sampling_params=SamplingParams(
                temperature=float(self.policy_config.get("temperature", 0.6)),
                top_p=float(self.policy_config.get("top_p", 0.98)),
                max_tokens=int(self.policy_config.get("max_tokens", 128)),
                stop_token_ids=[int(self.model_config.traj_ids["future_start"])],
                extra_args=extra_args,
            ),
            request_id=request_id,
        )

    def build_vqa_request(
        self,
        observation: dict[str, Any],
        *,
        question: str,
        request_id: str,
    ) -> OpenPIEngineRequest:
        from alpamayo2_super.text_tasks import build_text_task_messages

        frames, camera_indices, _ = self._frames(observation)
        data = {
            "image_frames": frames,
            "camera_indices": camera_indices,
            "question": question,
        }
        messages = build_text_task_messages(data, self.model_config, "vqa")
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return OpenPIEngineRequest(
            prompt={
                "prompt": prompt,
                "multi_modal_data": {"image": list(frames.flatten(0, 1))},
            },
            sampling_params=SamplingParams(max_tokens=256),
            request_id=request_id,
        )
