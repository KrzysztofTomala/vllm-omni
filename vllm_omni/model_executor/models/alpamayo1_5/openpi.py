# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""OpenPI request adaptation owned by the Alpamayo model integration."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer
from vllm import SamplingParams

from vllm_omni.entrypoints.openpi.request_adapters import OpenPIEngineRequest

from .processing import (
    create_policy_messages,
    create_vqa_messages,
    fuse_history_tokens,
    get_observation_history,
)
from .tokenizer import ensure_extended_tokenizer


def _to_sampling_value(value: Any) -> Any:
    """Convert observation values to builtins accepted by extra_args RPC."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _to_sampling_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_sampling_value(item) for item in value]
    return value


class AlpamayoOpenPIRequestAdapter:
    """Build the training-compatible Alpamayo AR policy request."""

    def __init__(self, policy_config: dict[str, Any]) -> None:
        self.policy_config = policy_config
        tokenizer_path = ensure_extended_tokenizer(policy_config.get("tokenizer_backbone", "nvidia/Cosmos-Reason2-8B"))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    @staticmethod
    def _vision_cache_enabled() -> bool:
        return os.getenv("NIM_ALPAMAYO_VISION_EMBED_CACHE", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _image_uuids(
        self,
        observation: Mapping[str, Any],
        *,
        request_id: str,
        image_count: int,
    ) -> list[str]:
        if not self._vision_cache_enabled():
            return [f"{request_id}:image:{index}" for index in range(image_count)]
        image_uuids = observation.get("image_uuids")
        if not isinstance(image_uuids, list) or len(image_uuids) != image_count:
            uuid_count = len(image_uuids) if isinstance(image_uuids, list) else 0
            raise ValueError(
                "Vision embedding caching requires one content UUID per image; "
                f"got {uuid_count} IDs for {image_count} images."
            )
        return [str(image_uuid) for image_uuid in image_uuids]

    def build_request(
        self,
        observation: dict[str, Any],
        *,
        request_id: str,
        session_id: str,
        reset: bool,
    ) -> OpenPIEngineRequest:
        image_frames = observation.get("image_frames", observation.get("images"))
        if image_frames is None:
            raise KeyError("Alpamayo observations require 'image_frames' or 'images'")
        frames = torch.as_tensor(image_frames) if isinstance(image_frames, np.ndarray) else image_frames
        if isinstance(frames, torch.Tensor):
            if frames.ndim == 5:
                frames = frames.flatten(0, 1)
            if frames.ndim != 4:
                raise ValueError("Alpamayo image frames must be rank 4 or 5")
            images = list(frames)
        else:
            images = list(frames)

        camera_indices = observation.get("camera_indices")
        if isinstance(camera_indices, np.ndarray):
            camera_indices = camera_indices.tolist()
        messages = create_policy_messages(
            images,
            camera_indices=camera_indices,
            frames_per_camera=int(observation.get("num_frames_per_camera", 4)),
            navigation=observation.get("nav_text", observation.get("navigation")),
        )
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        prompt_ids = torch.tensor(self.tokenizer.encode(prompt_text))
        prompt_ids = fuse_history_tokens(
            prompt_ids,
            get_observation_history(observation),
        ).tolist()

        policy_observation = {"ego_history_xyz": observation["ego_history_xyz"]}
        if "ego_history_rot" in observation:
            policy_observation["ego_history_rot"] = observation["ego_history_rot"]
        extra_args = {
            "reset": reset,
            "session_id": session_id,
            "robot_obs": _to_sampling_value(policy_observation),
            "num_traj_samples": int(self.policy_config.get("num_trajectory_samples", 6)),
            "diffusion_steps": int(self.policy_config.get("diffusion_steps", 10)),
            "action_temperature": float(self.policy_config.get("action_temperature", 1.0)),
            "_sampling_seed": int(self.policy_config.get("seed", 42)),
            "_nim_action_rng_compat": bool(self.policy_config.get("nim_action_rng_compat", True)),
            "_compile_expert": bool(self.policy_config.get("compile_actions", False)),
            "_manual_action_cudagraph": bool(self.policy_config.get("manual_action_cudagraph", False)),
            "_static_expert_cache_max_len": int(self.policy_config.get("static_expert_cache_max_len", 3328)),
        }
        sampling_params = SamplingParams(
            temperature=float(self.policy_config.get("temperature", 0.6)),
            top_p=float(self.policy_config.get("top_p", 0.98)),
            max_tokens=int(self.policy_config.get("max_tokens", 128)),
            stop_token_ids=[155683],
            extra_args=extra_args,
        )
        return OpenPIEngineRequest(
            prompt={
                "prompt_token_ids": prompt_ids,
                "multi_modal_data": {"image": images},
                # Live frames must miss the multimodal cache. Explicit IDs
                # avoid hashing the decoded RGB tensors to prove uniqueness.
                "multi_modal_uuids": {
                    "image": self._image_uuids(
                        observation,
                        request_id=request_id,
                        image_count=len(images),
                    )
                },
                "mm_processor_kwargs": {"device": "cuda"},
            },
            sampling_params=sampling_params,
            request_id=request_id,
        )

    def build_vqa_request(
        self,
        observation: dict[str, Any],
        *,
        question: str,
        request_id: str,
    ) -> OpenPIEngineRequest:
        """Build the native Alpamayo 1.5 visual-question request."""

        image_frames = observation.get("image_frames", observation.get("images"))
        if image_frames is None:
            raise KeyError("Alpamayo observations require 'image_frames' or 'images'")
        frames = torch.as_tensor(image_frames) if isinstance(image_frames, np.ndarray) else image_frames
        if isinstance(frames, torch.Tensor):
            if frames.ndim == 5:
                frames = frames.flatten(0, 1)
            if frames.ndim != 4:
                raise ValueError("Alpamayo image frames must be rank 4 or 5")
            images = list(frames)
        else:
            images = list(frames)
        camera_indices = observation.get("camera_indices")
        if isinstance(camera_indices, np.ndarray):
            camera_indices = camera_indices.tolist()
        messages = create_vqa_messages(
            images,
            question,
            camera_indices=camera_indices,
        )
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        return OpenPIEngineRequest(
            prompt={
                "prompt": prompt,
                "multi_modal_data": {"image": images},
                "multi_modal_uuids": {
                    "image": self._image_uuids(
                        observation,
                        request_id=request_id,
                        image_count=len(images),
                    )
                },
                "mm_processor_kwargs": {"device": "cuda"},
            },
            sampling_params=SamplingParams(max_tokens=256),
            request_id=request_id,
        )
