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
from vllm.sampling_params import RequestOutputKind

from vllm_omni.entrypoints.openpi.request_adapters import OpenPIEngineRequest


_ACTION_EXPERT_MAX_BATCH_SIZE_ENV = "NIM_ALPAMAYO_ACTION_EXPERT_MAX_BATCH_SIZE"


def _wire_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _wire_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
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
    def _frames(
        observation: Mapping[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
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
            frame_tensor = frame_tensor.unflatten(0, (camera_indices.numel(), frames_per_camera))
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
        sample_count = int(self.policy_config.get("num_trajectory_samples", 1))
        if sample_count < 1:
            raise ValueError("num_trajectory_samples must be positive")
        sampling_seed = int(self.policy_config.get("seed", 42))
        precision = str(
            self.policy_config.get("precision")
            or os.getenv("NIM_PRECISION")
            or os.getenv("NIM_ALPAMAYO_PRECISION")
            or "bf16"
        ).strip().lower()
        user_batch_size = os.getenv(_ACTION_EXPERT_MAX_BATCH_SIZE_ENV)
        configured_batch_size = self.policy_config.get("action_expert_max_batch_size")
        # Three samples is the largest BF16 action-expert batch qualified on an
        # 80 GB H100. Keep FP8 fully batched by default. A user's environment
        # override takes priority over the profile configuration in both modes.
        if user_batch_size is not None:
            action_expert_max_batch_size = int(user_batch_size)
        elif configured_batch_size is not None:
            action_expert_max_batch_size = int(configured_batch_size)
        elif precision == "bf16":
            action_expert_max_batch_size = min(sample_count, 3)
        else:
            action_expert_max_batch_size = sample_count
        if action_expert_max_batch_size < 1:
            raise ValueError(
                f"{_ACTION_EXPERT_MAX_BATCH_SIZE_ENV} and "
                "action_expert_max_batch_size must be positive"
            )
        extra_args = {
            "reset": reset,
            "session_id": session_id,
            "robot_obs": _wire_value(
                {
                    "ego_history_xyz": observation["ego_history_xyz"],
                    "ego_history_rot": observation["ego_history_rot"],
                }
            ),
            # vLLM's native parallel-sampling path creates one independently
            # decoded VLM completion per trajectory.  Each child invokes the
            # action expert once, conditioned on its own reasoning prefix.
            "num_traj_samples": 1,
            "_parallel_sample_count": sample_count,
            "_batch_action_expert": bool(self.policy_config.get("batch_action_expert", True)),
            "_action_expert_max_batch_size": action_expert_max_batch_size,
            "diffusion_steps": int(self.policy_config.get("diffusion_steps", 10)),
            "_static_expert_cache": bool(self.policy_config.get("static_expert_cache", False)),
            "_compile_expert": bool(self.policy_config.get("compile_actions", False)),
            "_manual_action_cudagraph": bool(self.policy_config.get("manual_action_cudagraph", False)),
            "_static_expert_cache_max_len": int(self.policy_config.get("static_expert_cache_max_len", 4800)),
        }
        return OpenPIEngineRequest(
            prompt={
                "prompt_token_ids": prompt_ids,
                "multi_modal_data": {"image": images},
                "multi_modal_uuids": {"image": [f"{request_id}:image:{index}" for index in range(len(images))]},
                "mm_processor_kwargs": {"device": "cuda"},
            },
            sampling_params=SamplingParams(
                n=sample_count,
                seed=sampling_seed,
                temperature=float(self.policy_config.get("temperature", 0.6)),
                top_p=float(self.policy_config.get("top_p", 0.98)),
                max_tokens=int(self.policy_config.get("max_tokens", 128)),
                output_kind=RequestOutputKind.FINAL_ONLY,
                # ``future_start`` triggers the in-model action expert on the
                # following decode step. Stop only after that hook forces the
                # terminal ``future_end`` token and publishes its payload.
                stop_token_ids=[int(self.model_config.traj_ids["future_end"])],
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
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = self.tokenizer.encode(prompt)
        return OpenPIEngineRequest(
            prompt={
                # Pass token IDs, as the trajectory path does. Passing the
                # multimodal chat-template string back through AsyncOmni's
                # generic prompt parser makes it reinterpret Alpamayo image
                # placeholders and fails with ``TypeError: must be str, not int``.
                "prompt_token_ids": prompt_ids,
                "multi_modal_data": {"image": list(frames.flatten(0, 1))},
            },
            sampling_params=SamplingParams(max_tokens=256),
            request_id=request_id,
        )

    @staticmethod
    def _normalize_future_tensor(value: Any, *, rotations: bool) -> torch.Tensor:
        tensor = torch.as_tensor(value)
        unbatched_ndim = 3 if rotations else 2
        if tensor.ndim == unbatched_ndim:
            return tensor.unsqueeze(0).unsqueeze(0)
        if tensor.ndim == unbatched_ndim + 1:
            return tensor.unsqueeze(1)
        if tensor.ndim == unbatched_ndim + 2:
            return tensor
        kind = "future_rot" if rotations else "future_xyz"
        raise ValueError(f"{kind} has an unsupported trajectory shape {tuple(tensor.shape)}")

    def build_text_task_request(
        self,
        observation: dict[str, Any],
        *,
        task: str,
        request_id: str,
        question: str | None = None,
        future_xyz: Any | None = None,
        future_rot: Any | None = None,
    ) -> OpenPIEngineRequest:
        """Build a vLLM request for the released structured text tasks."""
        from alpamayo2_super.models.utils import fuse_traj_tokens
        from alpamayo2_super.text_tasks import (
            DEFAULT_GROUNDING_QUESTION,
            build_text_task_messages,
        )

        if task not in {"meta_action", "auto_labeling", "grounding"}:
            raise ValueError(f"Unsupported Alpamayo 2 Super text task: {task!r}")
        if (future_xyz is None) != (future_rot is None):
            raise ValueError("future_xyz and future_rot must be provided together")
        if task == "auto_labeling" and future_xyz is None:
            raise ValueError("auto_labeling requires a future trajectory")

        frames, camera_indices, _ = self._frames(observation)
        data: dict[str, Any] = {
            "image_frames": frames,
            "camera_indices": camera_indices,
        }
        prompt_task = task
        if task == "grounding":
            # Grounding uses the released no-special VQA generation path; the
            # caller/backend parses the returned Qwen-style bounding-box JSON.
            prompt_task = "vqa"
            data["question"] = question or DEFAULT_GROUNDING_QUESTION

        messages = build_text_task_messages(data, self.model_config, prompt_task)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = torch.tensor([self.tokenizer.encode(prompt)])
        if task != "grounding":
            trajectory_data = {
                "ego_history_xyz": torch.as_tensor(observation["ego_history_xyz"]),
                "ego_history_rot": torch.as_tensor(observation["ego_history_rot"]),
            }
            if task == "auto_labeling":
                trajectory_data["ego_future_xyz"] = self._normalize_future_tensor(future_xyz, rotations=False)
                trajectory_data["ego_future_rot"] = self._normalize_future_tensor(future_rot, rotations=True)
            prompt_ids = fuse_traj_tokens(
                self.history_tokenizer,
                self.future_tokenizer,
                prompt_ids,
                trajectory_data,
                self.model_config.traj_ids,
            )

        stop_token_ids = None
        if task == "meta_action":
            # The next special token starts policy trajectory generation. Text
            # tasks stop at that boundary instead of invoking the action expert.
            stop_token_ids = [int(self.model_config.traj_ids["future_start"])]
        return OpenPIEngineRequest(
            prompt={
                "prompt_token_ids": prompt_ids[0].tolist(),
                "multi_modal_data": {"image": list(frames.flatten(0, 1))},
            },
            sampling_params=SamplingParams(
                max_tokens=1024 if task == "auto_labeling" else 512,
                stop_token_ids=stop_token_ids,
            ),
            request_id=request_id,
        )
