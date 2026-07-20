# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-contained single-GPU Alpamayo 1.5 policy pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import AutoProcessor, AutoTokenizer

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.model_executor.models.alpamayo1_5.configuration_alpamayo1_5 import (
    Alpamayo1_5Config,
)
from vllm_omni.model_executor.models.alpamayo1_5.processing import (
    create_policy_messages,
    extend_tokenizer,
    observation_tensor,
)

from .modeling_alpamayo1_5 import Alpamayo1_5TorchModel


def _frames_from_observation(observation: Mapping[str, Any]) -> torch.Tensor:
    value = observation.get("image_frames", observation.get("images"))
    if value is None:
        raise KeyError("Alpamayo observations require 'image_frames' or 'images'")
    frames = observation_tensor(value)
    if frames.ndim == 5:
        frames = frames.flatten(0, 1)
    if frames.ndim != 4:
        raise ValueError(f"Alpamayo image frames must be rank 4 or 5, got {frames.ndim}")
    return frames


class Alpamayo1_5Pipeline(nn.Module):
    """Run reasoning and action prediction entirely inside model-owned code."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        del prefix
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = od_config.dtype
        self.model_config = od_config.model_config
        config = Alpamayo1_5Config.from_pretrained(od_config.model)
        config._attn_implementation = str(od_config.model_config.get("attn_implementation", "sdpa"))
        self.model = Alpamayo1_5TorchModel.from_pretrained(
            od_config.model,
            config=config,
            dtype=self.dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        tokenizer = extend_tokenizer(AutoTokenizer.from_pretrained(config.vlm_name_or_path))
        self.processor = AutoProcessor.from_pretrained(
            config.vlm_name_or_path,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
        )
        self.processor.tokenizer = tokenizer
        self.tokenizer = tokenizer

    @property
    def weights_sources(self) -> tuple[Any, ...]:
        return ()

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str] | None:
        consumed = list(weights)
        if consumed:
            raise RuntimeError("Alpamayo weights are loaded directly by the model pipeline")
        # The Transformers components are loaded by ``from_pretrained`` in
        # ``__init__``. Returning None tells the generic diffusion loader that
        # this model owns its loading lifecycle, so it must not validate those
        # parameters against an empty external weight stream.
        return None

    def _dummy_output(self) -> DiffusionOutput:
        return DiffusionOutput(
            output={
                "actions": np.zeros((1, 64, 3), dtype=np.float32),
                "rotations": np.zeros((1, 64, 3, 3), dtype=np.float32),
                "normalized_controls": np.zeros((1, 64, 2), dtype=np.float32),
                "reasoning": "",
            }
        )

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch, **kwargs: Any) -> DiffusionOutput:
        del kwargs
        if req.request_id == DUMMY_DIFFUSION_REQUEST_ID:
            return self._dummy_output()
        extra_args = req.sampling_params.extra_args or {}
        observation = extra_args.get("robot_obs")
        if not isinstance(observation, Mapping):
            return DiffusionOutput(error="Alpamayo requires sampling_params.extra_args['robot_obs']")
        try:
            frames = _frames_from_observation(observation)
            camera_indices = observation.get("camera_indices")
            if camera_indices is not None:
                camera_indices = observation_tensor(camera_indices).tolist()
            messages = create_policy_messages(
                list(frames),
                camera_indices=camera_indices,
                frames_per_camera=int(observation.get("num_frames_per_camera", 4)),
                navigation=observation.get("nav_text", observation.get("navigation")),
            )
            tokenized = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                continue_final_message=True,
                return_dict=True,
                return_tensors="pt",
                device=str(self.device),
            )
            tokenized = {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in tokenized.items()
            }
            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                result = self.model.sample_trajectories(
                    tokenized,
                    observation,
                    tokenizer=self.tokenizer,
                    temperature=float(extra_args.get("temperature", self.model_config.get("temperature", 0.6))),
                    top_p=float(extra_args.get("top_p", self.model_config.get("top_p", 0.98))),
                    top_k=extra_args.get("top_k", self.model_config.get("top_k")),
                    max_tokens=int(extra_args.get("max_tokens", self.model_config.get("max_tokens", 128))),
                    sample_count=int(
                        extra_args.get(
                            "num_traj_samples",
                            self.model_config.get("num_trajectory_samples", 6),
                        )
                    ),
                    inference_steps=int(
                        extra_args.get("diffusion_steps", self.model_config.get("diffusion_steps", 10))
                    ),
                    action_temperature=float(extra_args.get("action_temperature", 1.0)),
                    seed=extra_args.get("seed"),
                )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            return DiffusionOutput(error=str(exc))
        return DiffusionOutput(
            output={
                key: value.detach().float().cpu().numpy() if isinstance(value, torch.Tensor) else value
                for key, value in result.items()
            }
        )
