# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prompt and trajectory-token utilities for Alpamayo 1.5."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

DISCRETE_TRAJECTORY_VOCAB_SIZE = 4000
HISTORY_TRAJECTORY_OFFSET = 3000
TOKENS_PER_HISTORY = 48

SPECIAL_TOKEN_KEYS = (
    "prompt_start",
    "prompt_end",
    "image_start",
    "_padding_0",
    "image_end",
    "traj_history_start",
    "_padding_1",
    "traj_history_end",
    "cot_start",
    "cot_end",
    "_padding_2",
    "_padding_3",
    "traj_future_start",
    "_padding_4",
    "traj_future_end",
    "traj_history",
    "traj_future",
    "image_pad",
    "_padding_5",
    "_padding_6",
    "_padding_7",
    "_padding_8",
    "route_start",
    "route_pad",
    "route_end",
    "question_start",
    "question_end",
    "answer_start",
    "answer_end",
)
SPECIAL_TOKENS = {key: f"<|{key}|>" for key in SPECIAL_TOKEN_KEYS}

CAMERA_DISPLAY_NAMES = {
    0: "Front left camera",
    1: "Front camera",
    2: "Front right camera",
    3: "Rear left camera",
    4: "Rear camera",
    5: "Rear right camera",
    6: "Front telephoto camera",
}


def extend_tokenizer(tokenizer: Any) -> Any:
    """Apply the exact token extension used to train Alpamayo 1.5.

    This operation is idempotent, which is useful when a tokenizer directory
    produced by :func:`save_extended_tokenizer` is loaded again.
    """

    discrete_tokens = [f"<i{index}>" for index in range(DISCRETE_TRAJECTORY_VOCAB_SIZE)]
    tokenizer.add_tokens(discrete_tokens)
    tokenizer.add_tokens(list(SPECIAL_TOKENS.values()), special_tokens=True)
    tokenizer.traj_token_start_idx = tokenizer.convert_tokens_to_ids("<i0>")
    tokenizer.traj_token_ids = {
        key: tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[f"traj_{key}"])
        for key in (
            "history",
            "future",
            "history_start",
            "future_start",
            "history_end",
            "future_end",
        )
    }
    return tokenizer


def encode_history_trajectory(
    history_xyz: torch.Tensor,
    *,
    token_start_idx: int = 151669 + HISTORY_TRAJECTORY_OFFSET,
    num_bins: int = 1000,
) -> torch.LongTensor:
    """Encode 16 XYZ history waypoints into Alpamayo's 48 delta tokens."""

    if history_xyz.shape[-2:] != (16, 3):
        raise ValueError(f"ego_history_xyz must end in shape (16, 3); received {tuple(history_xyz.shape)}")
    xyz = history_xyz.to(dtype=torch.float32)
    origin = torch.zeros_like(xyz[..., :1, :])
    delta = torch.diff(torch.cat((origin, xyz), dim=-2), dim=-2)
    minimum = delta.new_tensor((-4.0, -4.0, -10.0))
    maximum = delta.new_tensor((4.0, 4.0, 10.0))
    bins = ((delta - minimum) / (maximum - minimum) * (num_bins - 1)).round()
    return bins.clamp_(0, num_bins - 1).to(torch.long).flatten(start_dim=-2) + token_start_idx


def fuse_history_tokens(
    input_ids: torch.Tensor,
    history_xyz: torch.Tensor,
    *,
    placeholder_id: int = 155684,
    token_start_idx: int = 154669,
) -> torch.Tensor:
    """Replace all ``<|traj_history|>`` placeholders in a batch."""

    encoded = encode_history_trajectory(history_xyz, token_start_idx=token_start_idx)
    if encoded.ndim == 1:
        encoded = encoded.unsqueeze(0)
    result = input_ids.clone()
    if result.ndim == 1:
        mask = result == placeholder_id
        if int(mask.sum()) != TOKENS_PER_HISTORY:
            raise ValueError(f"expected {TOKENS_PER_HISTORY} history placeholders")
        result.masked_scatter_(mask, encoded.reshape(-1).to(result.device))
        return result
    if result.shape[0] != encoded.shape[0]:
        raise ValueError("input and history batch sizes differ")
    for row, tokens in zip(result, encoded, strict=True):
        mask = row == placeholder_id
        if int(mask.sum()) != TOKENS_PER_HISTORY:
            raise ValueError(f"expected {TOKENS_PER_HISTORY} history placeholders")
        row.masked_scatter_(mask, tokens.to(row.device))
    return result


def _image_content(
    images: Sequence[Any], camera_indices: Sequence[int] | None, frames_per_camera: int
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for index, image in enumerate(images):
        camera_index = index // frames_per_camera
        frame_index = index % frames_per_camera
        if frame_index == 0 and camera_indices is not None:
            camera_id = int(camera_indices[camera_index])
            content.append({"type": "text", "text": f"{CAMERA_DISPLAY_NAMES.get(camera_id, f'Camera {camera_id}')}: "})
        if camera_indices is not None:
            content.append({"type": "text", "text": f"frame {frame_index} "})
        content.append({"type": "image", "image": image})
    return content


def create_policy_messages(
    images: Sequence[Any],
    *,
    camera_indices: Sequence[int] | None = None,
    frames_per_camera: int = 4,
    navigation: str | None = None,
) -> list[dict[str, Any]]:
    """Build the training-compatible policy conversation."""

    placeholders = SPECIAL_TOKENS["traj_history"] * TOKENS_PER_HISTORY
    history = SPECIAL_TOKENS["traj_history_start"] + placeholders + SPECIAL_TOKENS["traj_history_end"]
    route = ""
    if navigation:
        route = SPECIAL_TOKENS["route_start"] + navigation + SPECIAL_TOKENS["route_end"]
    instruction = "output the chain-of-thought reasoning of the driving process, then output the future trajectory."
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a driving assistant that generates safe and accurate actions.",
                }
            ],
        },
        {
            "role": "user",
            "content": _image_content(images, camera_indices, frames_per_camera)
            + [{"type": "text", "text": history + route + instruction}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": SPECIAL_TOKENS["cot_start"]}],
        },
    ]


def create_vqa_messages(
    images: Sequence[Any],
    question: str,
    *,
    camera_indices: Sequence[int] | None = None,
    frames_per_camera: int = 4,
) -> list[dict[str, Any]]:
    """Build an Alpamayo visual-question-answering conversation."""

    question_text = SPECIAL_TOKENS["question_start"] + question + SPECIAL_TOKENS["question_end"]
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a driving assistant that generates safe and accurate actions.",
                }
            ],
        },
        {
            "role": "user",
            "content": _image_content(images, camera_indices, frames_per_camera)
            + [{"type": "text", "text": question_text}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": SPECIAL_TOKENS["answer_start"]}],
        },
    ]


def get_observation_history(observation: Mapping[str, Any]) -> torch.Tensor:
    """Read and normalize the OpenPI history field."""

    value = observation.get("ego_history_xyz")
    if value is None:
        raise KeyError("Alpamayo policy observations require 'ego_history_xyz'")
    history = observation_tensor(value)
    while history.ndim > 2 and history.shape[0] == 1:
        history = history.squeeze(0)
    if history.shape != (16, 3):
        raise ValueError(f"ego_history_xyz must have shape (16, 3), got {tuple(history.shape)}")
    return history


def observation_tensor(value: Any) -> torch.Tensor:
    """Convert an observation, including vLLM's msgpack ndarray form."""

    if (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and isinstance(value[0], str)
        and isinstance(value[1], (list, tuple))
    ):
        array = np.frombuffer(value[2], dtype=np.dtype(value[0])).reshape(value[1])
        return torch.from_numpy(array.copy())
    return torch.as_tensor(value)
