# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import numpy as np
import pytest

from vllm_omni.model_executor.models.alpamayo1_5.openpi import (
    AlpamayoOpenPIRequestAdapter,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeTokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return "policy prompt"

    def encode(self, prompt):
        assert prompt == "policy prompt"
        return [1] + [155684] * 48 + [2]


def test_alpamayo_openpi_adapter_builds_ar_request_with_unique_image_ids():
    adapter = object.__new__(AlpamayoOpenPIRequestAdapter)
    adapter.policy_config = {
        "num_trajectory_samples": 1,
        "diffusion_steps": 10,
    }
    adapter.tokenizer = _FakeTokenizer()
    history = np.zeros((16, 3), dtype=np.float32)
    images = np.zeros((1, 4, 3, 8, 8), dtype=np.uint8)

    request = adapter.build_request(
        {
            "image_frames": images,
            "camera_indices": np.array([1]),
            "ego_history_xyz": history,
            "navigation": "Drive forward.",
        },
        request_id="robot-session-7",
        session_id="session",
        reset=True,
    )

    assert request.request_id == "robot-session-7"
    assert len(request.prompt["multi_modal_data"]["image"]) == 4
    assert request.prompt["multi_modal_uuids"]["image"] == [
        f"robot-session-7:image:{index}" for index in range(4)
    ]
    assert len(request.prompt["prompt_token_ids"]) == 50
    assert request.sampling_params.extra_args["robot_obs"]["ego_history_xyz"] == history.tolist()
    assert request.sampling_params.extra_args["reset"] is True
    assert request.sampling_params.extra_args["session_id"] == "session"
