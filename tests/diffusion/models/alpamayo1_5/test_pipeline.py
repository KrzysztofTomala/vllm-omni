# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.alpamayo1_5.modeling_alpamayo1_5 import (
    _MaskTrajectoryTokens,
    _StopAfterToken,
)
from vllm_omni.diffusion.models.alpamayo1_5.pipeline_alpamayo1_5 import (
    Alpamayo1_5Pipeline,
    _frames_from_observation,
)
from vllm_omni.model_executor.models.alpamayo1_5.action import (
    UnicycleTrajectoryDecoder,
)
from vllm_omni.model_executor.models.alpamayo1_5.pipeline import (
    ALPAMAYO1_5_PIPELINE,
)
from vllm_omni.model_executor.models.alpamayo1_5.processing import (
    SPECIAL_TOKENS,
    create_policy_messages,
    encode_history_trajectory,
    extend_tokenizer,
    fuse_history_tokens,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _FakeTokenizer:
    def __init__(self) -> None:
        self.tokens = {f"base-{index}": index for index in range(151669)}
        del self.tokens["base-151655"]
        self.tokens["<|image_pad|>"] = 151655

    def add_tokens(self, values, special_tokens=False):
        del special_tokens
        added = 0
        for value in values:
            if value not in self.tokens:
                self.tokens[value] = len(self.tokens)
                added += 1
        return added

    def convert_tokens_to_ids(self, value):
        return self.tokens[value]

    def __len__(self):
        return len(self.tokens)


def test_tokenizer_extension_matches_checkpoint_ids():
    tokenizer = extend_tokenizer(_FakeTokenizer())

    assert len(tokenizer) == 155697
    assert tokenizer.traj_token_start_idx == 151669
    assert tokenizer.traj_token_ids == {
        "history_start": 155674,
        "history_end": 155676,
        "future_start": 155681,
        "future_end": 155683,
        "history": 155684,
        "future": 155685,
    }


def test_history_delta_encoding_and_fusion():
    history = torch.zeros(16, 3)
    history[:, 0] = torch.arange(16) * 0.1
    tokens = encode_history_trajectory(history)
    fused = fuse_history_tokens(torch.tensor([1] + [155684] * 48 + [2]), history)

    assert tokens.shape == (48,)
    assert tokens.min() >= 154669
    assert tokens.max() <= 155668
    assert torch.equal(fused[1:-1], tokens)
    assert fused[[0, -1]].tolist() == [1, 2]


def test_policy_messages_use_training_control_tokens():
    messages = create_policy_messages(
        [torch.zeros(3, 8, 8) for _ in range(4)],
        camera_indices=[1],
        navigation="turn left",
    )
    policy_text = messages[1]["content"][-1]["text"]

    assert policy_text.count(SPECIAL_TOKENS["traj_history"]) == 48
    assert SPECIAL_TOKENS["route_start"] + "turn left" in policy_text
    assert messages[-1]["content"][0]["text"] == SPECIAL_TOKENS["cot_start"]


def test_frames_accept_openpi_ndarray_encoding():
    frames = np.zeros((1, 4, 3, 8, 8), dtype=np.uint8)
    encoded = [frames.dtype.str, list(frames.shape), frames.tobytes()]

    result = _frames_from_observation({"image_frames": encoded})

    assert result.shape == (4, 3, 8, 8)
    assert result.dtype == torch.uint8


def test_generation_guards_mask_actions_and_stop_one_step_late():
    scores = torch.zeros(1, 10)
    masked = _MaskTrajectoryTokens(3, 4)(torch.zeros(1, 1, dtype=torch.long), scores)
    stopper = _StopAfterToken(7)

    assert torch.isneginf(masked[0, 3:7]).all()
    assert not stopper(torch.tensor([[1, 7]]), scores)
    assert stopper(torch.tensor([[1, 7, 2]]), scores)


def test_pipeline_topology_and_dummy_output_are_action_native():
    stage = ALPAMAYO1_5_PIPELINE.stages[0]
    pipeline = object.__new__(Alpamayo1_5Pipeline)
    output = pipeline._dummy_output().output

    assert stage.model_stage == "diffusion"
    assert stage.final_output_type == "actions"
    assert output["actions"].shape == (1, 64, 3)


def test_trajectory_normalization_is_not_loader_managed_state():
    decoder = UnicycleTrajectoryDecoder(
        {
            "accel_mean": 0.03,
            "accel_std": 0.68,
            "curvature_mean": 0.0003,
            "curvature_std": 0.026,
        }
    )

    assert decoder.state_dict() == {}
    assert decoder.accel_std == 0.68
    assert decoder.curvature_std == 0.026


def test_pipeline_declares_model_owned_weight_loading():
    pipeline = object.__new__(Alpamayo1_5Pipeline)

    assert pipeline.load_weights(iter(())) is None
    with pytest.raises(RuntimeError, match="loaded directly"):
        pipeline.load_weights(iter((("unexpected", torch.zeros(1)),)))
