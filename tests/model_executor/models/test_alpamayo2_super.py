# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the Alpamayo 2 Super integration."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration

from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.model_executor.models.alpamayo2_super import Alpamayo2SuperConfig
from vllm_omni.model_executor.models.alpamayo2_super import openpi as alpamayo_openpi
from vllm_omni.model_executor.models.alpamayo2_super.action_expert import (
    UnicycleAccelCurvatureActionSpace,
    _velocity_from_history,
)
from vllm_omni.model_executor.models.alpamayo2_super.alpamayo2_super import (
    ACTIONS_KEY,
    Alpamayo2SuperForConditionalGeneration,
)
from vllm_omni.model_executor.models.alpamayo2_super.processing import (
    SPECIAL_TOKENS,
    build_trajectory_messages,
    encode_history_tokens,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.registry import _OMNI_MODELS

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

FUTURE_START = 20
FUTURE_END = 21
TEXT_EOS = 2


def test_config_exposes_nested_qwen3_vl_fields():
    config = Alpamayo2SuperConfig(
        vlm_config={"text_config": {"hidden_size": 64}, "vision_config": {"depth": 2}, "image_token_id": 7},
        traj_ids={"future_start": FUTURE_START},
        history_vocab_size=10,
        future_vocab_size=30,
    )
    assert config.text_config.hidden_size == 64
    assert config.vision_config.depth == 2
    assert config.image_token_id == 7
    assert config.traj_vocab_size == 40
    assert config.traj_ids["future_start"] == FUTURE_START


def test_registered_as_single_stage_policy():
    pipeline = OMNI_PIPELINES["alpamayo2_super"]
    assert pipeline.model_arch == "Alpamayo2Super"
    assert len(pipeline.stages) == 1 and pipeline.stages[0].final_output
    assert _OMNI_MODELS["Alpamayo2Super"][2] == "Alpamayo2SuperForConditionalGeneration"


def test_history_tokens_quantize_waypoint_deltas():
    # Constant 1 m steps along x; y and z stay fixed.
    history = torch.zeros(16, 3)
    history[:, 0] = torch.arange(-15.0, 1.0)
    tokens = encode_history_tokens(history, num_bins=1000)
    assert tokens.shape == (45,)
    x_bin = round((1.0 + 4.0) / 8.0 * 999)
    y_bin = round(4.0 / 8.0 * 999)
    z_bin = round(10.0 / 20.0 * 999)
    assert tokens.view(15, 3).tolist() == [[x_bin, y_bin, z_bin]] * 15


def test_trajectory_messages_follow_released_prompt():
    config = SimpleNamespace(include_camera_ids=True, frame_label="frame_num", tokens_per_history_traj=3)
    messages = build_trajectory_messages([["a0", "a1"], ["b0", "b1"]], [1, 6], config)
    assert [message["role"] for message in messages] == ["system", "user"]
    content = messages[1]["content"]
    assert content[0] == {"type": "text", "text": "Front camera: "}
    assert content[1] == {"type": "text", "text": "frame 0 "}
    assert content[2] == {"type": "image", "image": "a0"}
    assert content[5] == {"type": "text", "text": "Front telephoto camera: "}
    assert content[7] == {"type": "image", "image": "b0"}
    history = SPECIAL_TOKENS["traj_history"] * 3
    assert content[-2]["text"] == f"<|traj_history_start|>{history}<|traj_history_end|>"
    assert content[-1]["text"].endswith("then output the future trajectory.")
    with pytest.raises(ValueError, match="sorted"):
        build_trajectory_messages([["a"], ["b"]], [6, 1], config)


def test_zero_controls_keep_constant_speed_straight():
    action_space = UnicycleAccelCurvatureActionSpace(dt=0.1, n_waypoints=64)
    speed = 10.0
    history_xyz = torch.zeros(1, 16, 3)
    history_xyz[0, :, 0] = speed * 0.1 * torch.arange(-15.0, 1.0)
    history_rot = torch.eye(3).expand(1, 16, 3, 3)
    xyz, rot = action_space.action_to_traj(torch.zeros(1, 64, 2), history_xyz, history_rot)
    assert xyz.shape == (1, 64, 3) and rot.shape == (1, 64, 3, 3)
    expected_x = speed * 0.1 * torch.arange(1.0, 65.0)
    torch.testing.assert_close(xyz[0, :, 0], expected_x, rtol=1e-3, atol=1e-2)
    torch.testing.assert_close(xyz[0, :, 1], torch.zeros(64), rtol=0, atol=1e-4)


def test_velocity_solver_matches_direct_least_squares():
    # Reference formulation: minimize sum_t |v_t u_t + v_{t+1} u_{t+1} - 2 dp_t / dt|^2
    # + lam / dt^6 * |D3 v|^2 + ridge * |v|^2, solved densely.
    generator = torch.Generator().manual_seed(0)
    n, dt, lam, ridge = 15, 0.1, 1e-6, 1e-4
    theta = torch.cumsum(0.05 * torch.randn(n + 1, generator=generator, dtype=torch.float64), dim=0)
    dxy = torch.randn(n, 2, generator=generator, dtype=torch.float64)
    a = torch.zeros(2 * n, n + 1, dtype=torch.float64)
    for t in range(n):
        for row, trig in ((2 * t, torch.cos), (2 * t + 1, torch.sin)):
            a[row, t] = trig(theta[t])
            a[row, t + 1] = trig(theta[t + 1])
    d3 = torch.zeros(n - 2, n + 1, dtype=torch.float64)
    for row in range(n - 2):
        d3[row, row : row + 4] = torch.tensor([-1.0, 3.0, -3.0, 1.0], dtype=torch.float64)
    lhs = a.T @ a + lam / dt**6 * d3.T @ d3 + ridge * torch.eye(n + 1, dtype=torch.float64)
    expected = torch.linalg.solve(lhs, a.T @ (2 / dt * dxy).flatten())
    actual = _velocity_from_history(dxy, theta, dt=dt, v_lambda=lam, v_ridge=ridge)
    torch.testing.assert_close(actual, expected)


def _make_model() -> Alpamayo2SuperForConditionalGeneration:
    model = Alpamayo2SuperForConditionalGeneration.__new__(Alpamayo2SuperForConditionalGeneration)
    nn.Module.__init__(model)
    model.future_start_id = FUTURE_START
    model.future_end_id = FUTURE_END
    model.traj_vocab_start = 10
    model.traj_vocab_size = 5
    model.text_eos_ids = [TEXT_EOS]
    model._step = None
    model._policy_rows = []
    model._finished_rows = []
    model._sample_trajectory = lambda **kwargs: (
        torch.full((64, 3), float(kwargs["request_index"])),
        torch.eye(3).expand(64, 3, 3),
    )
    return model


def _run_step(model, *, input_ids, num_scheduled, num_computed, extra_args):
    hidden = torch.zeros(len(input_ids), 4)
    model.prepare_runner_inputs(
        input_ids=torch.tensor(input_ids),
        positions=torch.arange(len(input_ids)).repeat(3, 1),
        inputs_embeds=None,
        req_ids=[str(i) for i in range(len(num_scheduled))],
        num_computed_tokens=num_computed,
        num_scheduled_tokens=num_scheduled,
        input_ids_buffer=torch.tensor(input_ids),
        sampling_extra_args=extra_args,
    )
    return model.make_omni_output(hidden)


def test_trajectory_runs_when_future_start_is_fed_back():
    model = _make_model()
    observation: dict[str, dict[str, list]] = {"robot_obs": {"ego_history_xyz": [], "ego_history_rot": []}}
    # Request 0 is still prefilling, request 1 feeds back future_start, and
    # request 2 has no observation (plain text generation).
    output = _run_step(
        model,
        input_ids=[5, 6, FUTURE_START, FUTURE_START],
        num_scheduled=[2, 1, 1],
        num_computed=[0, 100, 50],
        extra_args=[observation, observation, {}],
    )
    assert isinstance(output, OmniOutput)
    trajectories = output.multimodal_outputs[ACTIONS_KEY]
    assert trajectories[0] is None and trajectories[2] is None
    assert torch.equal(trajectories[1], torch.full((64, 3), 1.0))
    assert model._policy_rows == [0, 1]
    assert model._finished_rows == [1]


def test_compute_logits_masks_trajectory_vocab_and_forces_end(monkeypatch):
    model = _make_model()
    monkeypatch.setattr(Qwen3VLForConditionalGeneration, "compute_logits", lambda self, hidden: torch.zeros(3, 32))
    model._policy_rows = [0, 1]
    model._finished_rows = [1]
    logits = model.compute_logits(torch.zeros(3, 4))
    assert torch.isinf(logits[:, 10:15]).all()
    assert torch.isinf(logits[0, TEXT_EOS]) and not torch.isinf(logits[2, TEXT_EOS])
    assert logits[1].argmax().item() == FUTURE_END
    assert torch.isinf(logits[1, :FUTURE_END]).all()
    # State is consumed: the next step only applies the vocabulary mask.
    logits = model.compute_logits(torch.zeros(3, 4))
    assert not torch.isinf(logits[0, TEXT_EOS])


def test_openpi_adapter_builds_trajectory_request(monkeypatch):
    calls: dict[str, object] = {}

    def fake_prompt(tokenizer, config, *, frames, camera_indices, ego_history_xyz):
        calls.update(frames=frames.shape, cameras=camera_indices, history=ego_history_xyz.shape)
        return [1, 2, 3]

    monkeypatch.setattr(alpamayo_openpi, "build_trajectory_prompt", fake_prompt)
    adapter = alpamayo_openpi.Alpamayo2SuperOpenPIAdapter.__new__(alpamayo_openpi.Alpamayo2SuperOpenPIAdapter)
    adapter.config = SimpleNamespace(traj_ids={"future_end": FUTURE_END})
    adapter.tokenizer = None
    obs = {
        "image_frames": np.zeros((2, 4, 3, 8, 8), dtype=np.uint8),
        "camera_indices": np.array([1, 6]),
        "ego_history_xyz": np.zeros((16, 3)),
        "ego_history_rot": np.tile(np.eye(3), (16, 1, 1)),
        "seed": 5,
        "sampling_params": {"diffusion_steps": 4},
    }
    request = adapter.build_request(obs, request_id="robot-a-0", session_id="a", reset=True)

    assert calls == {"frames": (2, 4, 3, 8, 8), "cameras": [1, 6], "history": (16, 3)}
    assert request.request_id == "robot-a-0"
    assert request.prompt["prompt_token_ids"] == [1, 2, 3]
    assert len(request.prompt["multi_modal_data"]["image"]) == 8
    params = request.sampling_params
    assert params.seed == 5 and params.stop_token_ids == [FUTURE_END]
    assert params.extra_args["action_seed"] == 5 and params.extra_args["diffusion_steps"] == 4
    assert np.asarray(params.extra_args["robot_obs"]["ego_history_rot"]).shape == (16, 3, 3)
