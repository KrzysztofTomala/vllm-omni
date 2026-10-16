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
from vllm_omni.config.stage_config import StageExecutionType
from vllm_omni.diffusion.models.alpamayo2_super.action_expert import (
    Alpamayo2SuperActionExpert,
    UnicycleAccelCurvatureActionSpace,
    _velocity_from_history,
)
from vllm_omni.diffusion.models.alpamayo2_super.pipeline_alpamayo2_super import (
    ACTIONS_KEY,
    Alpamayo2SuperPipeline,
)
from vllm_omni.diffusion.registry import _DIFFUSION_MODELS
from vllm_omni.model_executor.models.alpamayo2_super import Alpamayo2SuperConfig
from vllm_omni.model_executor.models.alpamayo2_super import openpi as alpamayo_openpi
from vllm_omni.model_executor.models.alpamayo2_super.alpamayo2_super import (
    Alpamayo2SuperForConditionalGeneration,
)
from vllm_omni.model_executor.models.alpamayo2_super.processing import (
    SPECIAL_TOKENS,
    build_trajectory_messages,
    encode_history_tokens,
)
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


def test_registered_as_reasoning_and_expert_stages():
    pipeline = OMNI_PIPELINES["alpamayo2_super"]
    reasoning, expert = pipeline.stages
    assert reasoning.execution_type == StageExecutionType.LLM_AR
    assert reasoning.omni_kv_config == {"need_send_cache": True}
    assert expert.execution_type == StageExecutionType.DIFFUSION and expert.input_sources == (0,)
    assert expert.omni_kv_config == {"need_recv_cache": True} and expert.final_output
    assert _OMNI_MODELS["Alpamayo2Super"][2] == "Alpamayo2SuperForConditionalGeneration"
    assert _DIFFUSION_MODELS[expert.model_arch][2] == "Alpamayo2SuperPipeline"


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
    config = Alpamayo2SuperConfig(include_camera_ids=True, frame_label="frame_num", tokens_per_history_traj=3)
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
    model.text_eos_ids = torch.tensor([TEXT_EOS])
    model._finished_rows = []
    model._kv_transfer_metadata = {}
    return model


def test_future_start_ends_request_and_publishes_kv_metadata():
    model = _make_model()
    input_ids = torch.tensor([5, 6, FUTURE_START, FUTURE_START + 1])
    # Request "a" is prefilling, "b" feeds back future_start, "c" is decoding.
    model.prepare_runner_inputs(
        input_ids=input_ids,
        positions=torch.stack([torch.arange(4), torch.arange(4) + 10, torch.arange(4) + 20]),
        inputs_embeds=None,
        req_ids=["a", "b", "c"],
        num_computed_tokens=[0, 100, 50],
        num_scheduled_tokens=[2, 1, 1],
        input_ids_buffer=input_ids,
    )
    assert model._finished_rows == [1]
    assert model.get_kv_transfer_metadata("b") == {"prefix_len": 101, "trigger_position": [2, 12, 22]}
    assert model.get_kv_transfer_metadata("b") is None
    assert model.get_kv_transfer_metadata("a") is None


def test_compute_logits_masks_trajectory_vocab_and_forces_end(monkeypatch):
    model = _make_model()
    monkeypatch.setattr(Qwen3VLForConditionalGeneration, "compute_logits", lambda self, hidden: torch.zeros(3, 32))
    model._finished_rows = [1]
    logits = model.compute_logits(torch.zeros(3, 4))
    assert torch.isinf(logits[:, 10:15]).all()
    assert torch.isinf(logits[:, TEXT_EOS]).all()
    assert logits[1].argmax().item() == FUTURE_END
    assert torch.isinf(logits[1, :FUTURE_END]).all()
    # State is consumed: the next step only applies the masks.
    logits = model.compute_logits(torch.zeros(3, 4))
    assert not torch.isinf(logits[1, FUTURE_END + 1])


_TINY_EXPERT = {
    "llm_config": {
        "model_type": "qwen3_vl_text",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 16,
        "rope_scaling": {"mrope_interleaved": True, "mrope_section": [2, 1, 1], "rope_type": "default"},
    },
    "action_in_proj_cfg": {"hidden_size": 16, "num_enc_layers": 2, "num_fourier_feats": 4},
    "action_space_cfg": {"n_waypoints": 64},
}


def _make_pipeline() -> Alpamayo2SuperPipeline:
    torch.manual_seed(0)
    pipeline = Alpamayo2SuperPipeline.__new__(Alpamayo2SuperPipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.expert = Alpamayo2SuperActionExpert(_TINY_EXPERT).eval()
    return pipeline


def _expert_request(prefix_len: int, seed: int) -> SimpleNamespace:
    generator = torch.Generator().manual_seed(prefix_len)
    keys = [torch.randn(prefix_len, 2, 8, generator=generator) for _ in range(2)]
    values = [torch.randn(prefix_len, 2, 8, generator=generator) for _ in range(2)]
    history_xyz = torch.zeros(16, 3)
    history_xyz[:, 0] = torch.arange(-15.0, 1.0)
    params = SimpleNamespace(
        past_key_values=SimpleNamespace(key_cache=keys, value_cache=values),
        kv_metadata={"seq_len": prefix_len, "prefix_len": prefix_len, "trigger_position": [prefix_len - 1] * 3},
        extra_args={
            "robot_obs": {"ego_history_xyz": history_xyz.tolist(), "ego_history_rot": [torch.eye(3).tolist()] * 16}
        },
        generator=None,
        seed=seed,
        num_inference_steps=2,
    )
    return SimpleNamespace(sampling_params=params, is_dummy_run=lambda: False)


def test_expert_batch_matches_requests_run_alone():
    pipeline = _make_pipeline()
    lengths_and_seeds = [(5, 1), (9, 2)]
    batched = pipeline(SimpleNamespace(requests=[_expert_request(*args) for args in lengths_and_seeds], num_reqs=2))
    for output, args in zip(batched, lengths_and_seeds):
        (alone,) = pipeline(SimpleNamespace(requests=[_expert_request(*args)], num_reqs=1))
        assert output.output[ACTIONS_KEY].shape == (64, 3)
        torch.testing.assert_close(output.output[ACTIONS_KEY], alone.output[ACTIONS_KEY], rtol=1e-4, atol=1e-4)
    assert not torch.equal(batched[0].output[ACTIONS_KEY], batched[1].output[ACTIONS_KEY])


def test_expert_reports_missing_kv_prefix():
    pipeline = _make_pipeline()
    request = _expert_request(5, 1)
    request.sampling_params.past_key_values = None
    (output,) = pipeline(SimpleNamespace(requests=[request], num_reqs=1))
    assert output.error is not None and "KV prefix" in output.error


def test_expert_rejects_out_of_range_steps():
    pipeline = _make_pipeline()
    request = _expert_request(5, 1)
    request.sampling_params.num_inference_steps = 101
    (output,) = pipeline(SimpleNamespace(requests=[request], num_reqs=1))
    assert "num_inference_steps" in output.error


def _make_prompt_builder() -> alpamayo_openpi.Alpamayo2SuperOpenPIPrompt:
    builder = alpamayo_openpi.Alpamayo2SuperOpenPIPrompt.__new__(alpamayo_openpi.Alpamayo2SuperOpenPIPrompt)
    builder.config = Alpamayo2SuperConfig()
    builder.tokenizer = None
    builder.frames_per_camera = 4
    builder.history_steps = 16
    return builder


def _observation(**overrides):
    obs = {
        "image_frames": np.zeros((2, 4, 3, 8, 8), dtype=np.uint8),
        "camera_indices": np.array([1, 6]),
        "ego_history_xyz": np.zeros((16, 3)),
        "ego_history_rot": np.tile(np.eye(3), (16, 1, 1)),
    }
    obs.update(overrides)
    return obs


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"image_frames": np.zeros((2, 3, 3, 8, 8), dtype=np.uint8)}, "image_frames"),
        ({"image_frames": np.zeros((2, 4, 3, 8, 8), dtype=np.float32)}, "image_frames"),
        ({"camera_indices": np.array([6, 1])}, "camera_indices"),
        ({"camera_indices": np.array([1, 1])}, "camera_indices"),
        ({"camera_indices": np.array([1, 9])}, "camera_indices"),
        ({"ego_history_xyz": np.zeros((3, 16))}, "ego_history"),
    ],
)
def test_openpi_prompt_rejects_invalid_observations(overrides, message):
    with pytest.raises(ValueError, match=message):
        _make_prompt_builder().build_prompt(_observation(**overrides))


def test_openpi_prompt_builds_reasoning_prompt(monkeypatch):
    calls: dict[str, object] = {}

    def fake_prompt(tokenizer, config, *, frames, camera_indices, ego_history_xyz):
        calls.update(frames=frames.shape, cameras=camera_indices, history=ego_history_xyz.shape)
        return [1, 2, 3]

    monkeypatch.setattr(alpamayo_openpi, "build_trajectory_prompt", fake_prompt)
    # A leading batch dimension on the history is accepted.
    prompt = _make_prompt_builder().build_prompt(_observation(ego_history_xyz=np.zeros((1, 16, 3))))
    assert calls == {"frames": (2, 4, 3, 8, 8), "cameras": [1, 6], "history": (16, 3)}
    assert prompt["prompt_token_ids"] == [1, 2, 3]
    assert len(prompt["multi_modal_data"]["image"]) == 8


def test_expert_samples_one_trajectory_per_reasoning_sample():
    # A reasoning request with n = 2 hands over one prefix per sample; sample i
    # draws its noise from seed + i, like separate requests with those seeds.
    pipeline = _make_pipeline()
    first, second = _expert_request(5, 7), _expert_request(9, 8)
    sampled = _expert_request(5, 7)
    sampled.sampling_params.past_key_values = [
        first.sampling_params.past_key_values,
        second.sampling_params.past_key_values,
    ]
    sampled.sampling_params.kv_metadata = [first.sampling_params.kv_metadata, second.sampling_params.kv_metadata]
    (output,) = pipeline(SimpleNamespace(requests=[sampled], num_reqs=1))
    assert output.output[ACTIONS_KEY].shape == (2, 64, 3)
    for sample, (prefix_len, seed) in enumerate([(5, 7), (9, 8)]):
        (alone,) = pipeline(SimpleNamespace(requests=[_expert_request(prefix_len, seed)], num_reqs=1))
        torch.testing.assert_close(output.output[ACTIONS_KEY][sample], alone.output[ACTIONS_KEY], rtol=1e-4, atol=1e-4)
