# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from vllm.sampling_params import RequestOutputKind

from vllm_omni.model_executor.models.alpamayo2_super.openpi import (
    Alpamayo2SuperOpenPIRequestAdapter,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["prompt"]

    def encode(self, prompt):
        return [len(prompt), 2]


@pytest.fixture
def text_task_calls(monkeypatch):
    calls = []
    text_tasks = ModuleType("alpamayo2_super.text_tasks")
    text_tasks.DEFAULT_GROUNDING_QUESTION = "find the lead vehicle"

    def build_text_task_messages(data, _model_config, task):
        calls.append((data, task))
        return [{"prompt": f"{task}:{data.get('question', '')}"}]

    text_tasks.build_text_task_messages = build_text_task_messages
    models = ModuleType("alpamayo2_super.models")
    utils = ModuleType("alpamayo2_super.models.utils")

    def fuse_traj_tokens(_history, _future, _input_ids, trajectory_data, _traj_ids):
        calls.append((trajectory_data, "fused"))
        return torch.tensor([[91, 92]])

    utils.fuse_traj_tokens = fuse_traj_tokens
    package = ModuleType("alpamayo2_super")
    package.models = models
    models.utils = utils
    monkeypatch.setitem(sys.modules, "alpamayo2_super", package)
    monkeypatch.setitem(sys.modules, "alpamayo2_super.text_tasks", text_tasks)
    monkeypatch.setitem(sys.modules, "alpamayo2_super.models", models)
    monkeypatch.setitem(sys.modules, "alpamayo2_super.models.utils", utils)
    return calls


def _adapter():
    adapter = object.__new__(Alpamayo2SuperOpenPIRequestAdapter)
    adapter.policy_config = {}
    adapter.tokenizer = _FakeTokenizer()
    adapter.history_tokenizer = object()
    adapter.future_tokenizer = object()
    adapter.model_config = SimpleNamespace(traj_ids={"future_start": 155681})
    return adapter


def _observation():
    return {
        "image_frames": torch.zeros(1, 4, 3, 8, 8),
        "camera_indices": torch.tensor([1]),
        "num_frames_per_camera": 4,
        "ego_history_xyz": torch.zeros(1, 1, 16, 3),
        "ego_history_rot": torch.zeros(1, 1, 16, 3, 3),
    }


def test_meta_action_request_fuses_history_and_stops_before_policy(text_task_calls):
    request = _adapter().build_text_task_request(_observation(), task="meta_action", request_id="meta-1")

    assert request.prompt["prompt_token_ids"] == [91, 92]
    assert len(request.prompt["multi_modal_data"]["image"]) == 4
    assert request.sampling_params.stop_token_ids == [155681]
    assert text_task_calls[0][1] == "meta_action"


def test_vqa_request_bypasses_generic_string_prompt_parser(text_task_calls):
    request = _adapter().build_vqa_request(_observation(), question="What is ahead?", request_id="vqa-1")

    assert "prompt" not in request.prompt
    assert request.prompt["prompt_token_ids"] == [18, 2]
    assert len(request.prompt["multi_modal_data"]["image"]) == 4
    data, task = text_task_calls[0]
    assert task == "vqa"
    assert data["question"] == "What is ahead?"


def test_auto_labeling_request_normalizes_future_trajectory(text_task_calls):
    request = _adapter().build_text_task_request(
        _observation(),
        task="auto_labeling",
        request_id="auto-1",
        future_xyz=torch.zeros(64, 3),
        future_rot=torch.zeros(64, 3, 3),
    )

    fused_data = text_task_calls[1][0]
    assert fused_data["ego_future_xyz"].shape == (1, 1, 64, 3)
    assert fused_data["ego_future_rot"].shape == (1, 1, 64, 3, 3)
    assert request.sampling_params.max_tokens == 1024


def test_grounding_uses_vqa_prompt_without_trajectory_fusion(text_task_calls):
    request = _adapter().build_text_task_request(_observation(), task="grounding", request_id="grounding-1")

    assert request.prompt["prompt_token_ids"] == [25, 2]
    assert len(text_task_calls) == 1
    data, task = text_task_calls[0]
    assert task == "vqa"
    assert data["question"] == "find the lead vehicle"


def test_policy_request_uses_independent_vllm_completions(monkeypatch):
    adapter = _adapter()
    adapter.policy_config = {
        "num_trajectory_samples": 3,
        "seed": 100,
        "temperature": 0.7,
    }
    adapter.model_config.traj_ids["future_end"] = 155683
    monkeypatch.setattr(adapter, "_policy_prompt", lambda _observation: ([1, 2], []))

    request = adapter.build_request(_observation(), request_id="policy-1", session_id="session-1", reset=True)

    assert request.sampling_params.n == 3
    assert request.sampling_params.output_kind == RequestOutputKind.FINAL_ONLY
    assert request.sampling_params.seed == 100
    assert request.sampling_params.temperature == 0.7
    # Each VLM child owns one expert result. This must not be 3, which would
    # recreate three trajectories from a single sampled reasoning string.
    assert request.sampling_params.extra_args["num_traj_samples"] == 1
    assert request.sampling_params.extra_args["_parallel_sample_count"] == 3
    assert request.sampling_params.extra_args["_batch_action_expert"] is True
    assert "_sampling_seed" not in request.sampling_params.extra_args
