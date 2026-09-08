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
    adapter.model_config = SimpleNamespace(
        traj_ids={"future_start": 155681, "future_end": 155683},
        tokens_per_history_traj=16,
        tokens_per_future_traj=64,
        include_camera_ids=True,
        frame_label="frame_num",
    )
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
    assert request.sampling_params.max_tokens == 512
    assert request.sampling_params.stop_token_ids == [155681]
    assert text_task_calls[0][1] == "meta_action"


def test_vqa_request_bypasses_generic_string_prompt_parser(text_task_calls):
    request = _adapter().build_vqa_request(_observation(), question="What is ahead?", request_id="vqa-1")

    assert "prompt" not in request.prompt
    assert request.prompt["prompt_token_ids"] == [18, 2]
    assert request.sampling_params.max_tokens == 256
    assert request.sampling_params.stop_token_ids == []
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
    assert request.sampling_params.stop_token_ids == []


def test_grounding_uses_vqa_prompt_without_trajectory_fusion(text_task_calls):
    request = _adapter().build_text_task_request(_observation(), task="grounding", request_id="grounding-1")

    assert request.prompt["prompt_token_ids"] == [25, 2]
    assert request.sampling_params.max_tokens == 512
    assert request.sampling_params.stop_token_ids == []
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
    assert request.sampling_params.max_tokens == 128
    assert request.sampling_params.stop_token_ids == [155683]
    # Each VLM child owns one expert result. This must not be 3, which would
    # recreate three trajectories from a single sampled reasoning string.
    assert request.sampling_params.extra_args["num_traj_samples"] == 1
    assert request.sampling_params.extra_args["_parallel_sample_count"] == 3
    assert request.sampling_params.extra_args["_batch_action_expert"] is True
    assert request.sampling_params.extra_args["_action_expert_max_batch_size"] == 3
    assert "_sampling_seed" not in request.sampling_params.extra_args


def test_policy_request_caps_default_action_expert_batch_at_three(monkeypatch):
    adapter = _adapter()
    adapter.policy_config = {"num_trajectory_samples": 7}
    monkeypatch.setenv("NIM_PRECISION", "bf16")
    monkeypatch.delenv("NIM_ALPAMAYO_ACTION_EXPERT_MAX_BATCH_SIZE", raising=False)
    monkeypatch.setattr(adapter, "_policy_prompt", lambda _observation: ([1, 2], []))

    request = adapter.build_request(_observation(), request_id="policy-1", session_id="session-1", reset=True)

    assert request.sampling_params.n == 7
    assert request.sampling_params.extra_args["_parallel_sample_count"] == 7
    assert request.sampling_params.extra_args["_action_expert_max_batch_size"] == 3


def test_policy_request_keeps_fp8_fully_batched_by_default(monkeypatch):
    adapter = _adapter()
    adapter.policy_config = {"num_trajectory_samples": 7}
    monkeypatch.setenv("NIM_PRECISION", "fp8")
    monkeypatch.delenv("NIM_ALPAMAYO_ACTION_EXPERT_MAX_BATCH_SIZE", raising=False)
    monkeypatch.setattr(adapter, "_policy_prompt", lambda _observation: ([1, 2], []))

    request = adapter.build_request(_observation(), request_id="policy-1", session_id="session-1", reset=True)

    assert request.sampling_params.extra_args["_action_expert_max_batch_size"] == 7


def test_policy_request_user_expert_batch_override_has_priority(monkeypatch):
    adapter = _adapter()
    adapter.policy_config = {
        "num_trajectory_samples": 7,
        "action_expert_max_batch_size": 1,
    }
    monkeypatch.setenv("NIM_PRECISION", "bf16")
    monkeypatch.setenv("NIM_ALPAMAYO_ACTION_EXPERT_MAX_BATCH_SIZE", "6")
    monkeypatch.setattr(adapter, "_policy_prompt", lambda _observation: ([1, 2], []))

    request = adapter.build_request(_observation(), request_id="policy-1", session_id="session-1", reset=True)

    assert request.sampling_params.extra_args["_action_expert_max_batch_size"] == 6


def test_policy_request_profile_expert_batch_override_has_priority(monkeypatch):
    adapter = _adapter()
    adapter.policy_config = {
        "num_trajectory_samples": 7,
        "precision": "bf16",
        "action_expert_max_batch_size": 1,
    }
    monkeypatch.delenv("NIM_ALPAMAYO_ACTION_EXPERT_MAX_BATCH_SIZE", raising=False)
    monkeypatch.setattr(adapter, "_policy_prompt", lambda _observation: ([1, 2], []))

    request = adapter.build_request(_observation(), request_id="policy-1", session_id="session-1", reset=True)

    assert request.sampling_params.extra_args["_action_expert_max_batch_size"] == 1


def test_policy_request_profile_precision_has_priority_over_environment(monkeypatch):
    adapter = _adapter()
    adapter.policy_config = {
        "num_trajectory_samples": 7,
        "precision": "fp8",
    }
    monkeypatch.setenv("NIM_PRECISION", "bf16")
    monkeypatch.delenv("NIM_ALPAMAYO_ACTION_EXPERT_MAX_BATCH_SIZE", raising=False)
    monkeypatch.setattr(adapter, "_policy_prompt", lambda _observation: ([1, 2], []))

    request = adapter.build_request(_observation(), request_id="policy-1", session_id="session-1", reset=True)

    assert request.sampling_params.extra_args["_action_expert_max_batch_size"] == 7


def test_policy_request_supports_legacy_precision_environment(monkeypatch):
    adapter = _adapter()
    adapter.policy_config = {"num_trajectory_samples": 7}
    monkeypatch.delenv("NIM_PRECISION", raising=False)
    monkeypatch.setenv("NIM_ALPAMAYO_PRECISION", "fp8")
    monkeypatch.delenv("NIM_ALPAMAYO_ACTION_EXPERT_MAX_BATCH_SIZE", raising=False)
    monkeypatch.setattr(adapter, "_policy_prompt", lambda _observation: ([1, 2], []))

    request = adapter.build_request(_observation(), request_id="policy-1", session_id="session-1", reset=True)

    assert request.sampling_params.extra_args["_action_expert_max_batch_size"] == 7


@pytest.mark.parametrize("value", ["not-an-int", "0", "-1"])
def test_policy_request_rejects_invalid_user_expert_batch_override(monkeypatch, value):
    adapter = _adapter()
    adapter.policy_config = {"num_trajectory_samples": 7}
    monkeypatch.setenv("NIM_ALPAMAYO_ACTION_EXPERT_MAX_BATCH_SIZE", value)
    monkeypatch.setattr(adapter, "_policy_prompt", lambda _observation: ([1, 2], []))

    with pytest.raises(ValueError, match="NIM_ALPAMAYO_ACTION_EXPERT_MAX_BATCH_SIZE"):
        adapter.build_request(_observation(), request_id="policy-1", session_id="session-1", reset=True)


@pytest.mark.parametrize("value", ["not-an-int", 0, -1])
def test_policy_request_rejects_invalid_profile_expert_batch_override(monkeypatch, value):
    adapter = _adapter()
    adapter.policy_config = {
        "num_trajectory_samples": 7,
        "action_expert_max_batch_size": value,
    }
    monkeypatch.delenv("NIM_ALPAMAYO_ACTION_EXPERT_MAX_BATCH_SIZE", raising=False)
    monkeypatch.setattr(adapter, "_policy_prompt", lambda _observation: ([1, 2], []))

    with pytest.raises(ValueError, match="action_expert_max_batch_size"):
        adapter.build_request(_observation(), request_id="policy-1", session_id="session-1", reset=True)


@pytest.fixture
def policy_prompt_fakes(monkeypatch):
    """Fake the released prompt builders so prompt assembly is observable."""
    calls = {"create_messages": [], "build_conversation": []}
    helper = ModuleType("alpamayo2_super.helper")

    def create_messages(data, _model_config):
        calls["create_messages"].append(data)
        return [{"role": "user", "content": ["base"], "prompt": "image|traj_history|prompt"}]

    helper.create_messages = create_messages
    chat_template = ModuleType("alpamayo2_super.chat_template")
    conversation = ModuleType("alpamayo2_super.chat_template.conversation")

    def build_conversation(*, data, components_order, **kwargs):
        calls["build_conversation"].append({"data": data, "components_order": list(components_order), **kwargs})
        text = "|".join(components_order)
        if "nav_instruction" in components_order:
            text += ":" + data["nav_text"][0]
        return [
            {"role": "user", "content": [text], "prompt": text},
            {"role": "assistant", "content": []},
        ]

    conversation.build_conversation = build_conversation
    models = ModuleType("alpamayo2_super.models")
    utils = ModuleType("alpamayo2_super.models.utils")

    def fuse_traj_tokens(_history, _future, input_ids, _trajectory_data, _traj_ids):
        return input_ids + 1000

    utils.fuse_traj_tokens = fuse_traj_tokens
    package = ModuleType("alpamayo2_super")
    package.helper = helper
    package.chat_template = chat_template
    package.models = models
    chat_template.conversation = conversation
    models.utils = utils
    for name, module in {
        "alpamayo2_super": package,
        "alpamayo2_super.helper": helper,
        "alpamayo2_super.chat_template": chat_template,
        "alpamayo2_super.chat_template.conversation": conversation,
        "alpamayo2_super.models": models,
        "alpamayo2_super.models.utils": utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return calls


def _legacy_policy_prompt_ids(adapter, observation):
    """The pre-navigation-CFG prompt path, reproduced verbatim as the reference."""
    from alpamayo2_super.helper import create_messages
    from alpamayo2_super.models.utils import fuse_traj_tokens

    frames, camera_indices, _ = adapter._frames(observation)
    data = {"image_frames": frames, "camera_indices": camera_indices}
    messages = create_messages(data, adapter.model_config)
    has_assistant = messages[-1]["role"] == "assistant" and bool(messages[-1]["content"])
    prompt = adapter.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not has_assistant,
        continue_final_message=has_assistant,
    )
    prompt_ids = torch.tensor([adapter.tokenizer.encode(prompt)])
    return fuse_traj_tokens(
        adapter.history_tokenizer,
        adapter.future_tokenizer,
        prompt_ids,
        {
            "ego_history_xyz": torch.as_tensor(observation["ego_history_xyz"]),
            "ego_history_rot": torch.as_tensor(observation["ego_history_rot"]),
        },
        adapter.model_config.traj_ids,
    )[0].tolist()


def _nav_observation(text="turn left at the light", **extra):
    observation = _observation()
    observation["nav_text"] = text
    observation.update(extra)
    return observation


def test_policy_prompt_without_nav_is_bit_identical_to_create_messages_path(policy_prompt_fakes):
    adapter = _adapter()
    for observation in (_observation(), _nav_observation(text=""), _nav_observation(text=None)):
        prompt_ids, images = adapter._policy_prompt(observation)
        assert prompt_ids == _legacy_policy_prompt_ids(adapter, observation)
        assert len(images) == 4
    assert policy_prompt_fakes["build_conversation"] == []

    request = adapter.build_request(_observation(), request_id="policy-1", session_id="s", reset=True)
    assert request.prompt["prompt_token_ids"] == _legacy_policy_prompt_ids(adapter, _observation())
    assert "_nav_guidance_weight" not in request.sampling_params.extra_args
    assert "_nav_cfg_max_hold_steps" not in request.sampling_params.extra_args
    assert "_nav_cfg_group" not in request.sampling_params.extra_args
    assert request.sampling_params.max_tokens == 128


def test_policy_prompt_with_nav_inserts_instruction_between_history_and_prompt(policy_prompt_fakes):
    adapter = _adapter()
    observation = _nav_observation()

    prompt_ids, _ = adapter._policy_prompt(observation)

    assert len(policy_prompt_fakes["build_conversation"]) == 1
    call = policy_prompt_fakes["build_conversation"][0]
    assert call["components_order"] == ["image", "traj_history", "nav_instruction", "prompt"]
    assert call["data"]["nav_text"] == ["turn left at the light"]
    assert call["components_prompt"] == ["cot", "traj_future"]
    assert call["generation_mode"] is True
    assert call["include_camera_ids"] is True
    assert call["include_frame_nums"] is True
    assert call["num_tokens_per_history_traj"] == 16
    assert call["num_tokens_per_future_traj"] == 64
    assert torch.equal(call["camera_ids"], torch.tensor([1]))
    assert prompt_ids != _legacy_policy_prompt_ids(adapter, observation)


def test_build_prompt_pair_unguided_equals_no_nav_prompt(policy_prompt_fakes):
    adapter = _adapter()
    observation = _nav_observation()

    guided, unguided = adapter.build_prompt_pair(observation)

    assert unguided == _legacy_policy_prompt_ids(adapter, _observation())
    assert unguided == _legacy_policy_prompt_ids(adapter, observation)
    assert guided != unguided
    assert guided == adapter._policy_prompt(observation)[0]
    assert adapter.build_prompt_pair(_observation()) == (unguided, unguided)


def test_policy_request_with_nav_requests_unguided_twin_by_default(policy_prompt_fakes):
    adapter = _adapter()

    request = adapter.build_request(_nav_observation(), request_id="policy-1", session_id="s", reset=True)

    extra = request.sampling_params.extra_args
    assert extra["_nav_guidance_weight"] == 3.0
    assert extra["_nav_cfg_max_hold_steps"] == 128
    assert extra["_nav_cfg_group"] == "policy-1"
    assert request.sampling_params.max_tokens == 128 + 128
    assert request.sampling_params.n == 1
    assert request.sampling_params.stop_token_ids == [155683]
    assert request.prompt["multi_modal_uuids"]["image"] == [f"policy-1:image:{index}" for index in range(4)]


def test_policy_request_weight_one_does_not_request_twin(policy_prompt_fakes):
    adapter = _adapter()

    request = adapter.build_request(
        _nav_observation(nav_guidance_weight=1.0), request_id="policy-1", session_id="s", reset=True
    )

    extra = request.sampling_params.extra_args
    assert "_nav_guidance_weight" not in extra
    assert "_nav_cfg_max_hold_steps" not in extra
    assert request.sampling_params.max_tokens == 128
    # The prompt still carries the navigation instruction.
    assert policy_prompt_fakes["build_conversation"][0]["components_order"][2] == "nav_instruction"


def test_policy_request_nav_weight_precedence(policy_prompt_fakes):
    adapter = _adapter()
    adapter.model_config.expert_config = {"diffusion_cfg": {"inference_guidance_weight": 2.5}}

    def weight(observation):
        return adapter.build_request(
            observation, request_id="p", session_id="s", reset=True
        ).sampling_params.extra_args["_nav_guidance_weight"]

    assert weight(_nav_observation()) == 2.5
    adapter.model_config.nav_guidance_weight = 4.0
    assert weight(_nav_observation()) == 4.0
    adapter.policy_config = {"nav_guidance_weight": 5.0, "nav_cfg_max_hold_steps": 7}
    assert weight(_nav_observation()) == 5.0
    assert weight(_nav_observation(nav_guidance_weight=6.0)) == 6.0
    request = adapter.build_request(_nav_observation(), request_id="p", session_id="s", reset=True)
    assert request.sampling_params.extra_args["_nav_cfg_max_hold_steps"] == 7
    assert request.sampling_params.max_tokens == 128 + 7
    with pytest.raises(ValueError, match="finite"):
        weight(_nav_observation(nav_guidance_weight=float("nan")))


def test_policy_request_prefers_content_addressed_image_uuids(policy_prompt_fakes):
    adapter = _adapter()
    uuids = [f"alpamayo-image-sha256:{index:02d}" for index in range(4)]

    request = adapter.build_request(
        _observation() | {"image_uuids": uuids}, request_id="policy-1", session_id="s", reset=True
    )
    assert request.prompt["multi_modal_uuids"]["image"] == uuids

    fallback = adapter.build_request(_observation() | {"image_uuids": None}, request_id="p", session_id="s", reset=True)
    assert fallback.prompt["multi_modal_uuids"]["image"] == [f"p:image:{index}" for index in range(4)]

    with pytest.raises(ValueError, match="one entry per image"):
        adapter.build_request(_observation() | {"image_uuids": uuids[:2]}, request_id="p", session_id="s", reset=True)


def test_unguided_twin_request_replays_guided_reasoning_after_unguided_prompt(policy_prompt_fakes):
    adapter = _adapter()
    adapter.policy_config = {"num_trajectory_samples": 3, "diffusion_steps": 4, "manual_action_cudagraph": True}
    uuids = [f"alpamayo-image-sha256:{index:02d}" for index in range(4)]
    observation = _nav_observation(image_uuids=uuids)
    guided = adapter.build_request(observation, request_id="parent", session_id="s", reset=True)
    _, unguided_ids = adapter.build_prompt_pair(observation)

    twin = adapter.build_unguided_twin_request(
        observation,
        guided_child_request_id="1_parent",
        generated_token_ids=[11, 12, 13, 155681, 155681],
        request_id="twin-1",
    )

    assert twin.request_id == "twin-1"
    assert twin.prompt["prompt_token_ids"] == unguided_ids + [11, 12, 13]
    assert twin.prompt["multi_modal_uuids"] == guided.prompt["multi_modal_uuids"]
    assert len(twin.prompt["multi_modal_data"]["image"]) == 4
    params = twin.sampling_params
    assert params.n == 1
    assert params.temperature == 0
    assert params.max_tokens == 128 + 2
    assert params.output_kind == RequestOutputKind.FINAL_ONLY
    assert params.stop_token_ids == [155683]
    extra = params.extra_args
    assert extra["_nav_cfg_role"] == "unguided"
    assert extra["_nav_cfg_partner"] == "1_parent"
    assert extra["_nav_cfg_group"] == "parent"
    assert extra["_nav_cfg_partner_index"] == 1
    assert extra["_force_first_token"] == 155681
    assert extra["_nav_cfg_max_hold_steps"] == 128
    assert "_nav_cfg_prompt_len" not in extra
    assert extra["session_id"] == "twin-1"
    assert extra["reset"] is True
    assert extra["robot_obs"] == guided.sampling_params.extra_args["robot_obs"]
    assert extra.get("_parallel_sample_count", 1) == 1
    assert "_nav_guidance_weight" not in extra
    for key in (
        "num_traj_samples",
        "_batch_action_expert",
        "_action_expert_max_batch_size",
        "diffusion_steps",
        "_static_expert_cache",
        "_compile_expert",
        "_manual_action_cudagraph",
        "_static_expert_cache_max_len",
    ):
        assert extra[key] == guided.sampling_params.extra_args[key], key

    with pytest.raises(ValueError, match="future_start"):
        adapter.build_unguided_twin_request(
            observation,
            guided_child_request_id="1_parent",
            generated_token_ids=[11, 155681, 12],
            request_id="twin-2",
        )


def test_unguided_twin_request_parses_child_id_on_first_underscore(policy_prompt_fakes):
    adapter = _adapter()
    observation = _nav_observation()

    def keys(child_id):
        extra = adapter.build_unguided_twin_request(
            observation, guided_child_request_id=child_id, generated_token_ids=[], request_id="t"
        ).sampling_params.extra_args
        return extra["_nav_cfg_group"], extra["_nav_cfg_partner_index"], extra["_nav_cfg_partner"]

    assert keys("2_nim-trajectory_12") == ("nim-trajectory_12", 2, "2_nim-trajectory_12")
    assert keys("0_nim-trajectory-3") == ("nim-trajectory-3", 0, "0_nim-trajectory-3")
    # A bare request id is the single child of an n=1 request.
    assert keys("nim-trajectory-3") == ("nim-trajectory-3", 0, "nim-trajectory-3")
