# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.alpamayo1_5.alpamayo1_5 import (
    Alpamayo1_5ForConditionalGeneration,
    Alpamayo1_5ProcessingInfo,
)
from vllm_omni.model_executor.models.alpamayo1_5.processing import (
    SPECIAL_TOKENS,
    create_policy_messages,
    create_vqa_messages,
    encode_history_trajectory,
    extend_tokenizer,
    fuse_history_tokens,
)
from vllm_omni.model_executor.models.runner_context import RunnerKVCacheContext

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeTokenizer:
    def __init__(self) -> None:
        self.tokens = {f"base-{index}": index for index in range(151669)}
        # Qwen3-VL already owns this token at ID 151655, so adding Alpamayo's
        # full special-token list does not grow the vocabulary for this entry.
        del self.tokens["base-151655"]
        self.tokens["<|image_pad|>"] = 151655

    def add_tokens(self, values, special_tokens=False):
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


def test_processing_info_uses_vllm_processor_cache(monkeypatch):
    config = SimpleNamespace(
        vlm_name_or_path="backbone",
        min_pixels=123,
        max_pixels=456,
    )
    tokenizer = object()
    expected = object()
    calls = []

    def fake_cached_get_processor(model, **kwargs):
        calls.append((model, kwargs))
        return expected

    monkeypatch.setattr(
        "vllm_omni.model_executor.models.alpamayo1_5.alpamayo1_5.cached_get_processor",
        fake_cached_get_processor,
    )
    info = object.__new__(Alpamayo1_5ProcessingInfo)
    monkeypatch.setattr(info, "get_hf_config", lambda: config)
    monkeypatch.setattr(info, "get_tokenizer", lambda: tokenizer)

    assert info.get_hf_processor(use_fast=False, device="cuda") is expected
    assert calls == [
        (
            "backbone",
            {
                "processor_cls": pytest.importorskip(
                    "vllm.model_executor.models.qwen3_vl"
                ).Qwen3VLProcessor,
                "tokenizer": tokenizer,
                "min_pixels": 123,
                "max_pixels": 456,
                "use_fast": False,
            },
        )
    ]


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

    assert tokens.shape == (48,)
    assert tokens.min() >= 154669
    assert tokens.max() <= 155668
    input_ids = torch.tensor([1] + [155684] * 48 + [2])
    fused = fuse_history_tokens(input_ids, history)
    assert torch.equal(fused[1:-1], tokens)
    assert fused[[0, -1]].tolist() == [1, 2]


def test_policy_and_vqa_messages_use_training_control_tokens():
    images = [SimpleNamespace(name=f"frame-{index}") for index in range(4)]
    policy = create_policy_messages(images, camera_indices=[1], navigation="turn left")
    policy_text = policy[1]["content"][-1]["text"]
    assert policy_text.count(SPECIAL_TOKENS["traj_history"]) == 48
    assert SPECIAL_TOKENS["route_start"] + "turn left" in policy_text
    assert policy[-1]["content"][0]["text"] == SPECIAL_TOKENS["cot_start"]

    vqa = create_vqa_messages(images, "What is ahead?", camera_indices=[1])
    vqa_text = vqa[1]["content"][-1]["text"]
    assert vqa_text == "<|question_start|>What is ahead?<|question_end|>"
    assert vqa[-1]["content"][0]["text"] == SPECIAL_TOKENS["answer_start"]


def test_gather_prefix_cache_preserves_block_order():
    key = torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2, 1, 1)
    value = key + 100
    paged_cache = torch.stack((key, value))

    gathered = Alpamayo1_5ForConditionalGeneration._gather_prefix_cache(
        [paged_cache],
        block_table=torch.tensor([1, 0], dtype=torch.int32),
        seq_len=3,
    )

    assert gathered.get_seq_length() == 3
    assert gathered.layers[0].keys.flatten().tolist() == [2.0, 3.0, 0.0]
    assert gathered.layers[0].values.flatten().tolist() == [102.0, 103.0, 100.0]


def _minimal_policy_model() -> Alpamayo1_5ForConditionalGeneration:
    model = object.__new__(Alpamayo1_5ForConditionalGeneration)
    nn.Module.__init__(model)
    model.alpamayo_config = SimpleNamespace(
        traj_token_ids={"future_start": 155681, "future_end": 155683},
    )
    model._force_future_end = False
    return model


def test_policy_observation_rejects_multiple_requests():
    model = _minimal_policy_model()

    with pytest.raises(RuntimeError, match="max_num_seqs=1"):
        model._policy_observation(
            [
                {"robot_obs": {"ego_history_xyz": []}},
                {"robot_obs": {"ego_history_xyz": []}},
            ]
        )


def test_action_trigger_requires_runner_kv_context(monkeypatch):
    model = _minimal_policy_model()
    monkeypatch.setattr(
        model,
        "_sample_actions",
        lambda **kwargs: {"actions": torch.zeros(1, 64, 3)},
    )
    kwargs = {
        "hidden_states": torch.zeros(1, 8),
        "input_ids": torch.tensor([155681]),
        "positions": torch.zeros(3, 1, dtype=torch.long),
        "sampling_extra_args": [{"robot_obs": {"ego_history_xyz": []}}],
    }

    with pytest.raises(RuntimeError, match="requires runner KV-cache"):
        model.make_omni_output(**kwargs)

    context = RunnerKVCacheContext(
        caches=[],
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        sequence_lengths=(1,),
        request_ids=("request-0",),
    )
    output = model.make_omni_output(
        **kwargs,
        runner_kv_cache_context=context,
    )

    assert output.multimodal_outputs is not None
    assert output.multimodal_outputs["actions"].shape == (1, 64, 3)
    assert model._force_future_end is True
