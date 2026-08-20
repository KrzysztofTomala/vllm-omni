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
from vllm_omni.model_executor.models.alpamayo1_5.tokenizer import (
    resolve_backbone_path,
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
                "processor_cls": pytest.importorskip("vllm.model_executor.models.qwen3_vl").Qwen3VLProcessor,
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


def test_bundled_backbone_is_preferred_for_offline_workspace(tmp_path):
    policy = tmp_path / "alpamayo"
    backbone = tmp_path / "cosmos-reason2"
    policy.mkdir()
    backbone.mkdir()
    (backbone / "config.json").write_text("{}", encoding="utf-8")

    assert resolve_backbone_path(
        "nvidia/Cosmos-Reason2-8B", model_path=policy
    ) == str(backbone)


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


def test_gather_prefix_cache_supports_fa3_packed_layout():
    blocks, kv_heads, block_size, head_dim = 3, 2, 2, 3
    key = torch.arange(
        blocks * kv_heads * block_size * head_dim, dtype=torch.float32
    ).reshape(blocks, kv_heads, block_size, head_dim)
    value = key + 1000
    paged_cache = torch.cat((key, value), dim=-1)
    block_table = torch.tensor([2, 0, 1])

    gathered = Alpamayo1_5ForConditionalGeneration._gather_prefix_cache(
        [paged_cache], block_table, seq_len=5
    )

    expected_key = key.index_select(0, block_table).permute(0, 2, 1, 3)
    expected_value = value.index_select(0, block_table).permute(0, 2, 1, 3)
    torch.testing.assert_close(
        gathered.layers[0].keys,
        expected_key.flatten(0, 1)[:5].transpose(0, 1).unsqueeze(0),
    )
    torch.testing.assert_close(
        gathered.layers[0].values,
        expected_value.flatten(0, 1)[:5].transpose(0, 1).unsqueeze(0),
    )


def test_static_action_cache_refreshes_prefix_and_rewinds_cursor():
    source_layer = SimpleNamespace(
        keys=torch.tensor([[[[1.0], [2.0]]]]),
        values=torch.tensor([[[[101.0], [102.0]]]]),
    )
    target_layer = SimpleNamespace(
        keys=torch.zeros(1, 1, 4, 1),
        values=torch.zeros(1, 1, 4, 1),
        cumulative_length=torch.tensor(4),
    )
    source = SimpleNamespace(layers=[source_layer])
    target = SimpleNamespace(layers=[target_layer])

    Alpamayo1_5ForConditionalGeneration._copy_action_prefix(target, source, 2)

    assert target_layer.keys.flatten().tolist() == [1.0, 2.0, 0.0, 0.0]
    assert target_layer.values.flatten().tolist() == [101.0, 102.0, 0.0, 0.0]
    assert target_layer.cumulative_length.item() == 2

    target_layer.cumulative_length.fill_(4)
    Alpamayo1_5ForConditionalGeneration._rewind_static_action_cache(target, 2)
    assert target_layer.cumulative_length.item() == 2

    target_layer.cumulative_length.fill_(4)
    Alpamayo1_5ForConditionalGeneration._rewind_static_action_cache(
        target,
        torch.tensor(1),
    )
    assert target_layer.cumulative_length.item() == 1


def test_fixed_action_attention_mask_hides_unused_static_cache():
    mask = Alpamayo1_5ForConditionalGeneration._make_action_attention_mask(
        sample_count=1,
        prefix_length=3,
        suffix_length=2,
        max_cache_len=8,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert mask.shape == (1, 1, 2, 8)
    assert torch.equal(mask[..., :5], torch.zeros(1, 1, 2, 5, dtype=torch.bfloat16))
    assert torch.all(mask[..., 5:] == torch.finfo(torch.bfloat16).min)


def test_fixed_action_attention_mask_rejects_short_cache():
    with pytest.raises(ValueError, match="need at least 5 tokens"):
        Alpamayo1_5ForConditionalGeneration._make_action_attention_mask(
            sample_count=1,
            prefix_length=3,
            suffix_length=2,
            max_cache_len=4,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )


def test_action_noise_advances_rng_like_nim():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    actual = Alpamayo1_5ForConditionalGeneration._make_action_noise(
        sample_count=1,
        device=torch.device("cpu"),
        generator=generator,
        nim_rng_advance_steps=7,
    )

    reference_generator = torch.Generator(device="cpu")
    reference_generator.manual_seed(42)
    probs = torch.ones((1, 1))
    for _ in range(7):
        torch.multinomial(probs, num_samples=1, generator=reference_generator)
    expected = torch.randn((1, 64, 2), generator=reference_generator)

    assert torch.equal(actual, expected)


def test_action_noise_rejects_negative_rng_advance():
    with pytest.raises(ValueError, match="must be non-negative"):
        Alpamayo1_5ForConditionalGeneration._make_action_noise(
            sample_count=1,
            device=torch.device("cpu"),
            generator=None,
            nim_rng_advance_steps=-1,
        )


def _minimal_policy_model() -> Alpamayo1_5ForConditionalGeneration:
    model = object.__new__(Alpamayo1_5ForConditionalGeneration)
    nn.Module.__init__(model)
    model.alpamayo_config = SimpleNamespace(
        traj_token_ids={"future_start": 155681, "future_end": 155683},
    )
    model._policy_prompt_lengths = {}
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


def test_nim_rng_compat_uses_expanded_prefill_length(monkeypatch):
    model = _minimal_policy_model()
    sample_kwargs = {}

    def fake_sample_actions(**kwargs):
        sample_kwargs.update(kwargs)
        return {"actions": torch.zeros(1, 64, 3)}

    monkeypatch.setattr(model, "_sample_actions", fake_sample_actions)
    common = {
        "hidden_states": torch.zeros(1, 8),
        "positions": torch.zeros(3, 1, dtype=torch.long),
        "sampling_extra_args": [
            {
                "robot_obs": {"ego_history_xyz": []},
                "_nim_action_rng_compat": True,
            }
        ],
    }
    prefill_context = RunnerKVCacheContext(
        caches=[],
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        sequence_lengths=(3091,),
        request_ids=("request-0",),
    )
    model.make_omni_output(
        **common,
        input_ids=torch.arange(3091),
        runner_kv_cache_context=prefill_context,
    )

    decode_context = RunnerKVCacheContext(
        caches=[],
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        sequence_lengths=(3111,),
        request_ids=("request-0",),
    )
    model.make_omni_output(
        **common,
        input_ids=torch.tensor([155681]),
        runner_kv_cache_context=decode_context,
    )

    assert sample_kwargs["extra_args"]["_prompt_token_count"] == 3091
    assert "request-0" not in model._policy_prompt_lengths
