# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration

from vllm_omni.model_executor.models.alpamayo2_super.alpamayo2_super import (
    Alpamayo2SuperForConditionalGeneration,
    alpamayo_flash_attention_3_forward,
)
from vllm_omni.model_executor.models.alpamayo2_super.configuration_alpamayo2_super import (
    Alpamayo2SuperConfig,
)
from vllm_omni.model_executor.models.alpamayo2_super.pipeline import (
    ALPAMAYO2_SUPER_PIPELINE,
)


def test_super_config_flattens_nested_qwen_config(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_type": "alpamayo2_super",
                "architectures": ["Alpamayo2Super"],
                "traj_ids": {
                    "history_id0": 151669,
                    "future_id0": 152669,
                    "future_start": 155681,
                },
                "vlm_config": {
                    "model_type": "qwen3_vl",
                    "image_token_id": 151655,
                    "vision_start_token_id": 151652,
                    "vision_end_token_id": 151653,
                    "text_config": {
                        "model_type": "qwen3_vl_text",
                        "hidden_size": 1024,
                        "intermediate_size": 3072,
                        "num_attention_heads": 8,
                        "num_hidden_layers": 2,
                        "num_key_value_heads": 2,
                        "vocab_size": 155776,
                    },
                    "vision_config": {
                        "model_type": "qwen3_vl",
                        "depth": 2,
                        "hidden_size": 128,
                        "intermediate_size": 256,
                        "num_heads": 4,
                        "out_hidden_size": 1024,
                        "patch_size": 14,
                        "spatial_merge_size": 2,
                        "temporal_patch_size": 2,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    config = Alpamayo2SuperConfig.from_pretrained(tmp_path)
    assert config.text_config.hidden_size == 1024
    assert config.vision_config.out_hidden_size == 1024
    assert config.traj_ids["future_start"] == 155681


def test_super_pipeline_is_single_stage_policy() -> None:
    assert ALPAMAYO2_SUPER_PIPELINE.model_type == "alpamayo2_super"
    assert len(ALPAMAYO2_SUPER_PIPELINE.stages) == 1
    assert ALPAMAYO2_SUPER_PIPELINE.stages[0].model_stage == "policy"
    assert ALPAMAYO2_SUPER_PIPELINE.stages[0].sampling_constraints[
        "stop_token_ids"
    ] == [155683]


def test_super_gathers_standard_paged_kv_layout() -> None:
    blocks = 3
    block_size = 2
    kv_heads = 2
    head_dim = 3
    cache = torch.arange(
        2 * blocks * block_size * kv_heads * head_dim,
        dtype=torch.float32,
    ).reshape(2, blocks, block_size, kv_heads, head_dim)
    block_table = torch.tensor([2, 0, 1])

    gathered = Alpamayo2SuperForConditionalGeneration._gather_prefix_cache(
        [cache], block_table, seq_len=5, target_dtype=torch.float32
    )

    expected_key = cache[0].index_select(0, block_table).flatten(0, 1)[:5]
    expected_value = cache[1].index_select(0, block_table).flatten(0, 1)[:5]
    assert gathered.get_seq_length() == 5
    torch.testing.assert_close(
        gathered.layers[0].keys,
        expected_key.transpose(0, 1).unsqueeze(0),
    )
    torch.testing.assert_close(
        gathered.layers[0].values,
        expected_value.transpose(0, 1).unsqueeze(0),
    )


def test_super_gathers_v026_packed_paged_kv_layout() -> None:
    blocks, kv_heads, block_size, head_dim = 3, 2, 2, 3
    key = torch.arange(
        blocks * kv_heads * block_size * head_dim, dtype=torch.float32
    ).reshape(blocks, kv_heads, block_size, head_dim)
    value = key + 1000
    cache = torch.cat((key, value), dim=-1)
    block_table = torch.tensor([2, 0, 1])

    gathered = Alpamayo2SuperForConditionalGeneration._gather_prefix_cache(
        [cache], block_table, seq_len=5, target_dtype=torch.float32
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


def test_super_dequantizes_raw_fp8_paged_cache_for_expert(
    monkeypatch,
) -> None:
    key = torch.tensor(
        [[[[1.0, -2.0], [0.5, 4.0]]], [[[3.0, -1.0], [2.0, 0.25]]]],
        dtype=torch.float8_e4m3fn,
    )
    value = torch.tensor(
        [[[[-1.0, 2.0], [-0.5, -4.0]]], [[[-3.0, 1.0], [-2.0, -0.25]]]],
        dtype=torch.float8_e4m3fn,
    )
    cache = torch.cat((key, value), dim=-1).view(torch.uint8)
    block_table = torch.tensor([1, 0])
    monkeypatch.setenv("NIM_ALPAMAYO_FP8_EXPERT_KV_CAST", "1")

    gathered = Alpamayo2SuperForConditionalGeneration._gather_prefix_cache(
        [cache],
        block_table,
        seq_len=3,
        target_dtype=torch.bfloat16,
    )

    expected_key = key.index_select(0, block_table).permute(0, 2, 1, 3)
    expected_value = value.index_select(0, block_table).permute(0, 2, 1, 3)
    torch.testing.assert_close(
        gathered.layers[0].keys,
        expected_key.flatten(0, 1)[:3].transpose(0, 1).unsqueeze(0).bfloat16(),
    )
    torch.testing.assert_close(
        gathered.layers[0].values,
        expected_value.flatten(0, 1)[:3].transpose(0, 1).unsqueeze(0).bfloat16(),
    )


def test_super_static_action_cache_refreshes_prefix_and_rewinds_cursor() -> None:
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

    Alpamayo2SuperForConditionalGeneration._copy_action_prefix(target, source, 2)
    assert target_layer.keys.flatten().tolist() == [1.0, 2.0, 0.0, 0.0]
    assert target_layer.values.flatten().tolist() == [101.0, 102.0, 0.0, 0.0]
    assert target_layer.cumulative_length.item() == 2

    target_layer.cumulative_length.fill_(4)
    Alpamayo2SuperForConditionalGeneration._rewind_static_action_cache(target, 2)
    assert target_layer.cumulative_length.item() == 2


def test_super_fixed_action_mask_hides_unused_static_cache() -> None:
    mask = Alpamayo2SuperForConditionalGeneration._make_action_attention_mask(
        sample_count=1,
        prefix_length=3,
        suffix_length=2,
        max_cache_len=8,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    assert mask.shape == (1, 1, 2, 8)
    assert torch.equal(
        mask[..., :5],
        torch.zeros(1, 1, 2, 5, dtype=torch.bfloat16),
    )
    assert torch.all(mask[..., 5:] == torch.finfo(torch.bfloat16).min)


def test_super_fixed_action_mask_rejects_short_cache() -> None:
    with pytest.raises(ValueError, match="need at least 5 tokens"):
        Alpamayo2SuperForConditionalGeneration._make_action_attention_mask(
            sample_count=1,
            prefix_length=3,
            suffix_length=2,
            max_cache_len=4,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )


def test_super_flash_attention_3_falls_back_for_cpu(monkeypatch) -> None:
    expected = torch.randn(1, 2, 4, 3)
    calls = []

    def fake_sdpa(*args, **kwargs):
        calls.append((args, kwargs))
        return expected, None

    monkeypatch.setattr(
        "vllm_omni.model_executor.models.alpamayo2_super.alpamayo2_super."
        "sdpa_attention_forward",
        fake_sdpa,
    )
    query = torch.randn(1, 4, 2, 3)
    key = torch.randn(1, 2, 5, 3)
    output, weights = alpamayo_flash_attention_3_forward(
        SimpleNamespace(),
        query,
        key,
        key,
        None,
        is_causal=False,
    )

    assert output is expected
    assert weights is None
    assert len(calls) == 1


def test_super_policy_does_not_export_vlm_hidden_prefix() -> None:
    model = Alpamayo2SuperForConditionalGeneration
    assert model.have_multimodal_outputs
    assert model.needs_runner_kv_cache
    assert not model.requires_full_prefix_cached_hidden_states
    assert not model.omni_pooler_payload_include_hidden


def test_super_policy_masks_text_eos_until_action_boundary(monkeypatch) -> None:
    model = object.__new__(Alpamayo2SuperForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.alpamayo_config = SimpleNamespace(
        traj_ids={"history_id0": 4, "future_id0": 8, "future_end": 14},
        traj_vocab_size=3,
    )
    model._force_future_end = False
    model._policy_generation_active = True
    model._text_eos_ids = (1, 2)
    logits = torch.zeros(1, 16)
    monkeypatch.setattr(
        Qwen3VLForConditionalGeneration,
        "compute_logits",
        lambda _self, _hidden_states: logits.clone(),
    )

    result = model.compute_logits(torch.zeros(1, 1))

    assert torch.all(result[..., [1, 2]] == -torch.inf)
    assert torch.all(result[..., 4:7] == -torch.inf)
    assert result[..., 3].item() == 0


def test_super_action_boundary_uses_first_mrope_position_not_graph_padding() -> None:
    positions = torch.tensor(
        [
            [101, 0, 0, 0],
            [202, 0, 0, 0],
            [303, 0, 0, 0],
        ]
    )

    actual = Alpamayo2SuperForConditionalGeneration._action_boundary_position(
        positions,
        device=torch.device("cpu"),
    )

    assert torch.equal(actual, torch.tensor([[101], [202], [303]]))


def test_super_action_boundary_expands_first_linear_position() -> None:
    positions = torch.tensor([101, 0, 0, 0])

    actual = Alpamayo2SuperForConditionalGeneration._action_boundary_position(
        positions,
        device=torch.device("cpu"),
    )

    assert torch.equal(actual, torch.tensor([[101], [101], [101]]))
