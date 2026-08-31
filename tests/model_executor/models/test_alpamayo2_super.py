# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.alpamayo2_super.alpamayo2_super import (
    Alpamayo2SuperForConditionalGeneration,
)
from vllm_omni.model_executor.models.alpamayo2_super.configuration_alpamayo2_super import (
    Alpamayo2SuperConfig,
)
from vllm_omni.model_executor.models.alpamayo2_super.pipeline import (
    ALPAMAYO2_SUPER_PIPELINE,
)
from vllm_omni.model_executor.models.runner_context import RunnerKVCacheContext


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
        [cache], block_table, seq_len=5
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


def test_super_gathers_concatenated_paged_kv_layout() -> None:
    blocks = 3
    block_size = 2
    kv_heads = 2
    head_dim = 3
    cache = torch.arange(
        blocks * block_size * kv_heads * 2 * head_dim,
        dtype=torch.float32,
    ).reshape(blocks, kv_heads, block_size, 2 * head_dim)
    block_table = torch.tensor([2, 0, 1])

    gathered = Alpamayo2SuperForConditionalGeneration._gather_prefix_cache(
        [cache], block_table, seq_len=5
    )

    key_blocks, value_blocks = cache.chunk(2, dim=-1)
    expected_key = key_blocks.index_select(0, block_table)
    expected_value = value_blocks.index_select(0, block_table)
    expected_key = expected_key.permute(1, 0, 2, 3).flatten(1, 2)[:, :5]
    expected_value = expected_value.permute(1, 0, 2, 3).flatten(1, 2)[:, :5]
    assert gathered.get_seq_length() == 5
    torch.testing.assert_close(gathered.layers[0].keys, expected_key.unsqueeze(0))
    torch.testing.assert_close(gathered.layers[0].values, expected_value.unsqueeze(0))


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


def test_super_action_boundary_position_ignores_mrope_graph_padding() -> None:
    positions = torch.tensor(
        [
            [705, 17, 18, 19],
            [705, 17, 18, 19],
            [705, 17, 18, 19],
        ]
    )

    actual = Alpamayo2SuperForConditionalGeneration._action_boundary_position(
        positions,
        device=torch.device("cpu"),
    )

    assert torch.equal(actual, torch.tensor([[705], [705], [705]]))


def test_super_action_boundary_position_expands_scalar_position() -> None:
    positions = torch.tensor([705, 17, 18, 19])

    actual = Alpamayo2SuperForConditionalGeneration._action_boundary_position(
        positions,
        device=torch.device("cpu"),
    )

    assert torch.equal(actual, torch.tensor([[705], [705], [705]]))


def test_super_policy_does_not_export_vlm_hidden_prefix() -> None:
    model = Alpamayo2SuperForConditionalGeneration
    assert model.have_multimodal_outputs
    assert model.needs_runner_kv_cache
    assert not model.requires_full_prefix_cached_hidden_states
    assert not model.omni_pooler_payload_include_hidden


def _minimal_policy_model() -> Alpamayo2SuperForConditionalGeneration:
    model = object.__new__(Alpamayo2SuperForConditionalGeneration)
    nn.Module.__init__(model)
    model.alpamayo_config = SimpleNamespace(
        traj_ids={"future_start": 7, "future_end": 9},
    )
    model._force_future_end = False
    return model


def test_super_declares_future_start_as_speculation_terminal() -> None:
    model = _minimal_policy_model()

    assert model.speculation_terminal_token_id == 7


def test_super_action_runs_when_terminal_is_first_input(monkeypatch) -> None:
    model = _minimal_policy_model()
    calls = []
    monkeypatch.setattr(
        model,
        "_sample_actions",
        lambda **kwargs: calls.append(kwargs)
        or {
            "actions": torch.zeros(8, 64, 3),
            "action_profile_ms": torch.zeros(4),
        },
    )
    context = RunnerKVCacheContext(
        caches=[],
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        sequence_lengths=(101,),
        request_ids=("request-0",),
    )

    output = model.make_omni_output(
        torch.zeros(1, 8),
        input_ids=torch.tensor([7]),
        positions=torch.arange(3).reshape(3, 1),
        sampling_extra_args=[{"robot_obs": {}}],
        runner_kv_cache_context=context,
    )

    assert len(calls) == 1
    assert output.multimodal_outputs["actions"].shape == (1, 8, 64, 3)
    assert output.multimodal_outputs["action_profile_ms"].shape == (4,)
    assert model._force_future_end is True


def test_super_action_ignores_terminal_later_in_verification_block(monkeypatch) -> None:
    model = _minimal_policy_model()
    calls = []
    monkeypatch.setattr(
        model,
        "_sample_actions",
        lambda **kwargs: calls.append(kwargs) or {"actions": torch.zeros(1)},
    )

    output = model.make_omni_output(
        torch.zeros(3, 8),
        input_ids=torch.tensor([5, 7, 6]),
        positions=torch.arange(9).reshape(3, 3),
        sampling_extra_args=[{"robot_obs": {}}],
        runner_kv_cache_context=None,
    )

    assert calls == []
    assert output.multimodal_outputs == {}
    assert model._force_future_end is False
