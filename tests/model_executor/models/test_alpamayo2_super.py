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
    assert config.image_token_index == config.image_token_id == 151655
    assert config.traj_ids["future_start"] == 155681


def test_super_pipeline_is_single_stage_policy() -> None:
    assert ALPAMAYO2_SUPER_PIPELINE.model_type == "alpamayo2_super"
    assert len(ALPAMAYO2_SUPER_PIPELINE.stages) == 1
    assert ALPAMAYO2_SUPER_PIPELINE.stages[0].model_stage == "policy"
    assert ALPAMAYO2_SUPER_PIPELINE.stages[0].sampling_constraints == {"detokenize": True}


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

    gathered = Alpamayo2SuperForConditionalGeneration._gather_prefix_cache([cache], block_table, seq_len=5)

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

    gathered = Alpamayo2SuperForConditionalGeneration._gather_prefix_cache([cache], block_table, seq_len=5)

    key_blocks, value_blocks = cache.chunk(2, dim=-1)
    expected_key = key_blocks.index_select(0, block_table)
    expected_value = value_blocks.index_select(0, block_table)
    expected_key = expected_key.permute(1, 0, 2, 3).flatten(1, 2)[:, :5]
    expected_value = expected_value.permute(1, 0, 2, 3).flatten(1, 2)[:, :5]
    assert gathered.get_seq_length() == 5
    torch.testing.assert_close(gathered.layers[0].keys, expected_key.unsqueeze(0))
    torch.testing.assert_close(gathered.layers[0].values, expected_value.unsqueeze(0))


def test_super_repeats_dense_prefix_in_place() -> None:
    cache = Alpamayo2SuperForConditionalGeneration._gather_prefix_cache(
        [torch.arange(2 * 2 * 2 * 1 * 1, dtype=torch.float32).reshape(2, 2, 2, 1, 1)],
        torch.tensor([0, 1]),
        seq_len=3,
    )
    original_key = cache.layers[0].keys.clone()
    original_value = cache.layers[0].values.clone()

    repeated = Alpamayo2SuperForConditionalGeneration._repeat_cache(cache, 3)

    assert repeated is cache
    assert repeated.layers[0].keys.shape[0] == 3
    torch.testing.assert_close(repeated.layers[0].keys, original_key.repeat_interleave(3, dim=0))
    torch.testing.assert_close(repeated.layers[0].values, original_value.repeat_interleave(3, dim=0))


def test_super_gathers_batched_prefixes_one_layer_at_a_time() -> None:
    cache = torch.arange(
        2 * 4 * 2 * 1 * 2,
        dtype=torch.float32,
    ).reshape(2, 4, 2, 1, 2)
    block_tables = [torch.tensor([0, 1]), torch.tensor([2, 3])]
    seq_lens = [3, 4]

    combined = Alpamayo2SuperForConditionalGeneration._gather_prefix_cache_batch(
        [cache],
        block_tables,
        seq_lens,
    )

    first = Alpamayo2SuperForConditionalGeneration._gather_prefix_cache([cache], block_tables[0], seq_lens[0])
    second = Alpamayo2SuperForConditionalGeneration._gather_prefix_cache([cache], block_tables[1], seq_lens[1])
    expected_keys = torch.cat(
        [torch.nn.functional.pad(first.layers[0].keys, (0, 0, 0, 1)), second.layers[0].keys],
        dim=0,
    )
    expected_values = torch.cat(
        [torch.nn.functional.pad(first.layers[0].values, (0, 0, 0, 1)), second.layers[0].values],
        dim=0,
    )
    assert combined.get_seq_length() == 4
    torch.testing.assert_close(combined.layers[0].keys, expected_keys)
    torch.testing.assert_close(combined.layers[0].values, expected_values)


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
    model._force_future_end_indices = ()
    model._force_future_start_indices = ()
    model._pending_policy_groups = {}
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
    assert output.multimodal_outputs["actions"][0].shape == (8, 64, 3)
    assert output.multimodal_outputs["action_profile_ms"][0].shape == (4,)
    assert model._force_future_end_indices == (0,)


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
    assert model._force_future_end_indices == ()


def test_super_preserves_aux_hidden_states_for_dflash() -> None:
    model = _minimal_policy_model()
    text_hidden_states = torch.zeros(1, 8)
    aux_hidden_states = [torch.ones(1, 8), torch.full((1, 8), 2.0)]

    output = model.make_omni_output(
        (text_hidden_states, aux_hidden_states),
        positions=torch.arange(3).reshape(3, 1),
    )

    assert output.text_hidden_states is text_hidden_states
    assert output.aux_hidden_states is aux_hidden_states


def test_super_excludes_speculative_draft_cache_from_action_prefix(monkeypatch) -> None:
    model = _minimal_policy_model()
    model.expert = SimpleNamespace(
        config=SimpleNamespace(llm_config=SimpleNamespace(num_hidden_layers=2)),
    )
    received = []

    def capture_target_caches(caches, _block_tables, _seq_lens):
        received.extend(caches)
        raise RuntimeError("captured target caches")

    monkeypatch.setattr(model, "_gather_prefix_cache_batch", capture_target_caches)
    target_caches = [torch.zeros(1), torch.ones(1)]
    draft_cache = torch.full((1,), 2.0)

    with pytest.raises(RuntimeError, match="captured target caches"):
        model._sample_actions_batch(
            caches=[*target_caches, draft_cache],
            block_tables=[torch.zeros(1, dtype=torch.int32)],
            seq_lens=[1],
            positions=[torch.zeros(3, 1, dtype=torch.long)],
            observations=[{}],
            extra_args=[{}],
        )

    assert len(received) == 2
    assert received[0] is target_caches[0]
    assert received[1] is target_caches[1]


def test_super_action_routes_independent_batched_policy_children(monkeypatch) -> None:
    model = _minimal_policy_model()
    calls = []

    def sample_actions(**kwargs):
        calls.append(kwargs)
        marker = float(kwargs["seq_len"])
        return {
            "actions": torch.full((1, 64, 3), marker),
            "rotations": torch.full((1, 64, 3, 3), marker),
        }

    monkeypatch.setattr(model, "_sample_actions", sample_actions)
    context = RunnerKVCacheContext(
        caches=[],
        block_table=torch.tensor([[10], [20], [30]], dtype=torch.int32),
        sequence_lengths=(101, 202, 303),
        request_ids=("0-parent", "1-parent", "2-parent"),
    )
    extras = [
        {"robot_obs": {"branch": 0}, "_sampling_seed": 40},
        {"robot_obs": {"branch": 1}, "_sampling_seed": 41},
        {"robot_obs": {"branch": 2}, "_sampling_seed": 42},
    ]

    output = model.make_omni_output(
        torch.zeros(5, 8),
        input_ids=torch.tensor([7, 11, 5, 7, 12]),
        positions=torch.arange(15).reshape(3, 5),
        sampling_extra_args=extras,
        runner_kv_cache_context=context,
        request_token_spans=[(0, 2), (2, 3), (3, 5)],
    )

    assert [call["observation"]["branch"] for call in calls] == [0, 2]
    assert [call["extra_args"]["_sampling_seed"] for call in calls] == [40, 42]
    assert [call["block_table"].item() for call in calls] == [10, 30]
    assert torch.equal(calls[0]["positions"], torch.arange(15).reshape(3, 5)[:, 0:2])
    assert torch.equal(calls[1]["positions"], torch.arange(15).reshape(3, 5)[:, 3:5])
    assert output.multimodal_outputs["actions"][1] is None
    assert output.multimodal_outputs["actions"][0].shape == (1, 64, 3)
    assert output.multimodal_outputs["actions"][2][0, 0, 0].item() == 303
    assert model._force_future_end_indices == (0, 2)


def test_super_batched_action_expert_rendezvous_waits_for_every_vlm_child(
    monkeypatch,
) -> None:
    model = _minimal_policy_model()
    calls = []

    def sample_actions_batch(**kwargs):
        calls.append(kwargs)
        markers = torch.tensor(kwargs["seq_lens"], dtype=torch.float32)
        return {
            "actions": markers[:, None, None].expand(-1, 64, 3).clone(),
            "rotations": markers[:, None, None, None].expand(-1, 64, 3, 3).clone(),
        }

    monkeypatch.setattr(model, "_sample_actions_batch", sample_actions_batch)
    context = RunnerKVCacheContext(
        caches=[],
        block_table=torch.tensor([[10], [20], [30]], dtype=torch.int32),
        sequence_lengths=(101, 202, 303),
        request_ids=("0_parent", "1_parent", "2_parent"),
    )
    extras = [
        {
            "robot_obs": {"branch": index},
            "_sampling_seed": 40 + index,
            "_parallel_sample_count": 3,
            "_batch_action_expert": True,
        }
        for index in range(3)
    ]

    waiting = model.make_omni_output(
        torch.zeros(3, 8),
        input_ids=torch.tensor([7, 5, 7]),
        positions=torch.arange(9).reshape(3, 3),
        sampling_extra_args=extras,
        runner_kv_cache_context=context,
        request_token_spans=[(0, 1), (1, 2), (2, 3)],
    )

    assert calls == []
    assert waiting.multimodal_outputs == {}
    assert model._force_future_start_indices == (0, 2)

    completed = model.make_omni_output(
        torch.zeros(3, 8),
        input_ids=torch.tensor([7, 7, 7]),
        positions=torch.arange(9, 18).reshape(3, 3),
        sampling_extra_args=extras,
        runner_kv_cache_context=context,
        request_token_spans=[(0, 1), (1, 2), (2, 3)],
    )

    assert len(calls) == 1
    assert calls[0]["seq_lens"] == [101, 202, 303]
    assert [item["branch"] for item in calls[0]["observations"]] == [0, 1, 2]
    assert [item["_sampling_seed"] for item in calls[0]["extra_args"]] == [40, 41, 42]
    assert [item[0, 0, 0].item() for item in completed.multimodal_outputs["actions"]] == [
        101,
        202,
        303,
    ]
    assert model._force_future_end_indices == (0, 1, 2)
    assert model._pending_policy_groups == {}


def test_super_batched_action_expert_honors_microbatch_limit(monkeypatch) -> None:
    model = _minimal_policy_model()
    calls = []

    def sample_actions_batch(**kwargs):
        calls.append(kwargs)
        markers = torch.tensor(kwargs["seq_lens"], dtype=torch.float32)
        batch_size = len(markers)
        return {
            "actions": markers[:, None, None].expand(-1, 64, 3).clone(),
            "rotations": markers[:, None, None, None].expand(-1, 64, 3, 3).clone(),
            "action_expert_invocation_batch_size": torch.full((batch_size,), batch_size, dtype=torch.int32),
        }

    monkeypatch.setattr(model, "_sample_actions_batch", sample_actions_batch)
    context = RunnerKVCacheContext(
        caches=[],
        block_table=torch.arange(7, dtype=torch.int32)[:, None],
        sequence_lengths=(101, 102, 103, 104, 105, 106, 107),
        request_ids=tuple(f"{index}_parent" for index in range(7)),
    )
    extras = [
        {
            "robot_obs": {"branch": index},
            "_parallel_sample_count": 7,
            "_batch_action_expert": True,
            "_action_expert_max_batch_size": 3,
        }
        for index in range(7)
    ]

    output = model.make_omni_output(
        torch.zeros(7, 8),
        input_ids=torch.full((7,), 7),
        positions=torch.arange(21).reshape(3, 7),
        sampling_extra_args=extras,
        runner_kv_cache_context=context,
        request_token_spans=[(index, index + 1) for index in range(7)],
    )

    assert [call["seq_lens"] for call in calls] == [
        [101, 102, 103],
        [104, 105, 106],
        [107],
    ]
    assert [item[0, 0, 0].item() for item in output.multimodal_outputs["actions"]] == [
        101,
        102,
        103,
        104,
        105,
        106,
        107,
    ]
    assert [item.item() for item in output.multimodal_outputs["action_expert_invocation_batch_size"]] == [
        3,
        3,
        3,
        3,
        3,
        3,
        1,
    ]
