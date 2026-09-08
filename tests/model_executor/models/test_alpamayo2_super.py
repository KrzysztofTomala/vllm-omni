# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.alpamayo2_super import alpamayo2_super as alpamayo2_super_module
from vllm_omni.model_executor.models.alpamayo2_super.alpamayo2_super import (
    Alpamayo2SuperForConditionalGeneration,
    alpamayo_flash_attention_3_forward,
    expert_fa3_supports,
    resolve_expert_attention_backend,
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


def test_super_action_noise_can_be_recreated_after_inference() -> None:
    extra_args = [{"_sampling_seed": 42}, {"_sampling_seed": 43}]

    action = Alpamayo2SuperForConditionalGeneration._recreate_action_noise(
        extra_args,
        action_dims=(4, 3),
        device=torch.device("cpu"),
    )
    receipt = Alpamayo2SuperForConditionalGeneration._recreate_action_noise(
        extra_args,
        action_dims=(4, 3),
        device=torch.device("cpu"),
    )

    assert receipt.data_ptr() != action.data_ptr()
    torch.testing.assert_close(receipt, action, rtol=0, atol=0)
    assert not torch.equal(action[0], action[1])


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


def test_super_registers_fa3_expert_attention_with_transformers() -> None:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    assert "alpamayo_fa3" in ALL_ATTENTION_FUNCTIONS
    assert ALL_ATTENTION_FUNCTIONS["alpamayo_fa3"] is alpamayo_flash_attention_3_forward


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, True),
        ({"dtype": torch.float16}, True),
        ({"batch_rows": 3}, False),
        ({"batch_rows": 2}, False),
        ({"is_causal": True}, False),
        ({"device_type": "cpu"}, False),
        ({"dtype": torch.float32}, False),
        ({"dropout": 0.1}, False),
        ({"gqa_divisible": False}, False),
    ],
)
def test_super_expert_fa3_supports_truth_table(overrides, expected) -> None:
    kwargs = {
        "batch_rows": 1,
        "dtype": torch.bfloat16,
        "device_type": "cuda",
        "is_causal": False,
    }
    kwargs.update(overrides)

    assert expert_fa3_supports(**kwargs) is expected


def test_super_flash_attention_3_falls_back_to_sdpa_on_cpu() -> None:
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    torch.manual_seed(0)
    module = SimpleNamespace(is_causal=False, num_key_value_groups=1)
    query = torch.randn(1, 4, 2, 8)
    key = torch.randn(1, 4, 5, 8)
    value = torch.randn(1, 4, 5, 8)

    output, weights = alpamayo_flash_attention_3_forward(module, query, key, value, None, is_causal=False)
    expected, _ = sdpa_attention_forward(module, query, key, value, None, is_causal=False)

    assert weights is None
    assert output.shape == (1, 2, 4, 8)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_super_flash_attention_3_fallback_forwards_sdpa_arguments(monkeypatch) -> None:
    expected = torch.randn(1, 2, 4, 3)
    calls = []

    def fake_sdpa(*args, **kwargs):
        calls.append((args, kwargs))
        return expected, None

    monkeypatch.setattr(alpamayo2_super_module, "sdpa_attention_forward", fake_sdpa)
    query = torch.randn(1, 4, 2, 3)
    key = torch.randn(1, 2, 5, 3)

    output, weights = alpamayo_flash_attention_3_forward(SimpleNamespace(), query, key, key, None, is_causal=False)

    assert output is expected
    assert weights is None
    assert len(calls) == 1
    assert calls[0][1]["is_causal"] is False


def test_super_resolves_fa3_expert_backend_when_available() -> None:
    assert resolve_expert_attention_backend("alpamayo_fa3", fa3_available=True, allow_fallback=False) == "alpamayo_fa3"
    assert resolve_expert_attention_backend("alpamayo_fa3", fa3_available=True, allow_fallback=True) == "alpamayo_fa3"


def test_super_unavailable_fa3_expert_backend_fails_without_fallback() -> None:
    with pytest.raises(
        RuntimeError, match="alpamayo_fa3.*unsupported compute capability.*allow_expert_attention_fallback"
    ):
        resolve_expert_attention_backend(
            "alpamayo_fa3",
            fa3_available=False,
            allow_fallback=False,
            unavailable_reason="unsupported compute capability 8.0",
        )


def test_super_unavailable_fa3_expert_backend_falls_back_with_warning(monkeypatch) -> None:
    warning_logs = _capture_logs(monkeypatch, "warning")

    effective = resolve_expert_attention_backend(
        "alpamayo_fa3",
        fa3_available=False,
        allow_fallback=True,
        unavailable_reason="CUDA is unavailable",
    )

    assert effective == "sdpa"
    assert len(warning_logs) == 1
    assert "alpamayo_fa3" in warning_logs[0]
    assert "CUDA is unavailable" in warning_logs[0]
    assert "sdpa" in warning_logs[0]


def test_super_resolves_sdpa_expert_backend_regardless_of_fa3() -> None:
    assert resolve_expert_attention_backend("sdpa", fa3_available=False, allow_fallback=False) == "sdpa"
    assert resolve_expert_attention_backend("sdpa", fa3_available=True, allow_fallback=False) == "sdpa"


def test_super_rejects_unknown_expert_attention_backend() -> None:
    with pytest.raises(ValueError, match="flash_attention_2"):
        resolve_expert_attention_backend("flash_attention_2", fa3_available=True, allow_fallback=True)


def _minimal_policy_model() -> Alpamayo2SuperForConditionalGeneration:
    model = object.__new__(Alpamayo2SuperForConditionalGeneration)
    nn.Module.__init__(model)
    model.alpamayo_config = SimpleNamespace(
        traj_ids={
            "history_id0": 10,
            "future_id0": 20,
            "future_start": 7,
            "future_end": 9,
        },
        traj_vocab_size=4,
    )
    model._force_future_end_indices = ()
    model._force_future_start_indices = ()
    model._mask_text_eos_indices = ()
    model._text_eos_token_ids = (2, 3)
    model._pending_policy_groups = {}
    model._policy_group_hold_steps = {}
    model._pending_nav_twins = {}
    model._nav_twin_prefill_lengths = {}
    model._logits_request_count = None
    model._compiled_expert = None
    model.expert_attention_backend_requested = "sdpa"
    model.expert_attention_backend = "sdpa"
    model._decode_cudagraph_observation = -1
    model._uniform_decode_query_len = 1
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


def test_super_masks_text_eos_only_for_trajectory_rows(monkeypatch) -> None:
    model = _minimal_policy_model()
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_vl.Qwen3VLForConditionalGeneration.compute_logits",
        lambda _self, hidden_states: hidden_states.clone(),
    )
    model.make_omni_output(
        torch.zeros(3, 24),
        input_ids=torch.tensor([5, 5, 5]),
        positions=torch.arange(9).reshape(3, 3),
        sampling_extra_args=[{"robot_obs": {}}, {}, {"robot_obs": {}}],
        request_token_spans=[(0, 1), (1, 2), (2, 3)],
    )

    logits = model.compute_logits(torch.zeros(3, 24))

    assert torch.isneginf(logits[0, 2:4]).all()
    assert torch.equal(logits[1, 2:4], torch.zeros(2))
    assert torch.isneginf(logits[2, 2:4]).all()
    assert torch.equal(logits[:, model.future_start_id], torch.zeros(3))
    assert model._mask_text_eos_indices == ()


def test_super_forced_action_boundary_overrides_eos_mask(monkeypatch) -> None:
    model = _minimal_policy_model()
    model._text_eos_token_ids = (2, model.future_end_id)
    model._mask_text_eos_indices = (0,)
    model._force_future_end_indices = (0,)
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_vl.Qwen3VLForConditionalGeneration.compute_logits",
        lambda _self, hidden_states: hidden_states.clone(),
    )

    logits = model.compute_logits(torch.zeros(1, 24))

    assert torch.isneginf(logits[0]).sum() == logits.shape[-1] - 1
    assert logits[0, model.future_end_id] == 0


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


# The NIM-visible request id; vLLM renames the request internally and names
# children "<index>_<internal uuid>", so runner ids never contain this string.
_NIM_REQUEST_ID = "nim-trajectory-7"
_INTERNAL_PARENT = "9f2c1a3e"


def _child_id(index):
    return f"{index}_{_INTERNAL_PARENT}"


def _nav_child_extra(index, *, expected, weight=3.0, group=_NIM_REQUEST_ID, **overrides):
    extra = {
        "robot_obs": {"branch": index},
        "_sampling_seed": 40 + index,
        "_parallel_sample_count": expected,
        "_batch_action_expert": True,
        "_nav_guidance_weight": weight,
        "_nav_cfg_group": group,
    }
    extra.update(overrides)
    return extra


def _twin_extra(partner_index, *, group=_NIM_REQUEST_ID, **overrides):
    extra = {
        "robot_obs": {"twin_of": partner_index},
        "_nav_cfg_role": "unguided",
        # What the NIM passes: the child id built from its own request id.
        "_nav_cfg_partner": f"{partner_index}_{group}",
        "_nav_cfg_group": group,
        "_nav_cfg_partner_index": partner_index,
        "_force_first_token": 7,
    }
    extra.update(overrides)
    return extra


def _prime_twin_prefill(model, context, extras, *, offset=1):
    """Record each twin's in-engine prefill length as its forcing step would.

    The engine-side length includes multimodal placeholder expansion, so it is
    unrelated to the adapter's prompt token count; ``offset`` != 1 simulates a
    twin whose forced first token was not its first produced token.
    """
    for index, extra in enumerate(extras):
        if extra.get("_nav_cfg_role") == "unguided":
            key = (extra["_nav_cfg_group"], extra["_nav_cfg_partner_index"])
            model._nav_twin_prefill_lengths[key] = int(context.sequence_lengths[index]) - offset


def _capture_logs(monkeypatch, level):
    """Collect formatted messages emitted through the module logger at ``level``."""
    records = []
    monkeypatch.setattr(alpamayo2_super_module.logger, level, lambda msg, *args: records.append(msg % args))
    return records


def _marker_sample_actions_batch(calls):
    def sample_actions_batch(**kwargs):
        calls.append(kwargs)
        markers = torch.tensor(kwargs["seq_lens"], dtype=torch.float32)
        applied = "unguided_block_tables" in kwargs
        return {
            "actions": markers[:, None, None].expand(-1, 64, 3).clone(),
            "nav_cfg_applied": torch.tensor(applied),
            "nav_guidance_weight": torch.tensor(3.0 if applied else 1.0),
        }

    return sample_actions_batch


def test_super_forces_first_token_for_unguided_twin_during_prefill(monkeypatch) -> None:
    model = _minimal_policy_model()
    calls = []
    monkeypatch.setattr(model, "_sample_actions_batch", _marker_sample_actions_batch(calls))
    monkeypatch.setattr(model, "_sample_actions", lambda **kwargs: calls.append(kwargs) or {})
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_vl.Qwen3VLForConditionalGeneration.compute_logits",
        lambda _self, hidden_states: hidden_states.clone(),
    )
    context = RunnerKVCacheContext(
        caches=[],
        block_table=torch.tensor([[10], [20]], dtype=torch.int32),
        sequence_lengths=(50, 120),
        request_ids=(_child_id(0), "twin-0"),
    )

    # The twin's prompt (a chunk of it) is prefilled next to an ordinary
    # decode step of an unrelated request; nothing triggers.
    output = model.make_omni_output(
        torch.zeros(4, 8),
        input_ids=torch.tensor([5, 1, 2, 3]),
        positions=torch.arange(12).reshape(3, 4),
        sampling_extra_args=[{"robot_obs": {}}, _twin_extra(0)],
        runner_kv_cache_context=context,
        request_token_spans=[(0, 1), (1, 4)],
    )

    assert calls == []
    assert output.multimodal_outputs == {}
    assert model._force_future_start_indices == (1,)
    assert model._force_future_end_indices == ()
    assert model._pending_nav_twins == {}
    assert model._nav_twin_prefill_lengths == {(_NIM_REQUEST_ID, 0): 120}

    logits = model.compute_logits(torch.zeros(2, 24))
    assert torch.isneginf(logits[1]).sum() == logits.shape[-1] - 1
    assert logits[1, model.future_start_id] == 0
    assert logits[0, 5] == 0

    with pytest.raises(ValueError, match="_force_first_token"):
        model.make_omni_output(
            torch.zeros(1, 8),
            input_ids=torch.tensor([5]),
            positions=torch.arange(3).reshape(3, 1),
            sampling_extra_args=[_twin_extra(0, _force_first_token=9)],
            runner_kv_cache_context=None,
            request_token_spans=[(0, 1)],
        )


def test_super_compute_logits_skips_forcing_when_rows_are_not_requests(monkeypatch) -> None:
    model = _minimal_policy_model()
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_vl.Qwen3VLForConditionalGeneration.compute_logits",
        lambda _self, hidden_states: hidden_states.clone(),
    )
    model._logits_request_count = 2
    model._force_future_start_indices = (1,)
    model._force_future_end_indices = (0,)

    # Speculative verification: five logits rows for two requests.
    logits = model.compute_logits(torch.zeros(5, 24))
    assert not torch.isneginf(logits[..., 5]).any()
    assert model._logits_request_count is None

    model._logits_request_count = 2
    model._force_future_start_indices = (1,)
    logits = model.compute_logits(torch.zeros(2, 24))
    assert logits[1, model.future_start_id] == 0
    assert torch.isneginf(logits[1, 5])


def test_super_nav_cfg_rendezvous_waits_for_twins_then_runs_expert_with_pairs(monkeypatch) -> None:
    model = _minimal_policy_model()
    calls = []
    info_logs = _capture_logs(monkeypatch, "info")
    monkeypatch.setattr(model, "_sample_actions_batch", _marker_sample_actions_batch(calls))
    monkeypatch.setattr(model, "_sample_actions", lambda **kwargs: calls.append(kwargs) or {})
    request_ids = (_child_id(0), _child_id(1), "twin-0", "twin-1")
    block_table = torch.tensor([[10], [20], [30], [40]], dtype=torch.int32)

    def context(*sequence_lengths):
        return RunnerKVCacheContext(
            caches=[], block_table=block_table, sequence_lengths=sequence_lengths, request_ids=request_ids
        )

    # Four expert rows (two guided samples plus their twins) fit one call.
    # The adapter built ~300-token twin prompts; after image placeholder
    # expansion the engine prefill is ~4600 tokens (24 images). Only the
    # in-engine length may be used for the boundary check.
    extras = [
        _nav_child_extra(0, expected=2, _action_expert_max_batch_size=4),
        _nav_child_extra(1, expected=2, _action_expert_max_batch_size=4),
        _twin_extra(0),
        _twin_extra(1),
    ]
    positions = torch.arange(12).reshape(3, 4)
    spans = [(index, index + 1) for index in range(4)]

    # Step 1: both guided children reach future_start; the twins finish
    # prefilling (their spans do not start with future_start) and get their
    # first token forced.
    waiting = model.make_omni_output(
        torch.zeros(4, 8),
        input_ids=torch.tensor([7, 7, 3, 4]),
        positions=positions,
        sampling_extra_args=extras,
        runner_kv_cache_context=context(4601, 4602, 4593, 4610),
        request_token_spans=spans,
    )
    assert calls == []
    assert waiting.multimodal_outputs == {}
    assert model._force_future_start_indices == (0, 1, 2, 3)
    assert model._force_future_end_indices == ()
    assert model._policy_group_hold_steps == {_INTERNAL_PARENT: 1}
    assert model._nav_twin_prefill_lengths == {(_NIM_REQUEST_ID, 0): 4593, (_NIM_REQUEST_ID, 1): 4610}

    # Step 2: only twin-0 has triggered; the group keeps holding everyone.
    waiting = model.make_omni_output(
        torch.zeros(4, 8),
        input_ids=torch.tensor([7, 7, 7, 4]),
        positions=positions,
        sampling_extra_args=extras,
        runner_kv_cache_context=context(4602, 4603, 4594, 4610),
        request_token_spans=spans,
    )
    assert calls == []
    assert model._force_future_start_indices == (0, 1, 2, 3)
    assert model._force_future_end_indices == ()
    assert set(model._pending_nav_twins) == {(_NIM_REQUEST_ID, 0)}
    assert model._pending_nav_twins[(_NIM_REQUEST_ID, 0)]["seq_len"] == 4594
    assert model._nav_twin_prefill_lengths == {(_NIM_REQUEST_ID, 1): 4610}
    assert model._policy_group_hold_steps == {_INTERNAL_PARENT: 2}

    # Step 3: twin-1 triggers; one expert call with 2K prefix rows.
    completed = model.make_omni_output(
        torch.zeros(4, 8),
        input_ids=torch.tensor([7, 7, 7, 7]),
        positions=positions,
        sampling_extra_args=extras,
        runner_kv_cache_context=context(4603, 4604, 4595, 4611),
        request_token_spans=spans,
    )
    assert len(calls) == 1
    call = calls[0]
    assert call["seq_lens"] == [4601, 4602]
    assert call["unguided_seq_lens"] == [4594, 4611]
    assert [table.item() for table in call["block_tables"]] == [10, 20]
    assert [table.item() for table in call["unguided_block_tables"]] == [30, 40]
    assert torch.equal(call["unguided_positions"][0], positions[:, 2:3])
    assert torch.equal(call["unguided_positions"][1], positions[:, 3:4])
    assert [item["branch"] for item in call["observations"]] == [0, 1]
    actions = completed.multimodal_outputs["actions"]
    assert [item[0, 0, 0].item() for item in actions[:2]] == [4601, 4602]
    assert actions[2] is None and actions[3] is None
    assert completed.multimodal_outputs["nav_cfg_applied"][0].item() is True
    assert completed.multimodal_outputs["nav_guidance_weight"][1].item() == 3.0
    assert model._force_future_end_indices == (0, 1, 2, 3)
    assert model._force_future_start_indices == ()
    assert model._pending_policy_groups == {}
    assert model._pending_nav_twins == {}
    assert model._nav_twin_prefill_lengths == {}
    assert model._policy_group_hold_steps == {}
    assert info_logs == [
        f"Navigation CFG applied for group {_INTERNAL_PARENT} after 2 held decode steps (K=2, weight=3.00)"
    ]


def test_super_nav_cfg_single_sample_still_waits_for_twin(monkeypatch) -> None:
    model = _minimal_policy_model()
    calls = []
    monkeypatch.setattr(model, "_sample_actions_batch", _marker_sample_actions_batch(calls))
    monkeypatch.setattr(model, "_sample_actions", lambda **kwargs: calls.append(kwargs) or {})

    def context(*sequence_lengths):
        return RunnerKVCacheContext(
            caches=[],
            block_table=torch.tensor([[10], [30]], dtype=torch.int32),
            sequence_lengths=sequence_lengths,
            # With n=1 the runner still sees an internal id, never the NIM's.
            request_ids=("c0ffee42", "twin"),
        )

    extras = [_nav_child_extra(0, expected=1), _twin_extra(0)]
    assert extras[1]["_nav_cfg_partner"] not in context(0, 0).request_ids

    # The child triggers while the twin's last prefill chunk completes; the
    # twin's in-engine length (200) is recorded on this forcing step.
    model.make_omni_output(
        torch.zeros(2, 8),
        input_ids=torch.tensor([7, 3]),
        positions=torch.arange(6).reshape(3, 2),
        sampling_extra_args=extras,
        runner_kv_cache_context=context(101, 200),
        request_token_spans=[(0, 1), (1, 2)],
    )
    assert calls == []
    assert model._force_future_start_indices == (0, 1)
    assert model._nav_twin_prefill_lengths == {(_NIM_REQUEST_ID, 0): 200}

    completed = model.make_omni_output(
        torch.zeros(2, 8),
        input_ids=torch.tensor([7, 7]),
        positions=torch.arange(6).reshape(3, 2),
        sampling_extra_args=extras,
        runner_kv_cache_context=context(102, 201),
        request_token_spans=[(0, 1), (1, 2)],
    )
    assert len(calls) == 1
    assert calls[0]["seq_lens"] == [101]
    assert calls[0]["unguided_seq_lens"] == [201]
    assert completed.multimodal_outputs["actions"][1] is None
    assert model._force_future_end_indices == (0, 1)


def test_super_nav_cfg_hold_timeout_runs_guided_only_and_ends_late_twin(monkeypatch) -> None:
    model = _minimal_policy_model()
    calls = []
    info_logs = _capture_logs(monkeypatch, "info")
    warning_logs = _capture_logs(monkeypatch, "warning")
    monkeypatch.setattr(model, "_sample_actions_batch", _marker_sample_actions_batch(calls))
    context = RunnerKVCacheContext(
        caches=[],
        block_table=torch.tensor([[10]], dtype=torch.int32),
        sequence_lengths=(101,),
        request_ids=(_child_id(0),),
    )
    extras = [_nav_child_extra(0, expected=1, _nav_cfg_max_hold_steps=3)]

    for _ in range(2):
        output = model.make_omni_output(
            torch.zeros(1, 8),
            input_ids=torch.tensor([7]),
            positions=torch.arange(3).reshape(3, 1),
            sampling_extra_args=extras,
            runner_kv_cache_context=context,
        )
        assert calls == []
        assert output.multimodal_outputs == {}
        assert model._force_future_start_indices == (0,)

    output = model.make_omni_output(
        torch.zeros(1, 8),
        input_ids=torch.tensor([7]),
        positions=torch.arange(3).reshape(3, 1),
        sampling_extra_args=extras,
        runner_kv_cache_context=context,
    )
    assert len(calls) == 1
    assert "unguided_block_tables" not in calls[0]
    assert output.multimodal_outputs["nav_cfg_applied"][0].item() is False
    assert output.multimodal_outputs["nav_guidance_weight"][0].item() == 1.0
    assert model._force_future_end_indices == (0,)
    assert model._pending_policy_groups == {}
    assert model._policy_group_hold_steps == {}
    assert info_logs == []
    # With n=1 the group is the child's own runner id.
    assert warning_logs == [
        f"Navigation CFG twin missing for {_child_id(0)} after 3 held decode steps; "
        "running the action expert guided-only"
    ]

    # A twin arriving after its partner finished never runs the expert,
    # whether its prefill length was recorded (orphan) or not (invalid).
    late_context = RunnerKVCacheContext(
        caches=[],
        block_table=torch.tensor([[30]], dtype=torch.int32),
        sequence_lengths=(201,),
        request_ids=("twin-0",),
    )
    for prime in (True, False):
        if prime:
            _prime_twin_prefill(model, late_context, [_twin_extra(0)])
        output = model.make_omni_output(
            torch.zeros(1, 8),
            input_ids=torch.tensor([7]),
            positions=torch.arange(3).reshape(3, 1),
            sampling_extra_args=[_twin_extra(0)],
            runner_kv_cache_context=late_context,
        )
        assert len(calls) == 1
        assert output.multimodal_outputs == {}
        assert model._force_future_end_indices == (0,)
        assert model._force_future_start_indices == ()
        assert model._pending_nav_twins == {}
        assert model._nav_twin_prefill_lengths == {}


@pytest.mark.parametrize("recorded", [True, False])
def test_super_nav_cfg_twin_with_unexpected_prefix_length_disables_guidance(monkeypatch, recorded) -> None:
    model = _minimal_policy_model()
    calls = []
    monkeypatch.setattr(model, "_sample_actions_batch", _marker_sample_actions_batch(calls))
    context = RunnerKVCacheContext(
        caches=[],
        block_table=torch.tensor([[10], [30]], dtype=torch.int32),
        sequence_lengths=(101, 205),
        request_ids=(_child_id(0), "twin-0"),
    )
    extras = [_nav_child_extra(0, expected=1), _twin_extra(0)]
    if recorded:
        # The forcing step was skipped once, so a stray token precedes future_start.
        _prime_twin_prefill(model, context, extras, offset=2)

    output = model.make_omni_output(
        torch.zeros(2, 8),
        input_ids=torch.tensor([7, 7]),
        positions=torch.arange(6).reshape(3, 2),
        sampling_extra_args=extras,
        runner_kv_cache_context=context,
        request_token_spans=[(0, 1), (1, 2)],
    )

    assert len(calls) == 1
    assert "unguided_block_tables" not in calls[0]
    assert output.multimodal_outputs["nav_cfg_applied"][0].item() is False
    assert output.multimodal_outputs["actions"][1] is None
    assert model._force_future_end_indices == (0, 1)
    assert model._pending_nav_twins == {}
    assert model._nav_twin_prefill_lengths == {}
    assert model._pending_policy_groups == {}


def test_super_nav_cfg_microbatches_pairs_within_row_budget(monkeypatch) -> None:
    model = _minimal_policy_model()
    calls = []
    monkeypatch.setattr(model, "_sample_actions_batch", _marker_sample_actions_batch(calls))
    children = [_child_id(index) for index in range(3)]
    twins = [f"twin-{index}" for index in range(3)]
    context = RunnerKVCacheContext(
        caches=[],
        block_table=torch.arange(6, dtype=torch.int32)[:, None],
        sequence_lengths=(101, 102, 103, 201, 202, 203),
        request_ids=(*children, *twins),
    )
    extras = [_nav_child_extra(index, expected=3, _action_expert_max_batch_size=4) for index in range(3)] + [
        _twin_extra(index) for index in range(3)
    ]
    _prime_twin_prefill(model, context, extras)

    output = model.make_omni_output(
        torch.zeros(6, 8),
        input_ids=torch.full((6,), 7),
        positions=torch.arange(18).reshape(3, 6),
        sampling_extra_args=extras,
        runner_kv_cache_context=context,
        request_token_spans=[(index, index + 1) for index in range(6)],
    )

    # Four rows per call: two guided samples plus their two twins, then one pair.
    assert [call["seq_lens"] for call in calls] == [[101, 102], [103]]
    assert [call["unguided_seq_lens"] for call in calls] == [[201, 202], [203]]
    assert [item[0, 0, 0].item() for item in output.multimodal_outputs["actions"][:3]] == [101, 102, 103]
    assert output.multimodal_outputs["actions"][3:] == [None, None, None]
    assert model._force_future_end_indices == (0, 1, 2, 3, 4, 5)

    ordered = [(name, {}) for name in children]
    chunk = Alpamayo2SuperForConditionalGeneration._microbatch_children
    assert [len(part) for part in chunk(ordered, 3, rows_per_sample=2)] == [1, 1, 1]
    assert [len(part) for part in chunk(ordered, 1, rows_per_sample=2)] == [1, 1, 1]
    assert [len(part) for part in chunk(ordered, 3, rows_per_sample=1)] == [3]


def test_super_nav_cfg_pairs_twin_by_group_and_index_not_by_partner_id(monkeypatch) -> None:
    model = _minimal_policy_model()
    calls = []
    info_logs = _capture_logs(monkeypatch, "info")
    monkeypatch.setattr(model, "_sample_actions_batch", _marker_sample_actions_batch(calls))
    # Two concurrent NIM requests (K=2 and K=1); child index 0 exists in both
    # groups, so pairing must use the caller request id as well.
    context = RunnerKVCacheContext(
        caches=[],
        block_table=torch.tensor([[10], [20], [30], [40], [50], [60]], dtype=torch.int32),
        sequence_lengths=(101, 102, 111, 201, 202, 211),
        request_ids=("0_aaaa", "1_aaaa", "bbbb", "tw-a0", "tw-a1", "tw-b0"),
    )
    extras = [
        _nav_child_extra(0, expected=2, group="nim-A", _action_expert_max_batch_size=4),
        _nav_child_extra(1, expected=2, group="nim-A", _action_expert_max_batch_size=4),
        _nav_child_extra(0, expected=1, group="nim-B"),
        _twin_extra(0, group="nim-A"),
        _twin_extra(1, group="nim-A"),
        _twin_extra(0, group="nim-B"),
    ]
    assert not {extra["_nav_cfg_partner"] for extra in extras[3:]} & set(context.request_ids)
    _prime_twin_prefill(model, context, extras)

    output = model.make_omni_output(
        torch.zeros(6, 8),
        input_ids=torch.full((6,), 7),
        positions=torch.arange(18).reshape(3, 6),
        sampling_extra_args=extras,
        runner_kv_cache_context=context,
        request_token_spans=[(index, index + 1) for index in range(6)],
    )

    by_group = {tuple(call["seq_lens"]): call for call in calls}
    assert set(by_group) == {(101, 102), (111,)}
    assert by_group[(101, 102)]["unguided_seq_lens"] == [201, 202]
    assert by_group[(111,)]["unguided_seq_lens"] == [211]
    assert [item[0, 0, 0].item() for item in output.multimodal_outputs["actions"][:3]] == [101, 102, 111]
    assert output.multimodal_outputs["actions"][3:] == [None, None, None]
    assert model._force_future_end_indices == (0, 1, 2, 3, 4, 5)
    assert model._pending_nav_twins == {}
    assert model._pending_policy_groups == {}
    # Twins were already present when the children completed: no held steps.
    assert sorted(info_logs) == [
        "Navigation CFG applied for group aaaa after 0 held decode steps (K=2, weight=3.00)",
        "Navigation CFG applied for group bbbb after 0 held decode steps (K=1, weight=3.00)",
    ]

    with pytest.raises(RuntimeError, match="_nav_cfg_group"):
        model.make_omni_output(
            torch.zeros(1, 8),
            input_ids=torch.tensor([7]),
            positions=torch.arange(3).reshape(3, 1),
            sampling_extra_args=[{"robot_obs": {}, "_nav_cfg_role": "unguided", "_nav_cfg_partner": "0_x"}],
            runner_kv_cache_context=RunnerKVCacheContext(
                caches=[],
                block_table=torch.tensor([[1]], dtype=torch.int32),
                sequence_lengths=(5,),
                request_ids=("tw",),
            ),
        )


class _FakeProjection:
    def __init__(self, dtype=torch.bfloat16):
        self.weight = torch.zeros(1, dtype=dtype)

    def __call__(self, hidden_states, *_args):
        return hidden_states


class _PrefixSumExpert:
    """Return a per-row constant derived from that row's prefix keys."""

    def __call__(self, *, inputs_embeds, past_key_values, **_kwargs):
        rows = inputs_embeds.shape[0]
        keys = past_key_values.layers[0].keys
        assert keys.shape[0] == rows
        row_sums = keys.float().flatten(1).sum(dim=1)
        hidden = row_sums[:, None, None].expand(rows, inputs_embeds.shape[1], 1).clone()
        return SimpleNamespace(last_hidden_state=hidden.to(inputs_embeds.dtype))


def _cfg_test_model():
    model = _minimal_policy_model()
    model.expert = SimpleNamespace(
        config=SimpleNamespace(llm_config=SimpleNamespace(num_hidden_layers=1), expert_non_causal_attention=True),
        action_space=SimpleNamespace(
            get_action_space_dims=lambda: (2, 1),
            action_to_traj=lambda action, xyz, rot: (action, action),
        ),
        action_in_proj=lambda action, timestep: torch.zeros(action.shape[0], 2, 1, dtype=torch.float32),
        action_out_proj=_FakeProjection(),
        expert=_PrefixSumExpert(),
    )
    return model


def _cfg_test_cache():
    # Rank-5 paged layout [2, blocks, block_size=1, kv_heads=1, head_dim=1]:
    # block b holds key value b for both K and V.
    values = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1, 1, 1)
    return torch.cat([values, values], dim=0)


def _run_cfg(model, *, weight, with_twins, steps=4):
    observations = [{"ego_history_xyz": torch.zeros(4, 3), "ego_history_rot": torch.zeros(4, 3, 3)} for _ in range(2)]
    extras = [
        {
            "_sampling_seed": 40 + index,
            "diffusion_steps": steps,
            "_nav_guidance_weight": weight,
            "_return_action_noise": True,
        }
        for index in range(2)
    ]
    kwargs = dict(
        caches=[_cfg_test_cache()],
        # Guided prefixes sum to 1 and 5; unguided prefixes sum to 3 and 4.
        block_tables=[torch.tensor([1]), torch.tensor([2, 3])],
        seq_lens=[1, 2],
        positions=[torch.full((3, 1), 10), torch.full((3, 1), 20)],
        observations=observations,
        extra_args=extras,
    )
    if with_twins:
        kwargs.update(
            unguided_block_tables=[torch.tensor([3]), torch.tensor([4])],
            unguided_seq_lens=[1, 1],
            unguided_positions=[torch.full((3, 1), 8), torch.full((3, 1), 9)],
        )
    return model._sample_actions_batch(**kwargs)


def test_super_nav_cfg_combines_guided_and_unguided_velocities() -> None:
    model = _cfg_test_model()

    result = _run_cfg(model, weight=3.0, with_twins=True)

    noise = result["action_noise"]
    guided = torch.tensor([1.0, 5.0])
    unguided = torch.tensor([3.0, 4.0])
    # Integrating a constant velocity over the unit interval adds it once:
    # x1 = x0 + (1 - w) * v_unguided + w * v_guided, as in FlowMatching._guided_v.
    expected = noise + ((1 - 3.0) * unguided + 3.0 * guided)[:, None, None]
    torch.testing.assert_close(result["normalized_controls"], expected)
    assert result["nav_cfg_applied"].item() is True
    assert result["nav_guidance_weight"].item() == 3.0
    assert result["action_expert_invocation_batch_size"].tolist() == [2, 2]
    assert result["actions"].shape == (2, 2, 1)


def test_super_nav_cfg_with_unit_weight_matches_guided_only_run() -> None:
    model = _cfg_test_model()

    guided_only = _run_cfg(model, weight=None, with_twins=False)
    unit_weight = _run_cfg(model, weight=1.0, with_twins=True)

    torch.testing.assert_close(unit_weight["normalized_controls"], guided_only["normalized_controls"])
    torch.testing.assert_close(
        guided_only["normalized_controls"],
        guided_only["action_noise"] + torch.tensor([1.0, 5.0])[:, None, None],
    )
    assert guided_only["nav_cfg_applied"].item() is False
    assert guided_only["nav_guidance_weight"].item() == 1.0
    assert unit_weight["nav_cfg_applied"].item() is True


def test_super_nav_cfg_disables_manual_graph_and_static_cache(monkeypatch) -> None:
    model = _cfg_test_model()
    monkeypatch.setattr(
        model,
        "_run_manual_action_graph",
        lambda **_kwargs: pytest.fail("CFG must not take the captured graph path"),
    )
    monkeypatch.setattr(
        model,
        "_make_static_action_cache",
        lambda *_args, **_kwargs: pytest.fail("CFG must not build a StaticCache"),
    )
    observations = [{"ego_history_xyz": torch.zeros(4, 3), "ego_history_rot": torch.zeros(4, 3, 3)}]
    extras = [
        {
            "_sampling_seed": 1,
            "diffusion_steps": 2,
            "_nav_guidance_weight": 2.0,
            "_manual_action_cudagraph": True,
            "_static_expert_cache": True,
            "_static_expert_cache_max_len": 16,
        }
    ]

    result = model._sample_actions_batch(
        caches=[_cfg_test_cache()],
        block_tables=[torch.tensor([1])],
        seq_lens=[1],
        positions=[torch.full((3, 1), 10)],
        observations=observations,
        extra_args=extras,
        unguided_block_tables=[torch.tensor([3])],
        unguided_seq_lens=[1],
        unguided_positions=[torch.full((3, 1), 8)],
    )

    assert result["nav_cfg_applied"].item() is True
    with pytest.raises(ValueError, match="one unguided prefix per guided sample"):
        model._sample_actions_batch(
            caches=[_cfg_test_cache()],
            block_tables=[torch.tensor([1])],
            seq_lens=[1],
            positions=[torch.full((3, 1), 10)],
            observations=observations,
            extra_args=extras,
            unguided_block_tables=[torch.tensor([3]), torch.tensor([4])],
            unguided_seq_lens=[1, 1],
            unguided_positions=[torch.full((3, 1), 8)],
        )


def test_super_action_output_reports_attention_backend_and_decode_graph_state() -> None:
    model = _cfg_test_model()

    result = _run_cfg(model, weight=None, with_twins=False)

    assert result["action_expert_attention_fa3"].dtype == torch.int32
    assert result["action_expert_attention_fa3"].tolist() == [0, 0]
    assert result["action_expert_attention_backend_configured"].dtype == torch.int32
    assert result["action_expert_attention_backend_configured"].ndim == 0
    assert result["action_expert_attention_backend_configured"].item() == 0
    assert result["decode_full_cudagraph_observed"].dtype == torch.int32
    assert result["decode_full_cudagraph_observed"].ndim == 0
    assert result["decode_full_cudagraph_observed"].item() == -1

    model.expert_attention_backend = "alpamayo_fa3"
    model._decode_cudagraph_observation = 1
    result = _run_cfg(model, weight=None, with_twins=False)

    # Two prefix rows on CPU never satisfy the kernel predicate even though
    # FA3 is the configured backend.
    assert result["action_expert_attention_fa3"].tolist() == [0, 0]
    assert result["action_expert_attention_backend_configured"].item() == 1
    assert result["decode_full_cudagraph_observed"].item() == 1


def test_super_action_output_fa3_flag_uses_expert_row_count(monkeypatch) -> None:
    model = _cfg_test_model()
    model.expert_attention_backend = "alpamayo_fa3"
    predicate_calls = []

    def fake_supports(**kwargs):
        predicate_calls.append(kwargs)
        return kwargs["batch_rows"] == 1

    monkeypatch.setattr(alpamayo2_super_module, "expert_fa3_supports", fake_supports)

    guided_pairs = _run_cfg(model, weight=3.0, with_twins=True)
    single = model._sample_actions(
        caches=[_cfg_test_cache()],
        block_table=torch.tensor([1]),
        seq_len=1,
        positions=torch.full((3, 1), 10),
        observation={"ego_history_xyz": torch.zeros(4, 3), "ego_history_rot": torch.zeros(4, 3, 3)},
        extra_args={"_sampling_seed": 1, "diffusion_steps": 2},
    )

    # Navigation CFG doubles the expert rows (two guided + two unguided), so
    # the per-sample flag reports the eager two-row-per-sample path as 0.
    assert predicate_calls[0]["batch_rows"] == 4
    assert predicate_calls[0]["is_causal"] is False
    assert predicate_calls[0]["dtype"] == torch.bfloat16
    assert guided_pairs["action_expert_attention_fa3"].tolist() == [0, 0]
    assert predicate_calls[1]["batch_rows"] == 1
    assert single["action_expert_attention_fa3"].tolist() == [1]
    assert single["action_expert_attention_backend_configured"].item() == 1


def test_super_split_and_concat_handle_attention_report_keys() -> None:
    first = {
        "actions": torch.zeros(2, 2, 1),
        "action_expert_attention_fa3": torch.tensor([1, 1], dtype=torch.int32),
        "action_expert_attention_backend_configured": torch.tensor(1, dtype=torch.int32),
        "decode_full_cudagraph_observed": torch.tensor(1, dtype=torch.int32),
    }
    second = {
        "actions": torch.ones(1, 2, 1),
        "action_expert_attention_fa3": torch.tensor([0], dtype=torch.int32),
        "action_expert_attention_backend_configured": torch.tensor(1, dtype=torch.int32),
        "decode_full_cudagraph_observed": torch.tensor(1, dtype=torch.int32),
    }

    combined = Alpamayo2SuperForConditionalGeneration._concat_batched_policy_outputs([first, second])

    assert combined["action_expert_attention_fa3"].tolist() == [1, 1, 0]
    assert combined["action_expert_attention_backend_configured"].ndim == 0
    assert combined["decode_full_cudagraph_observed"].ndim == 0

    split = Alpamayo2SuperForConditionalGeneration._split_batched_policy_output(combined, 2)

    assert split["action_expert_attention_fa3"].tolist() == [0]
    assert split["actions"].shape == (1, 2, 1)
    assert split["action_expert_attention_backend_configured"].item() == 1
    assert split["decode_full_cudagraph_observed"].item() == 1


def _forward_with_context(model, monkeypatch, context_factory):
    sentinel = object()
    monkeypatch.setattr(
        alpamayo2_super_module.Qwen3VLForConditionalGeneration,
        "forward",
        lambda self, **kwargs: sentinel,
    )
    monkeypatch.setattr(alpamayo2_super_module, "get_forward_context", context_factory)
    output = model.forward(
        input_ids=torch.tensor([[7]]),
        positions=torch.zeros(3, 1, dtype=torch.long),
        sampling_extra_args=[{}],
        runner_kv_cache_context=None,
    )
    assert output is sentinel


def _decode_context(mode_name, *, uniform=True, max_query_len=None, dbo=False):
    """Forward context as vLLM 0.28 sets it: ``BatchDescriptor.uniform`` plus per-layer attention metadata."""
    attn_metadata = {} if max_query_len is None else {"layers.0.attn": SimpleNamespace(max_query_len=max_query_len)}
    return SimpleNamespace(
        cudagraph_runtime_mode=SimpleNamespace(name=mode_name),
        batch_descriptor=SimpleNamespace(uniform=uniform),
        attn_metadata=[attn_metadata, dict(attn_metadata)] if dbo else attn_metadata,
    )


def test_super_forward_observes_full_decode_cudagraph_capture(monkeypatch) -> None:
    model = _minimal_policy_model()

    _forward_with_context(model, monkeypatch, lambda: _decode_context("FULL"))

    assert model._decode_cudagraph_observation == 1

    # A later piecewise decode step (dispatched with a relaxed, non-uniform
    # descriptor) cannot revoke the capture evidence.
    _forward_with_context(model, monkeypatch, lambda: _decode_context("PIECEWISE", uniform=False, max_query_len=1))

    assert model._decode_cudagraph_observation == 1


def test_super_forward_observes_legacy_uniform_decode_descriptor(monkeypatch) -> None:
    model = _minimal_policy_model()

    _forward_with_context(
        model,
        monkeypatch,
        lambda: SimpleNamespace(
            cudagraph_runtime_mode=SimpleNamespace(name="FULL"),
            batch_descriptor=SimpleNamespace(uniform_decode=True),
        ),
    )

    assert model._decode_cudagraph_observation == 1


@pytest.mark.parametrize("mode_name", ["PIECEWISE", "NONE"])
@pytest.mark.parametrize("dbo", [False, True])
def test_super_forward_observes_decode_without_full_graph(monkeypatch, mode_name, dbo) -> None:
    model = _minimal_policy_model()

    # vLLM relaxes every non-FULL dispatch to uniform=False; the decode shape
    # is visible only through the attention metadata's max_query_len.
    _forward_with_context(
        model, monkeypatch, lambda: _decode_context(mode_name, uniform=False, max_query_len=1, dbo=dbo)
    )

    assert model._decode_cudagraph_observation == 0

    _forward_with_context(model, monkeypatch, lambda: _decode_context("FULL"))

    assert model._decode_cudagraph_observation == 1


def test_super_forward_decode_shape_follows_speculative_query_length(monkeypatch) -> None:
    model = _minimal_policy_model()
    model._uniform_decode_query_len = 3

    _forward_with_context(model, monkeypatch, lambda: _decode_context("PIECEWISE", uniform=False, max_query_len=1))
    assert model._decode_cudagraph_observation == -1

    _forward_with_context(model, monkeypatch, lambda: _decode_context("PIECEWISE", uniform=False, max_query_len=3))
    assert model._decode_cudagraph_observation == 0


def test_super_forward_ignores_prefill_and_missing_forward_context(monkeypatch) -> None:
    model = _minimal_policy_model()

    def missing_context():
        raise AssertionError("Forward context is not set")

    _forward_with_context(model, monkeypatch, missing_context)
    assert model._decode_cudagraph_observation == -1

    # Non-uniform FULL batches and prefill-shaped piecewise steps say nothing
    # about decode graphs.
    _forward_with_context(model, monkeypatch, lambda: _decode_context("FULL", uniform=False))
    assert model._decode_cudagraph_observation == -1

    _forward_with_context(model, monkeypatch, lambda: _decode_context("PIECEWISE", uniform=False, max_query_len=512))
    assert model._decode_cudagraph_observation == -1

    _forward_with_context(model, monkeypatch, lambda: _decode_context("PIECEWISE", uniform=False))
    assert model._decode_cudagraph_observation == -1

    _forward_with_context(
        model,
        monkeypatch,
        lambda: SimpleNamespace(cudagraph_runtime_mode=None, batch_descriptor=None, attn_metadata=None),
    )
    assert model._decode_cudagraph_observation == -1

    monkeypatch.setattr(alpamayo2_super_module, "get_forward_context", None)
    _forward_with_context(model, monkeypatch, None)
    assert model._decode_cudagraph_observation == -1
