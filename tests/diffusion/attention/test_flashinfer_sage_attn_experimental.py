# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("flashinfer")
pytest.importorskip("triton")

import vllm_omni.diffusion.attention.backends.flashinfer_sage_attn_experimental as sage
from vllm_omni.diffusion.attention.backends.flashinfer_sage_preprocess_compile import (
    allocate_sage_buffers,
)
from vllm_omni.diffusion.attention.backends.registry import (
    DiffusionAttentionBackendEnum,
)


class _Dense:
    def __init__(self, **_kwargs) -> None:
        self.calls = 0

    def forward_cuda(self, query, _key, _value, _metadata=None):
        self.calls += 1
        return query


class _TensorSpec:
    def __init__(self, shape, *, contiguous: bool = True) -> None:
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = torch.bfloat16
        self.device = torch.device("cuda")
        self._contiguous = contiguous

    def is_contiguous(self) -> bool:
        return self._contiguous


def _make_impl(monkeypatch: pytest.MonkeyPatch, *, causal: bool = False):
    monkeypatch.setattr(sage, "FlashInferAttentionImpl", _Dense)
    return sage.FlashInferSageAttentionExperimentalImpl(
        num_heads=8,
        num_kv_heads=2,
        head_size=128,
        softmax_scale=128**-0.5,
        causal=causal,
    )


def test_backend_is_registered():
    backend = DiffusionAttentionBackendEnum.FLASHINFER_SAGE_ATTN_EXPERIMENTAL
    assert backend.get_path().endswith("flashinfer_sage_attn_experimental.FlashInferSageAttentionExperimentalBackend")
    assert sage.FlashInferSageAttentionExperimentalBackend.get_supported_head_sizes() == [128]


@pytest.mark.parametrize(
    ("query", "key", "value", "causal", "match"),
    [
        (
            _TensorSpec((2, 32, 8, 128)),
            _TensorSpec((2, 33, 2, 128)),
            _TensorSpec((2, 33, 2, 128)),
            False,
            "batched odd",
        ),
        (
            _TensorSpec((1, 32, 8, 128), contiguous=False),
            _TensorSpec((1, 33, 2, 128)),
            _TensorSpec((1, 33, 2, 128)),
            False,
            "contiguous",
        ),
        (
            _TensorSpec((1, 32, 8, 128)),
            _TensorSpec((1, 33, 2, 128)),
            _TensorSpec((1, 33, 2, 128)),
            True,
            "non-causal",
        ),
    ],
)
def test_unsupported_inputs_are_rejected_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    query: _TensorSpec,
    key: _TensorSpec,
    value: _TensorSpec,
    causal: bool,
    match: str,
):
    impl = _make_impl(monkeypatch, causal=causal)
    reason = impl._ineligibility_reason(query, key, value, None)
    assert reason is not None
    assert match in reason


def test_fallback_calls_dense_backend(monkeypatch: pytest.MonkeyPatch):
    impl = _make_impl(monkeypatch, causal=True)
    query = torch.empty(1, 4, 8, 128)
    key = torch.empty(1, 5, 2, 128)
    output = impl.forward_cuda(query, key, key)
    assert output is query
    assert impl._dense.calls == 1


def test_stream_cache_reuses_workspace_and_replaces_shape(
    monkeypatch: pytest.MonkeyPatch,
):
    sage.FlashInferSageAttentionExperimentalImpl.clear_runtime_cache()
    monkeypatch.setattr(
        sage.FlashInferSageAttentionExperimentalImpl,
        "_stream_key",
        lambda _device: (0, 123),
    )
    first_q = torch.empty(1, 4, 8, 128)
    first_k = torch.empty(1, 5, 2, 128)
    second_q = torch.empty(1, 8, 8, 128)
    second_k = torch.empty(1, 9, 2, 128)

    first_workspace, first_buffers = sage.FlashInferSageAttentionExperimentalImpl._runtime(first_q, first_k, (10, 3))
    same_workspace, same_buffers = sage.FlashInferSageAttentionExperimentalImpl._runtime(first_q, first_k, (10, 3))
    second_workspace, second_buffers = sage.FlashInferSageAttentionExperimentalImpl._runtime(
        second_q, second_k, (10, 3)
    )

    assert first_workspace.numel() == sage._WORKSPACE_BYTES
    assert same_workspace is first_workspace is second_workspace
    assert same_buffers is first_buffers
    assert second_buffers is not first_buffers
    assert len(sage.FlashInferSageAttentionExperimentalImpl._runtime_by_stream) == 1
    sage.FlashInferSageAttentionExperimentalImpl.clear_runtime_cache()


@pytest.mark.parametrize(
    ("capability", "expected_dtype"),
    [((10, 0), torch.int8), ((10, 3), torch.float8_e4m3fn)],
)
def test_architecture_selects_qk_dtype(capability, expected_dtype):
    query = torch.empty(1, 4, 8, 128)
    key = torch.empty(1, 5, 2, 128)
    buffers = allocate_sage_buffers(query, key, capability)
    assert buffers.q.dtype == expected_dtype
    assert buffers.k.dtype == expected_dtype
    assert buffers.v.dtype == torch.float8_e4m3fn
    assert buffers.kv_lens.tolist() == [5]
    assert buffers.kv_indptr.tolist() == [0, 16]


def test_compilation_is_not_started_during_construction(monkeypatch: pytest.MonkeyPatch):
    compiled_module = importlib.import_module(
        "vllm_omni.diffusion.attention.backends.flashinfer_sage_preprocess_compile"
    )
    compiled_module._get_compiled_preprocessor.cache_clear()
    calls = 0

    def fake_compile(fn, **_kwargs):
        nonlocal calls
        calls += 1
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    _make_impl(monkeypatch)

    assert calls == 0


def test_compiled_callable_is_shared_across_layers(monkeypatch: pytest.MonkeyPatch):
    compiled_module = importlib.import_module(
        "vllm_omni.diffusion.attention.backends.flashinfer_sage_preprocess_compile"
    )
    compiled_module._get_compiled_preprocessor.cache_clear()
    calls = 0

    def fake_compile(fn, **kwargs):
        nonlocal calls
        calls += 1
        assert kwargs == {"fullgraph": True, "dynamic": False, "mode": "default"}
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    first = compiled_module._get_compiled_preprocessor((10, 0), 0, torch.bfloat16)
    second = compiled_module._get_compiled_preprocessor((10, 0), 0, torch.bfloat16)

    assert first is second
    assert calls == 1
    compiled_module._get_compiled_preprocessor.cache_clear()


def test_sm103_dispatch_contract(monkeypatch: pytest.MonkeyPatch):
    impl = _make_impl(monkeypatch)
    query = torch.empty(1, 4, 8, 128, dtype=torch.bfloat16)
    key = torch.empty(1, 5, 2, 128, dtype=torch.bfloat16)
    workspace = torch.zeros(sage._WORKSPACE_BYTES, dtype=torch.uint8)
    buffers = SimpleNamespace(
        kv_lens=torch.tensor([5], dtype=torch.int32),
        q_indptr=torch.tensor([0, 4], dtype=torch.int32),
        kv_indptr=torch.tensor([0, 16], dtype=torch.int32),
    )
    q = torch.empty(4, 8, 128, dtype=torch.float8_e4m3fn)
    k = torch.empty(16, 2, 128, dtype=torch.float8_e4m3fn)
    v = torch.empty_like(k)
    q_sfs = torch.empty(32, dtype=torch.float32)
    k_sfs = torch.empty(2, dtype=torch.float32)
    v_sfs = torch.empty(256, dtype=torch.float32)
    captured = {}

    monkeypatch.setattr(impl, "_ineligibility_reason", lambda *_args: None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (10, 3))
    monkeypatch.setattr(impl, "_runtime", lambda *_args: (workspace, buffers))
    monkeypatch.setattr(
        sage,
        "preprocess_sage_compiled",
        lambda *_args: (q, k, v, q_sfs, k_sfs, v_sfs),
    )

    def fake_launch(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return torch.empty(4, 8, 128, dtype=torch.bfloat16)

    monkeypatch.setattr(sage, "trtllm_ragged_attention_deepseek", fake_launch)
    output = impl.forward_cuda(query, key, key)

    assert output.shape == query.shape
    assert captured["args"][5:7] == (4, 16)
    assert captured["args"][9:13] == (-1, 1, -1, buffers.q_indptr)
    assert captured["kwargs"]["sage_attn_sfs"] == (q_sfs, k_sfs, None, v_sfs)
    assert captured["kwargs"]["num_elts_per_sage_attn_blk"] == (1, 16, 0, 1)


def _has_blackwell_sage() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in {
        (10, 0),
        (10, 3),
    }


@pytest.mark.skipif(not _has_blackwell_sage(), reason="requires SM100 or SM103")
def test_blackwell_output_is_finite_and_close_to_sdpa():
    torch.manual_seed(42)
    query = torch.randn(1, 256, 8, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, 269, 2, 128, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    impl = sage.FlashInferSageAttentionExperimentalImpl(
        num_heads=8,
        num_kv_heads=2,
        head_size=128,
        softmax_scale=128**-0.5,
    )

    output = impl.forward_cuda(query, key, value)
    reference = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        enable_gqa=True,
    ).transpose(1, 2)
    cosine = F.cosine_similarity(output.float().flatten(), reference.float().flatten(), dim=0)

    threshold = 0.998 if torch.cuda.get_device_capability() == (10, 3) else 0.995
    assert torch.isfinite(output).all()
    assert cosine > threshold
    workspace = next(iter(impl._runtime_by_stream.values()))[0]
    assert torch.count_nonzero(workspace) == 0
