# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import vllm_omni.diffusion.attention.backends.flashinfer_sage_attn_experimental as sage_backend
from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum


def _make_impl(monkeypatch: pytest.MonkeyPatch, **backend_kwargs):
    monkeypatch.setattr(sage_backend, "FlashInferAttentionImpl", lambda **_: object())
    return sage_backend.FlashInferSageAttentionExperimentalImpl(
        num_heads=8,
        num_kv_heads=2,
        head_size=128,
        softmax_scale=128**-0.5,
        backend_kwargs=backend_kwargs,
    )


def test_backend_is_registered():
    backend = DiffusionAttentionBackendEnum.FLASHINFER_SAGE_ATTN_EXPERIMENTAL
    assert backend.get_path().endswith(
        "flashinfer_sage_attn_experimental.FlashInferSageAttentionExperimentalBackend"
    )
    assert sage_backend.FlashInferSageAttentionExperimentalBackend.get_supported_head_sizes() == [128]


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("preprocess", "invalid", "preprocess must be"),
        ("workspace_reset", "invalid", "workspace_reset must be"),
    ],
)
def test_invalid_backend_option(monkeypatch: pytest.MonkeyPatch, key: str, value: str, match: str):
    with pytest.raises(ValueError, match=match):
        _make_impl(monkeypatch, **{key: value})


def test_strict_version_gate(monkeypatch: pytest.MonkeyPatch):
    impl = _make_impl(monkeypatch, require_tested_flashinfer_version=True)
    monkeypatch.setattr(impl, "_sm100_api_available", lambda: True)
    monkeypatch.setattr(impl, "_flashinfer_version", lambda: "999.0")
    query = torch.empty(1, 4, 8, 128)
    key = torch.empty(1, 4, 2, 128)
    reason = impl._sm100_ineligibility_reason(query, key, key)
    assert reason is not None
    assert "untested" in reason


def test_dynamic_shape_cache_replaces_stream_entry(monkeypatch: pytest.MonkeyPatch):
    sage_backend.FlashInferSageAttentionExperimentalImpl.clear_runtime_caches()
    allocated = []

    def fake_allocate(query, key):
        result = object()
        allocated.append(result)
        return result

    monkeypatch.setattr(sage_backend, "allocate_sage_quant_buffers", fake_allocate)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda _device=None: type("FakeStream", (), {"cuda_stream": 123})(),
    )
    first_q = torch.empty(1, 4, 8, 128)
    first_k = torch.empty(1, 5, 2, 128)
    second_q = torch.empty(1, 8, 8, 128)
    second_k = torch.empty(1, 9, 2, 128)

    first = sage_backend.FlashInferSageAttentionExperimentalImpl._triton_buffers(first_q, first_k)
    assert sage_backend.FlashInferSageAttentionExperimentalImpl._triton_buffers(first_q, first_k) is first
    second = sage_backend.FlashInferSageAttentionExperimentalImpl._triton_buffers(second_q, second_k)

    assert second is not first
    assert len(allocated) == 2
    assert len(sage_backend.FlashInferSageAttentionExperimentalImpl._triton_buffers_by_stream) == 1
    sage_backend.FlashInferSageAttentionExperimentalImpl.clear_runtime_caches()


def _has_sm100_sage() -> bool:
    return bool(
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (10, 0)
        and sage_backend.FlashInferSageAttentionExperimentalImpl._sm100_api_available()
        and sage_backend.HAS_SAGE_TRITON_PREPROCESS
    )


@pytest.mark.skipif(not _has_sm100_sage(), reason="requires FlashInfer SageAttention on SM100")
def test_sm100_triton_preprocess_matches_sdpa():
    torch.manual_seed(42)
    query = torch.randn(1, 256, 8, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, 269, 2, 128, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    impl = sage_backend.FlashInferSageAttentionExperimentalImpl(
        num_heads=8,
        num_kv_heads=2,
        head_size=128,
        softmax_scale=128**-0.5,
        backend_kwargs={
            "strict": True,
            "preprocess": "triton",
            "workspace_reset": "always",
        },
    )

    output = impl.forward_cuda(query, key, value)
    reference = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        enable_gqa=True,
    ).transpose(1, 2)
    cosine = F.cosine_similarity(output.float().flatten(), reference.float().flatten(), dim=0)

    assert torch.isfinite(output).all()
    assert cosine > 0.995
