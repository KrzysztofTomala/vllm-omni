# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal FlashInfer SageAttention backend for Blackwell diffusion models.

The fast path is deliberately narrow: contiguous FP16/BF16 BSHD tensors,
head dimension 128, non-causal attention, and SM100 or SM103. Other calls use
the dense FlashInfer backend. K/V storage is padded to select the measured K16
Sage cubin while logical sequence lengths keep padding out of the softmax.
"""

from __future__ import annotations

import torch
from flashinfer.prefill import trtllm_ragged_attention_deepseek
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.backends.flashinfer_attn import (
    FlashInferAttentionImpl,
)
from vllm_omni.diffusion.attention.backends.flashinfer_sage_preprocess_triton import (
    SageBuffers,
    allocate_sage_buffers,
    preprocess_sage,
)

logger = init_logger(__name__)

_CAPABILITIES = {(10, 0), (10, 3)}
_WORKSPACE_BYTES = 8192 * 256 * 4


class FlashInferSageAttentionExperimentalBackend(AttentionBackend):
    accept_output_buffer = True

    @classmethod
    def supports_attention_mask(cls) -> bool:
        return True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [128]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_SAGE_ATTN_EXPERIMENTAL"

    @staticmethod
    def get_impl_cls() -> type[FlashInferSageAttentionExperimentalImpl]:
        return FlashInferSageAttentionExperimentalImpl


class FlashInferSageAttentionExperimentalImpl(AttentionImpl):
    """Adapt vLLM-Omni BSHD attention to FlashInfer's Sage K16 cubins."""

    # One entry per CUDA stream is shared by every transformer layer. Reusing
    # these allocations is required for both memory use and steady-state speed.
    _runtime_by_stream: dict[tuple[int, int], tuple[torch.Tensor, tuple[object, ...], SageBuffers]] = {}

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        qkv_layout: str | None = None,
        backend_kwargs: dict | None = None,
        **extra_impl_args,
    ) -> None:
        self.softmax_scale = softmax_scale
        self.causal = causal
        self._dense = FlashInferAttentionImpl(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
            qkv_layout=qkv_layout,
            **extra_impl_args,
        )

    @staticmethod
    def _stream_key(device: torch.device) -> tuple[int, int]:
        device_index = device.index
        if device_index is None:
            device_index = torch.accelerator.current_device_index()
        return device_index, int(torch.cuda.current_stream(device).cuda_stream)

    @classmethod
    def _runtime(
        cls,
        query: torch.Tensor,
        key: torch.Tensor,
        capability: tuple[int, int],
    ) -> tuple[torch.Tensor, SageBuffers]:
        stream_key = cls._stream_key(query.device)
        shape_key: tuple[object, ...] = (
            capability,
            tuple(query.shape),
            tuple(key.shape),
        )
        cached = cls._runtime_by_stream.get(stream_key)
        if cached is None:
            workspace = torch.zeros(_WORKSPACE_BYTES, dtype=torch.uint8, device=query.device)
            buffers = allocate_sage_buffers(query, key, capability)
            cls._runtime_by_stream[stream_key] = (workspace, shape_key, buffers)
        elif cached[1] != shape_key:
            workspace = cached[0]
            buffers = allocate_sage_buffers(query, key, capability)
            cls._runtime_by_stream[stream_key] = (workspace, shape_key, buffers)
        else:
            workspace, _, buffers = cached
        return workspace, buffers

    @classmethod
    def clear_runtime_cache(cls) -> None:
        cls._runtime_by_stream.clear()

    def _ineligibility_reason(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> str | None:
        if query.device.type != "cuda":
            return "inputs are not CUDA tensors"
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            return "expected rank-4 BSHD tensors"
        if query.dtype not in (torch.float16, torch.bfloat16):
            return "expected FP16 or BF16 inputs"
        if key.dtype != query.dtype or value.dtype != query.dtype:
            return "Q/K/V dtypes differ"
        if key.device != query.device or value.device != query.device:
            return "Q/K/V devices differ"
        if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
            return "Q/K/V batch sizes differ"
        if key.shape[1] != value.shape[1]:
            return "K/V sequence lengths differ"
        if query.shape[1] == 0 or key.shape[1] == 0:
            return "empty sequence"
        if query.shape[3] != 128 or key.shape[3] != 128 or value.shape[3] != 128:
            return "head dimension is not 128"
        if key.shape[2] == 0 or key.shape[2] != value.shape[2] or query.shape[2] % key.shape[2]:
            return "unsupported Q/K/V head counts"
        if query.shape[0] > 1 and key.shape[1] % 16:
            return "batched odd K/V lengths are not packed safely"
        if not all(tensor.is_contiguous() for tensor in (query, key, value)):
            return "Triton preprocessing requires contiguous BSHD tensors"
        if self.causal:
            return "the Sage cubins are non-causal"
        if attn_metadata is not None and attn_metadata.attn_mask is not None:
            return "explicit masks are unsupported"
        capability = torch.cuda.get_device_capability(query.device)
        if capability not in _CAPABILITIES:
            return f"no Sage cubin for SM{capability[0]}{capability[1]}"
        return None

    def _fallback(
        self,
        reason: str,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> torch.Tensor:
        logger.warning_once("FlashInfer SageAttention using dense fallback: %s", reason)
        return self._dense.forward_cuda(query, key, value, attn_metadata)

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        reason = self._ineligibility_reason(query, key, value, attn_metadata)
        if reason is not None:
            return self._fallback(reason, query, key, value, attn_metadata)

        capability = torch.cuda.get_device_capability(query.device)
        batch, q_len, q_heads, head_dim = query.shape
        kv_len = key.shape[1]
        physical_kv_len = (kv_len + 15) // 16 * 16
        workspace, buffers = self._runtime(query, key, capability)
        q, k, v, q_sfs, k_sfs, v_sfs = preprocess_sage(query, key, value, buffers, capability)
        out = trtllm_ragged_attention_deepseek(
            q,
            k,
            v,
            workspace,
            buffers.kv_lens,
            q_len,
            physical_kv_len,
            self.softmax_scale,
            1.0,
            -1,
            batch,
            -1,
            buffers.q_indptr,
            buffers.kv_indptr,
            False,
            False,
            False,
            sage_attn_sfs=(q_sfs, k_sfs, None, v_sfs),
            num_elts_per_sage_attn_blk=(1, 16, 0, 1),
        )
        out = out.reshape(batch, q_len, q_heads, head_dim)
        return out if out.dtype == query.dtype else out.to(query.dtype)
