# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton preprocessing for FlashInfer's Blackwell SageAttention cubins."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@dataclass
class SageBuffers:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    q_sfs: torch.Tensor
    k_sfs: torch.Tensor
    scratch: torch.Tensor
    q_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    kv_lens: torch.Tensor

    @property
    def v_sfs(self) -> torch.Tensor:
        return self.scratch[: self.v.shape[1] * self.v.shape[2]]


@triton.jit
def _q_quant_kernel(
    x,
    out,
    scales,
    total_tokens: tl.constexpr,
    heads: tl.constexpr,
    is_fp8: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // heads
    head = row - token * heads
    offsets = tl.arange(0, 128)
    values = tl.load(x + row * 128 + offsets).to(tl.float32)
    qmax = 448.0 if is_fp8 else 127.0
    scale = tl.maximum(tl.max(tl.abs(values), axis=0), 1.0e-12) / qmax
    quantized = values / scale
    if is_fp8:
        quantized = tl.maximum(tl.minimum(quantized, 448.0), -448.0)
    else:
        quantized = libdevice.rint(quantized)
        quantized = tl.maximum(tl.minimum(quantized, 127.0), -128.0)
    tl.store(out + row * 128 + offsets, quantized)
    tl.store(scales + head * total_tokens + token, scale)


@triton.jit
def _k_mean_kernel(
    x,
    means,
    logical_tokens: tl.constexpr,
    heads: tl.constexpr,
    block_tokens: tl.constexpr,
):
    index = tl.program_id(0)
    dim = index % 128
    head_batch = index // 128
    head = head_batch % heads
    batch = head_batch // heads
    accumulator = 0.0
    offsets = tl.arange(0, block_tokens)
    for start in range(0, logical_tokens, block_tokens):
        tokens = start + offsets
        pointers = x + ((batch * logical_tokens + tokens) * heads + head) * 128 + dim
        values = tl.load(pointers, mask=tokens < logical_tokens, other=0.0).to(tl.float32)
        accumulator += tl.sum(values, axis=0)
    tl.store(means + index, accumulator / logical_tokens)


@triton.jit
def _k_quant_kernel(
    x,
    means,
    out,
    scales,
    logical_tokens: tl.constexpr,
    physical_tokens: tl.constexpr,
    heads: tl.constexpr,
    blocks_per_batch: tl.constexpr,
    is_fp8: tl.constexpr,
):
    program = tl.program_id(0)
    head = program % heads
    block_batch = program // heads
    block = block_batch % blocks_per_batch
    batch = block_batch // blocks_per_batch
    offsets = tl.arange(0, 16 * 128)
    token = block * 16 + offsets // 128
    dim = offsets % 128
    valid = token < logical_tokens
    pointers = x + ((batch * logical_tokens + token) * heads + head) * 128 + dim
    values = tl.load(pointers, mask=valid, other=0.0).to(tl.float32)
    mean = tl.load(means + (batch * heads + head) * 128 + dim)
    values = tl.where(valid, values - mean, 0.0)
    qmax = 448.0 if is_fp8 else 127.0
    scale = tl.maximum(tl.max(tl.abs(values), axis=0), 1.0e-12) / qmax
    quantized = values / scale
    if is_fp8:
        quantized = tl.maximum(tl.minimum(quantized, 448.0), -448.0)
    else:
        quantized = libdevice.rint(quantized)
        quantized = tl.maximum(tl.minimum(quantized, 127.0), -128.0)
    output = out + ((batch * physical_tokens + token) * heads + head) * 128 + dim
    tl.store(output, quantized)
    total_blocks = tl.num_programs(0) // heads
    tl.store(scales + head * total_blocks + block_batch, scale)


@triton.jit
def _v_scale_kernel(
    x,
    scales,
    total_tokens: tl.constexpr,
    heads: tl.constexpr,
    block_tokens: tl.constexpr,
):
    index = tl.program_id(0)
    dim = index % 128
    head = index // 128
    accumulator = 1.0e-12
    offsets = tl.arange(0, block_tokens)
    for start in range(0, total_tokens, block_tokens):
        tokens = start + offsets
        pointers = x + (tokens * heads + head) * 128 + dim
        values = tl.load(pointers, mask=tokens < total_tokens, other=0.0).to(tl.float32)
        accumulator = tl.maximum(accumulator, tl.max(tl.abs(values), axis=0))
    tl.store(scales + index, accumulator / 448.0)


@triton.jit
def _v_quant_kernel(
    x,
    scales,
    out,
    logical_tokens: tl.constexpr,
    physical_tokens: tl.constexpr,
    heads: tl.constexpr,
):
    row = tl.program_id(0)
    head = row % heads
    token_batch = row // heads
    token = token_batch % physical_tokens
    batch = token_batch // physical_tokens
    offsets = tl.arange(0, 128)
    valid = token < logical_tokens
    pointers = x + ((batch * logical_tokens + token) * heads + head) * 128 + offsets
    values = tl.load(pointers, mask=valid, other=0.0).to(tl.float32)
    scale = tl.load(scales + head * 128 + offsets)
    tl.store(out + row * 128 + offsets, values / scale)


def allocate_sage_buffers(
    query: torch.Tensor,
    key: torch.Tensor,
    capability: tuple[int, int],
) -> SageBuffers:
    batch, q_len, q_heads, head_dim = query.shape
    kv_len, kv_heads = key.shape[1:3]
    physical_kv_len = (kv_len + 15) // 16 * 16
    qk_dtype = torch.float8_e4m3fn if capability == (10, 3) else torch.int8
    device = query.device
    return SageBuffers(
        q=torch.empty((batch * q_len, q_heads, head_dim), dtype=qk_dtype, device=device),
        k=torch.empty(
            (batch * physical_kv_len, kv_heads, head_dim),
            dtype=qk_dtype,
            device=device,
        ),
        v=torch.empty(
            (batch * physical_kv_len, kv_heads, head_dim),
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
        q_sfs=torch.empty(q_heads * batch * q_len, dtype=torch.float32, device=device),
        k_sfs=torch.empty(
            kv_heads * batch * (physical_kv_len // 16),
            dtype=torch.float32,
            device=device,
        ),
        scratch=torch.empty(batch * kv_heads * head_dim, dtype=torch.float32, device=device),
        q_indptr=torch.arange(batch + 1, dtype=torch.int32, device=device) * q_len,
        kv_indptr=torch.arange(batch + 1, dtype=torch.int32, device=device) * physical_kv_len,
        kv_lens=torch.full((batch,), kv_len, dtype=torch.int32, device=device),
    )


def preprocess_sage(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    buffers: SageBuffers,
    capability: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Smooth and quantize contiguous BSHD Q/K/V for K16 SageAttention."""
    batch, q_len, q_heads, head_dim = query.shape
    kv_len, kv_heads = key.shape[1:3]
    physical_kv_len = (kv_len + 15) // 16 * 16
    total_q = batch * q_len
    total_kv = batch * kv_len
    blocks_per_batch = physical_kv_len // 16
    is_fp8 = capability == (10, 3)

    _q_quant_kernel[(total_q * q_heads,)](
        query,
        buffers.q,
        buffers.q_sfs,
        total_tokens=total_q,
        heads=q_heads,
        is_fp8=is_fp8,
        num_warps=4,
    )
    _k_mean_kernel[(batch * kv_heads * head_dim,)](
        key,
        buffers.scratch,
        logical_tokens=kv_len,
        heads=kv_heads,
        block_tokens=1024,
        num_warps=4,
    )
    _k_quant_kernel[(batch * blocks_per_batch * kv_heads,)](
        key,
        buffers.scratch,
        buffers.k,
        buffers.k_sfs,
        logical_tokens=kv_len,
        physical_tokens=physical_kv_len,
        heads=kv_heads,
        blocks_per_batch=blocks_per_batch,
        is_fp8=is_fp8,
        num_warps=8,
    )
    _v_scale_kernel[(kv_heads * head_dim,)](
        value,
        buffers.v_sfs,
        total_tokens=total_kv,
        heads=kv_heads,
        block_tokens=1024,
        num_warps=4,
    )
    _v_quant_kernel[(batch * physical_kv_len * kv_heads,)](
        value,
        buffers.v_sfs,
        buffers.v,
        logical_tokens=kv_len,
        physical_tokens=physical_kv_len,
        heads=kv_heads,
        num_warps=4,
    )
    return (
        buffers.q,
        buffers.k,
        buffers.v,
        buffers.q_sfs,
        buffers.k_sfs,
        buffers.v_sfs,
    )
