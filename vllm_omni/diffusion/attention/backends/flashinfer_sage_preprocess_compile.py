# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lazy ``torch.compile`` preprocessing for FlashInfer SageAttention.

Importing this module does not compile anything. The compiled callable is
created on the first opt-in SageAttention invocation and shared by every
transformer layer in the process. Static shape guards may compile another
graph when the prompt or video shape changes.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import torch
import torch.nn.functional as F
from vllm.logger import init_logger

logger = init_logger(__name__)

_Preprocessor = Callable[..., tuple[torch.Tensor, ...]]
_seen_shapes: set[tuple[object, ...]] = set()


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
        q=torch.empty(
            (batch * q_len, q_heads, head_dim),
            dtype=qk_dtype,
            device=device,
        ),
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
        q_sfs=torch.empty(
            q_heads * batch * q_len,
            dtype=torch.float32,
            device=device,
        ),
        k_sfs=torch.empty(
            kv_heads * batch * (physical_kv_len // 16),
            dtype=torch.float32,
            device=device,
        ),
        scratch=torch.empty(
            batch * kv_heads * head_dim,
            dtype=torch.float32,
            device=device,
        ),
        q_indptr=torch.arange(batch + 1, dtype=torch.int32, device=device) * q_len,
        kv_indptr=torch.arange(batch + 1, dtype=torch.int32, device=device) * physical_kv_len,
        kv_lens=torch.full((batch,), kv_len, dtype=torch.int32, device=device),
    )


def _make_preprocessor(*, is_fp8: bool) -> _Preprocessor:
    qmax = 448.0 if is_fp8 else 127.0

    def preprocess(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q_out: torch.Tensor,
        k_out: torch.Tensor,
        v_out: torch.Tensor,
        q_sfs_out: torch.Tensor,
        k_sfs_out: torch.Tensor,
        v_sfs_out: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        batch, q_len, q_heads, head_dim = query.shape
        kv_len, kv_heads = key.shape[1:3]
        physical_kv_len = (kv_len + 15) // 16 * 16

        q_float = query.float().reshape(batch * q_len, q_heads, head_dim)
        q_sfs = q_float.abs().amax(dim=-1).clamp_min(1.0e-12) / qmax
        q_quant = q_float / q_sfs.unsqueeze(-1)
        if is_fp8:
            q_quant = q_quant.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        else:
            q_quant = q_quant.round().clamp(-128.0, 127.0).to(torch.int8)

        k_float = key.float()
        k_centered = k_float - k_float.mean(dim=1, keepdim=True)
        k_centered = F.pad(
            k_centered,
            (0, 0, 0, 0, 0, physical_kv_len - kv_len),
        )
        k_blocks = k_centered.reshape(
            batch,
            physical_kv_len // 16,
            16,
            kv_heads,
            head_dim,
        )
        k_sfs = k_blocks.abs().amax(dim=(2, 4)).clamp_min(1.0e-12) / qmax
        k_quant = k_blocks / k_sfs[:, :, None, :, None]
        if is_fp8:
            k_quant = k_quant.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        else:
            k_quant = k_quant.round().clamp(-128.0, 127.0).to(torch.int8)

        v_float = value.float()
        v_sfs = v_float.abs().amax(dim=(0, 1)).clamp_min(1.0e-12) / 448.0
        v_quant = (v_float / v_sfs[None, None]).clamp(-448.0, 448.0)
        v_quant = F.pad(
            v_quant,
            (0, 0, 0, 0, 0, physical_kv_len - kv_len),
        ).to(torch.float8_e4m3fn)

        q_out.copy_(q_quant)
        k_out.copy_(k_quant.reshape(batch * physical_kv_len, kv_heads, head_dim))
        v_out.copy_(v_quant.reshape(batch * physical_kv_len, kv_heads, head_dim))
        q_sfs_out.copy_(q_sfs.T.reshape(-1))
        k_sfs_out.copy_(k_sfs.permute(2, 0, 1).reshape(-1))
        v_sfs_out.copy_(v_sfs.reshape(-1))
        return q_out, k_out, v_out, q_sfs_out, k_sfs_out, v_sfs_out

    return preprocess


@cache
def _get_compiled_preprocessor(
    capability: tuple[int, int],
    device_index: int,
    input_dtype: torch.dtype,
) -> _Preprocessor:
    # Device and dtype intentionally participate in the cache key even though
    # shape-specialization itself is managed by torch.compile guards.
    del device_index, input_dtype
    return torch.compile(
        _make_preprocessor(is_fp8=capability == (10, 3)),
        fullgraph=True,
        dynamic=False,
        mode="default",
    )


def preprocess_sage_compiled(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    buffers: SageBuffers,
    capability: tuple[int, int],
) -> tuple[torch.Tensor, ...]:
    """Quantize Q/K/V with one process-wide, lazily compiled callable."""
    device_index = query.device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    compiled = _get_compiled_preprocessor(capability, device_index, query.dtype)

    shape_key: tuple[object, ...] = (
        capability,
        device_index,
        query.dtype,
        tuple(query.shape),
        tuple(key.shape),
    )
    first_shape_call = shape_key not in _seen_shapes
    start = time.perf_counter()
    outputs = compiled(
        query,
        key,
        value,
        buffers.q,
        buffers.k,
        buffers.v,
        buffers.q_sfs,
        buffers.k_sfs,
        buffers.v_sfs,
    )
    if first_shape_call:
        elapsed = time.perf_counter() - start
        _seen_shapes.add(shape_key)
        logger.info(
            "FlashInfer Sage torch.compile preprocessing initialized for Q=%s K=%s in %.3f seconds",
            tuple(query.shape),
            tuple(key.shape),
            elapsed,
        )
    return outputs
