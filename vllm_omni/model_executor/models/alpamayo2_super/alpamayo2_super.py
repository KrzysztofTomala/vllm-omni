# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native Qwen3-VL rollout with the released Alpamayo 2 Super expert."""

from __future__ import annotations

import copy
import os
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import BatchFeature, DynamicCache, StaticCache
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
    Qwen3VLProcessor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.processor import cached_get_processor

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.runner_context import RunnerKVCacheContext

try:
    from vllm.forward_context import get_forward_context
except ImportError:  # pragma: no cover - vLLM builds without a forward context
    get_forward_context = None

logger = init_logger(__name__)

# Decode steps a guided child may wait for its unguided CFG twin before the
# expert runs guided-only. Overridden per request by ``_nav_cfg_max_hold_steps``.
_NAV_CFG_DEFAULT_MAX_HOLD_STEPS = 128

_BATCHED_POLICY_OUTPUT_KEYS = frozenset(
    {
        "pred_trajectories",
        "pred_rotations",
        "actions",
        "rotations",
        "normalized_controls",
        "action_noise",
        "action_expert_invocation_batch_size",
        "action_expert_attention_fa3",
    }
)

# HF attention implementation names the action expert may run with.
EXPERT_ATTENTION_BACKEND_FA3 = "alpamayo_fa3"
EXPERT_ATTENTION_BACKEND_SDPA = "sdpa"
_EXPERT_ATTENTION_BACKENDS = frozenset({EXPERT_ATTENTION_BACKEND_FA3, EXPERT_ATTENTION_BACKEND_SDPA})


def expert_fa3_supports(
    *,
    batch_rows: int,
    dtype: torch.dtype,
    device_type: str,
    is_causal: bool,
    dropout: float = 0.0,
    gqa_divisible: bool = True,
) -> bool:
    """Return whether one expert invocation may take the FA3 kernel path.

    Non-causal, dropout-free half precision on CUDA, with query heads that
    repeat the KV heads evenly. One prefix row takes the validated single varlen
    call; several rows (navigation guidance pairs a guided row with its unguided
    twin) take the two-segment paged call, provided the caller can hand over
    per-row prefix lengths. Anything else keeps the SDPA behavior.
    """
    return (
        int(batch_rows) >= 1
        and device_type == "cuda"
        and dtype in (torch.float16, torch.bfloat16)
        and not is_causal
        and float(dropout) == 0.0
        and bool(gqa_divisible)
    )


# Page size of the paged FA3 call used for multi-row batches; FA3 requires the
# page size to be a multiple of 256, so the allocated key length must be too.
EXPERT_FA3_PAGE_SIZE = 256


def _expert_fa3_attention_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Run vLLM's FA3 varlen kernel on one non-causal ``(1, heads, seq, dim)`` sample.

    ``torch.ops._vllm_fa3_C.fwd`` has no fake implementation, so calling it
    directly is a Dynamo graph break that splits every expert layer into
    separately compiled frames. Wrapping it in a custom op with a fake kernel
    lets the compiled expert trace through attention as one opaque node while
    eager execution (and manual CUDA-graph capture) still launches the real
    kernel. Callers decide FA3 eligibility in eager Python beforehand
    (``expert_fa3_supports``), so this op only sees dropout-free non-causal
    half-precision CUDA inputs. Returns ``(1, max_seqlen_q, query_heads, head_dim)``.
    """
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    query_heads, head_dim = query.shape[1], query.shape[-1]
    # With one sample the batch dimension merges into the sequence for free,
    # so these are views: the kernel only requires a unit stride on head_dim.
    output = flash_attn_varlen_func(
        query.transpose(1, 2).reshape(-1, query_heads, head_dim),
        key.transpose(1, 2).reshape(-1, key.shape[1], key.shape[-1]),
        value.transpose(1, 2).reshape(-1, value.shape[1], value.shape[-1]),
        max_seqlen_q,
        cu_seqlens_q,
        max_seqlen_k,
        cu_seqlens_k,
        softmax_scale=softmax_scale,
        causal=False,
        fa_version=3,
    )
    return output.view(1, max_seqlen_q, query_heads, head_dim)


def _expert_fa3_attention_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
) -> torch.Tensor:
    return query.new_empty((1, query.shape[2], query.shape[1], query.shape[-1]))


def _expert_fa3_attention_rows_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    prefix_lengths: torch.Tensor,
    suffix_positions: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Multi-row expert attention over right-padded prefixes plus a shared suffix.

    ``key``/``value`` are ``(rows, kv_heads, allocated, head_dim)`` cache
    tensors in which row ``r`` holds its valid prefix in ``[0, prefix_lengths[r])``
    and the current action tokens at ``suffix_positions`` (the same positions
    for every row), with padding in between and after. FA3's varlen kernel
    wants each sequence contiguous, so the attention is computed as two
    segments and merged exactly through the returned log-sum-exps: the prefix
    segment runs the paged kernel over the rows' page-aligned layout with
    ``seqused_k`` cutting each row at its own prefix length, and the suffix
    segment gathers the shared action positions. No prefix copy: the only data
    movement is the head/sequence transpose FA3 needs (also present in the
    single-row path) and the gather of the suffix tokens. Returns
    ``(rows, query_length, query_heads, head_dim)``.
    """
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    rows, query_heads, query_length, head_dim = query.shape
    kv_heads, allocated = key.shape[1], key.shape[2]
    pages = allocated // EXPERT_FA3_PAGE_SIZE
    device = query.device
    q = query.transpose(1, 2).reshape(rows * query_length, query_heads, head_dim)
    cu_seqlens_q = torch.arange(0, rows * query_length + 1, query_length, device=device, dtype=torch.int32)
    # Prefix segment: the cache rows are already page-aligned once heads and
    # sequence are swapped; page p of row r is block r * pages + p.
    paged_key = key.transpose(1, 2).reshape(rows * pages, EXPERT_FA3_PAGE_SIZE, kv_heads, head_dim)
    paged_value = value.transpose(1, 2).reshape(rows * pages, EXPERT_FA3_PAGE_SIZE, kv_heads, head_dim)
    block_table = (
        torch.arange(rows, device=device, dtype=torch.int32)[:, None] * pages
        + torch.arange(pages, device=device, dtype=torch.int32)[None, :]
    )
    prefix_out, prefix_lse = flash_attn_varlen_func(
        q,
        paged_key,
        paged_value,
        query_length,
        cu_seqlens_q,
        allocated,
        cu_seqlens_k=None,
        seqused_k=prefix_lengths.to(dtype=torch.int32),
        block_table=block_table,
        softmax_scale=softmax_scale,
        causal=False,
        fa_version=3,
        return_softmax_lse=True,
    )
    # Suffix segment: the action tokens of every row, gathered to a contiguous
    # varlen layout.
    suffix_key = key.index_select(2, suffix_positions).transpose(1, 2).reshape(rows * query_length, kv_heads, head_dim)
    suffix_value = (
        value.index_select(2, suffix_positions).transpose(1, 2).reshape(rows * query_length, kv_heads, head_dim)
    )
    suffix_out, suffix_lse = flash_attn_varlen_func(
        q,
        suffix_key,
        suffix_value,
        query_length,
        cu_seqlens_q,
        query_length,
        cu_seqlens_k=cu_seqlens_q,
        softmax_scale=softmax_scale,
        causal=False,
        fa_version=3,
        return_softmax_lse=True,
    )
    merged = _merge_lse_segments(prefix_out, prefix_lse, suffix_out, suffix_lse)
    return merged.to(query.dtype).view(rows, query_length, query_heads, head_dim)


def _merge_lse_segments(
    first_out: torch.Tensor,
    first_lse: torch.Tensor,
    second_out: torch.Tensor,
    second_lse: torch.Tensor,
) -> torch.Tensor:
    """Exactly combine two softmax partitions of one attention from their log-sum-exps.

    ``*_out`` are ``(total_q, heads, dim)``, ``*_lse`` are ``(heads, total_q)``.
    """
    first_lse = first_lse.transpose(0, 1).unsqueeze(-1)
    second_lse = second_lse.transpose(0, 1).unsqueeze(-1)
    peak = torch.maximum(first_lse, second_lse)
    first_weight = torch.exp(first_lse - peak)
    second_weight = torch.exp(second_lse - peak)
    return (first_out.float() * first_weight + second_out.float() * second_weight) / (first_weight + second_weight)


def _expert_fa3_attention_paged_impl(
    query: torch.Tensor,
    flat_cache: torch.Tensor,
    block_table: torch.Tensor,
    seqused_k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    key_size: list[int],
    key_stride: list[int],
    value_offset: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Expert attention over prefix plus action suffix, both already in vLLM's paged cache.

    ``flat_cache`` is one layer of the engine's cache as a ``(rows, head_dim)``
    view of its storage; the ``(num_blocks, block_size, kv_heads, head_dim)``
    K and V views are rebuilt from ``key_size``/``key_stride`` (elements) and
    ``value_offset`` (elements from K to V), so the op reads exactly what the
    caller wrote through ``flat_cache`` just before (the action tokens' K/V go
    into the rows' reserved pages by a plain ``index_copy_`` in the traced
    attention wrapper, which Inductor fuses like the StaticCache write).
    ``block_table`` (rows, max_blocks) int32 and ``seqused_k`` (rows,) int32
    cover prefix and suffix; one paged FA3 call serves both. Returns
    ``(rows, query_length, query_heads, head_dim)``.
    """
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    rows, query_heads, query_length, head_dim = query.shape
    base = flat_cache.storage_offset()
    key_cache = flat_cache.as_strided(key_size, key_stride, base)
    value_cache = flat_cache.as_strided(key_size, key_stride, base + value_offset)
    page_size = int(key_size[1])
    q = query.transpose(1, 2).reshape(rows * query_length, query_heads, head_dim)
    output = flash_attn_varlen_func(
        q,
        key_cache,
        value_cache,
        query_length,
        cu_seqlens_q,
        int(block_table.shape[1]) * page_size,
        cu_seqlens_k=None,
        seqused_k=seqused_k,
        block_table=block_table,
        softmax_scale=softmax_scale,
        causal=False,
        fa_version=3,
    )
    return output.view(rows, query_length, query_heads, head_dim)


def _expert_fa3_attention_paged_fake(
    query: torch.Tensor,
    flat_cache: torch.Tensor,
    block_table: torch.Tensor,
    seqused_k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    key_size: list[int],
    key_stride: list[int],
    value_offset: int,
    softmax_scale: float,
) -> torch.Tensor:
    return query.new_empty((query.shape[0], query.shape[2], query.shape[1], query.shape[-1]))


expert_fa3_attention_paged = torch.library.custom_op(
    "alpamayo2_super::expert_fa3_attention_paged",
    mutates_args=(),
    device_types="cuda",
)(_expert_fa3_attention_paged_impl)
expert_fa3_attention_paged.register_fake(_expert_fa3_attention_paged_fake)


@dataclass(frozen=True)
class PagedRowLayout:
    """Where token (block, slot, head) of a layer sits in its ``(rows, head_dim)`` flat view.

    ``row = block * block_stride + slot * slot_stride + head * head_stride``
    for K, plus ``value_offset`` for V (all in rows of ``head_dim`` elements).
    ``key_size``/``key_stride`` (elements) rebuild the ``(blocks, block_size,
    kv_heads, head_dim)`` K view over the flat storage; V starts
    ``value_offset * head_dim`` elements later. One layout serves every layer.
    """

    block_stride: int
    slot_stride: int
    head_stride: int
    value_offset: int
    key_size: tuple[int, ...]
    key_stride: tuple[int, ...]

    def rows(self, block: torch.Tensor, slot: torch.Tensor, kv_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Flat K and V rows for ``block``/``slot`` of shape (rows, tokens), ordered (row, head, token)."""
        heads = torch.arange(kv_heads, dtype=torch.long)[None, :, None]
        key_rows = (
            block.to(torch.long)[:, None, :] * self.block_stride
            + slot.to(torch.long)[:, None, :] * self.slot_stride
            + heads * self.head_stride
        ).reshape(-1)
        return key_rows, key_rows + self.value_offset


def paged_kv_flat_rows(cache: torch.Tensor, layer_index: int) -> tuple[torch.Tensor, PagedRowLayout]:
    """A layer's cache as a contiguous ``(rows, head_dim)`` view of its storage plus the row layout.

    vLLM hands out the logical ``[blocks, kv_heads, block_size, 2 * head_dim]``
    shape as a permuted view of the physical layout (NHD by default), so the
    row of token (block, slot, head) is read off the K view's strides rather
    than assumed from the shape. Works for every layout ``split_paged_kv_cache``
    accepts as long as the tensor is one permuted contiguous block (a packed
    multi-layer backing is not).
    """
    key_view, value_view = split_paged_kv_cache(cache, layer_index)
    head_dim = int(key_view.shape[-1])
    block_stride, slot_stride, head_stride, last_stride = key_view.stride()
    if last_stride != 1 or any(stride % head_dim for stride in (block_stride, slot_stride, head_stride)):
        raise RuntimeError(
            f"The paged expert KV path needs head_dim-aligned strides; layer {layer_index} has {key_view.stride()}"
        )
    by_stride = sorted(range(cache.ndim), key=lambda dim: -cache.stride(dim))
    physical = cache.permute(by_stride)
    if not physical.is_contiguous():
        raise RuntimeError(
            f"The paged expert KV path writes through a flat view; layer {layer_index} is not contiguous"
        )
    value_offset = value_view.storage_offset() - key_view.storage_offset()
    if value_offset % head_dim or key_view.storage_offset() != physical.storage_offset():
        raise RuntimeError(f"The paged expert KV path needs K/V offsets aligned to head_dim; layer {layer_index}")
    return physical.view(-1, head_dim), PagedRowLayout(
        block_stride=block_stride // head_dim,
        slot_stride=slot_stride // head_dim,
        head_stride=head_stride // head_dim,
        value_offset=value_offset // head_dim,
        key_size=tuple(int(dim) for dim in key_view.shape),
        key_stride=tuple(int(stride) for stride in key_view.stride()),
    )


@dataclass
class PagedPrefixContext:
    """Where the expert's rows live in the engine's paged KV cache.

    ``key_caches``/``value_caches`` are per-layer ``(blocks, block_size,
    kv_heads, head_dim)`` views of the engine's cache tensors (persistent for
    the engine's lifetime, so a captured graph may read them), ``flat_caches``
    the same layers as ``(rows, head_dim)`` and ``raw_caches`` the layer
    tensors themselves, for whole-block copies. Each row's page table lists
    its full prefix blocks followed by the row's reserved pages (blocks the
    runner allocated beyond the scheduler's pool). The prefix's last,
    partially filled block is copied into the first reserved page
    (``tail_source`` -> ``tail_target``; a self-copy when the prefix ends on
    a block boundary) so the action tokens continue it: the suffix K/V of
    (row, head, token) go to flat rows ``suffix_key_rows``/``suffix_value_rows``.
    ``seqused`` is prefix plus suffix length per row. All small tensors are
    refreshed per request with ``copy_`` so their addresses stay fixed.
    """

    key_caches: list[torch.Tensor]
    value_caches: list[torch.Tensor]
    flat_caches: list[torch.Tensor]
    raw_caches: list[torch.Tensor]
    layout: PagedRowLayout
    block_table: torch.Tensor
    seqused: torch.Tensor
    cu_seqlens_q: torch.Tensor
    suffix_key_rows: torch.Tensor
    suffix_value_rows: torch.Tensor
    tail_source: torch.Tensor
    tail_target: torch.Tensor

    def clone_indices(self) -> PagedPrefixContext:
        """A context with private copies of the per-request index tensors (graph inputs)."""
        return PagedPrefixContext(
            self.key_caches,
            self.value_caches,
            self.flat_caches,
            self.raw_caches,
            self.layout,
            self.block_table.clone(),
            self.seqused.clone(),
            self.cu_seqlens_q.clone(),
            self.suffix_key_rows.clone(),
            self.suffix_value_rows.clone(),
            self.tail_source.clone(),
            self.tail_target.clone(),
        )

    def copy_indices_from(self, other: PagedPrefixContext) -> None:
        self.block_table.copy_(other.block_table)
        self.seqused.copy_(other.seqused)
        self.suffix_key_rows.copy_(other.suffix_key_rows)
        self.suffix_value_rows.copy_(other.suffix_value_rows)
        self.tail_source.copy_(other.tail_source)
        self.tail_target.copy_(other.tail_target)

    def relocate_tails(self) -> None:
        """Copy each row's partial last prefix block into its first reserved page (every layer)."""
        for cache in self.raw_caches:
            block_dim = paged_kv_block_dim(cache)
            cache.index_copy_(block_dim, self.tail_target, cache.index_select(block_dim, self.tail_source))


class SuffixPassthroughCache(StaticCache):
    """A StaticCache-shaped cache that stores nothing.

    In paged mode the attention wrapper writes the action tokens' K/V into the
    engine's reserved pages itself, so HF's cache only has to hand the fresh
    states through; keeping the StaticCache interface leaves the expert's
    mask/position handling unchanged.
    """

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer = self.layers[layer_idx]
        if not layer.is_initialized:
            layer.lazy_initialization(key_states, value_states)
        return key_states, value_states


# Set while the expert runs (eagerly, or during graph capture) in paged mode;
# the attention wrapper picks it up because HF does not pass the cache object
# down to the attention interface.
_ACTIVE_PAGED_PREFIX: PagedPrefixContext | None = None


@contextmanager
def paged_prefix_scope(context: PagedPrefixContext | None):
    global _ACTIVE_PAGED_PREFIX
    previous = _ACTIVE_PAGED_PREFIX
    _ACTIVE_PAGED_PREFIX = context
    try:
        yield
    finally:
        _ACTIVE_PAGED_PREFIX = previous


def split_paged_kv_cache(cache: torch.Tensor, layer_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (key_cache, value_cache) as ``(num_blocks, block_size, kv_heads, head_dim)`` views.

    No copy is made: FA3 reads the pages through their strides (only the
    head_dim axis has to be contiguous), so the same layouts
    ``_gather_prefix_cache`` understands are exposed as strided views.
    """
    if cache.ndim == 4 and cache.shape[-1] % 2 == 0:
        # FlashAttention: [blocks, kv_heads, block_size, 2 * head_dim].
        head_dim = cache.shape[-1] // 2
        return (
            cache[..., :head_dim].permute(0, 2, 1, 3),
            cache[..., head_dim:].permute(0, 2, 1, 3),
        )
    if cache.ndim == 5 and cache.shape[0] == 2:
        # [2, blocks, block_size, kv_heads, head_dim].
        return cache[0], cache[1]
    if cache.ndim == 5 and cache.shape[1] == 2:
        # [blocks, 2, block_size, kv_heads, head_dim].
        return cache[:, 0], cache[:, 1]
    raise RuntimeError(
        "The paged expert KV path needs a supported vLLM paged KV layout; "
        f"layer {layer_index} has shape {tuple(cache.shape)}"
    )


def paged_kv_block_dim(cache: torch.Tensor) -> int:
    """The block axis of a supported paged layout (see ``split_paged_kv_cache``)."""
    if cache.ndim == 5 and cache.shape[0] == 2:
        return 1
    return 0


def _expert_fa3_attention_rows_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    prefix_lengths: torch.Tensor,
    suffix_positions: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    return query.new_empty((query.shape[0], query.shape[2], query.shape[1], query.shape[-1]))


expert_fa3_attention_rows = torch.library.custom_op(
    "alpamayo2_super::expert_fa3_attention_rows",
    mutates_args=(),
    device_types="cuda",
)(_expert_fa3_attention_rows_impl)
expert_fa3_attention_rows.register_fake(_expert_fa3_attention_rows_fake)


def expert_row_prefix_lengths(attention_mask: torch.Tensor, query_length: int) -> torch.Tensor:
    """Per-row valid prefix lengths from the expert's additive attention mask.

    The mask is ``(rows, 1, query_length, allocated)`` with zeros at attendable
    positions: each row's right-padded prefix plus the ``query_length`` action
    positions shared by all rows. Counting the zeros of one query row and
    removing the action positions yields the prefix length on the device, so
    the compiled expert traces this without a host synchronization.
    """
    attendable = (attention_mask[:, 0, 0, :] == 0).sum(dim=-1)
    return (attendable - query_length).to(dtype=torch.int32)


# CUDA-only: eligibility (device, dtype, non-causal) is decided by
# ``expert_fa3_supports`` in eager Python before the op is reached.
expert_fa3_attention = torch.library.custom_op(
    "alpamayo2_super::expert_fa3_attention",
    mutates_args=(),
    device_types="cuda",
)(_expert_fa3_attention_impl)
expert_fa3_attention.register_fake(_expert_fa3_attention_fake)


def alpamayo_flash_attention_3_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Run the single-sample action suffix through vLLM's bundled FA3.

    The action expert has 64 non-causal query tokens attending to a much
    longer VLM prefix. PyTorch SDPA selects its SM80 memory-efficient kernel
    for that unequal-Q/K shape, while vLLM's FA3 varlen kernel is native to
    Hopper and accepts the expert's GQA layout without repeating K/V heads.

    StaticCache allocates more K/V tokens than are valid. ``cache_position``
    supplies the live sequence length to FA3 through device-side cumulative
    lengths, preserving CUDA-graph replay without a host synchronization.
    Shapes outside ``expert_fa3_supports`` fall back to HF SDPA; supported ones
    launch the kernel through the ``expert_fa3_attention`` custom op so the
    compiled expert traces through attention without a graph break.
    """

    effective_is_causal = bool(is_causal) if is_causal is not None else bool(getattr(module, "is_causal", True))
    use_fa3 = (
        query.shape[0] == key.shape[0] == value.shape[0]
        and key.shape == value.shape
        and query.shape[-1] == key.shape[-1]
        and expert_fa3_supports(
            batch_rows=int(query.shape[0]),
            dtype=query.dtype,
            device_type=query.device.type,
            is_causal=effective_is_causal,
            dropout=dropout,
            gqa_divisible=query.shape[1] % key.shape[1] == 0,
        )
    )
    if not use_fa3:
        if _ACTIVE_PAGED_PREFIX is not None:
            raise RuntimeError(
                "The paged expert KV path requires the FA3 expert attention kernel (non-causal, half precision, CUDA)"
            )
        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            is_causal=is_causal,
            **kwargs,
        )

    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise RuntimeError(
            "The Alpamayo action expert FA3 path requires query, key, and value to share one dtype; "
            f"got query={query.dtype}, key={key.dtype}, value={value.dtype}"
        )

    query_length = query.shape[2]
    allocated_key_length = key.shape[2]
    cache_position = kwargs.get("cache_position")
    softmax_scale = float(scaling) if scaling is not None else query.shape[-1] ** -0.5
    paged = _ACTIVE_PAGED_PREFIX
    if paged is not None:
        # key/value are this step's action tokens. They go into the rows'
        # reserved pages with a plain index_copy_ (traced, so Inductor fuses it
        # with the RoPE output like the StaticCache write); the op then reads
        # prefix and suffix where the engine keeps them.
        layer_index = int(module.layer_idx)
        flat_cache = paged.flat_caches[layer_index]
        head_dim = query.shape[-1]
        flat_cache.index_copy_(0, paged.suffix_key_rows, key.reshape(-1, head_dim))
        flat_cache.index_copy_(0, paged.suffix_value_rows, value.reshape(-1, head_dim))
        output = expert_fa3_attention_paged(
            query,
            flat_cache,
            paged.block_table,
            paged.seqused,
            paged.cu_seqlens_q,
            list(paged.layout.key_size),
            list(paged.layout.key_stride),
            paged.layout.value_offset * head_dim,
            softmax_scale,
        )
        return output, None
    if query.shape[0] > 1:
        multi_row_ready = (
            attention_mask is not None
            and cache_position is not None
            and allocated_key_length % EXPERT_FA3_PAGE_SIZE == 0
            and attention_mask.shape[0] == query.shape[0]
            and attention_mask.shape[-1] == allocated_key_length
        )
        if not multi_row_ready:
            return sdpa_attention_forward(
                module,
                query,
                key,
                value,
                attention_mask,
                dropout=dropout,
                scaling=scaling,
                is_causal=is_causal,
                **kwargs,
            )
        output = expert_fa3_attention_rows(
            query,
            key,
            value,
            expert_row_prefix_lengths(attention_mask, query_length),
            cache_position.reshape(-1),
            softmax_scale,
        )
        return output, None

    if cache_position is None:
        valid_key_length = torch.scalar_tensor(
            allocated_key_length,
            device=query.device,
            dtype=torch.int32,
        )
    else:
        valid_key_length = cache_position.reshape(-1)[-1].to(dtype=torch.int32) + 1
    cu_seqlens_q = torch.stack((valid_key_length.new_zeros(()), valid_key_length.new_full((), query_length)))
    cu_seqlens_k = torch.stack((valid_key_length.new_zeros(()), valid_key_length))
    output = expert_fa3_attention(
        query,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_k,
        query_length,
        allocated_key_length,
        softmax_scale,
    )
    return output, None


AttentionInterface.register(EXPERT_ATTENTION_BACKEND_FA3, alpamayo_flash_attention_3_forward)


def fa3_expert_attention_available() -> tuple[bool, str]:
    """Report whether the ``alpamayo_fa3`` expert backend can run on this worker.

    Returns ``(ok, reason)``. The reason names the first unmet requirement, or
    the satisfied one, so startup logging and the fallback error can quote it.
    vLLM ships FA3 for Hopper only; newer parts are accepted solely when the
    bundled ``is_fa_version_supported`` helper vouches for version 3.
    """
    if EXPERT_ATTENTION_BACKEND_FA3 not in ALL_ATTENTION_FUNCTIONS:
        return False, f"{EXPERT_ATTENTION_BACKEND_FA3!r} is not registered with transformers AttentionInterface"
    try:
        from vllm.vllm_flash_attn import flash_attn_varlen_func  # noqa: F401
    except ImportError as exc:
        return False, f"vllm.vllm_flash_attn is unavailable: {exc}"
    if not torch.cuda.is_available():
        return False, "CUDA is unavailable"
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception as exc:  # pragma: no cover - driver/device query failure
        return False, f"CUDA compute capability query failed: {exc}"
    capability = f"{major}.{minor}"
    if major == 9:
        return True, f"FA3 available on compute capability {capability}"
    if major > 9:
        try:
            from vllm.vllm_flash_attn.flash_attn_interface import is_fa_version_supported

            if bool(is_fa_version_supported(3)):
                return True, f"FA3 reported supported on compute capability {capability}"
        except Exception:  # An absent helper or a failed probe both mean unsupported.
            pass
    return False, f"unsupported compute capability {capability} (FA3 requires 9.x)"


def resolve_expert_attention_backend(
    requested: str,
    *,
    fa3_available: bool,
    allow_fallback: bool,
    unavailable_reason: str | None = None,
) -> str:
    """Return the HF attention implementation the action expert will use.

    ``requested`` is the ``expert_attention_backend`` policy-server setting.
    An unavailable FA3 request fails loudly unless the deployment opted into
    ``allow_expert_attention_fallback``, in which case SDPA is used and a
    warning is logged so silent performance regressions cannot hide.
    """
    if requested not in _EXPERT_ATTENTION_BACKENDS:
        raise ValueError(
            f"Unknown Alpamayo expert_attention_backend {requested!r}; "
            f"expected one of {sorted(_EXPERT_ATTENTION_BACKENDS)}"
        )
    if requested != EXPERT_ATTENTION_BACKEND_FA3 or fa3_available:
        return requested
    reason = unavailable_reason or "FA3 is unavailable on this worker"
    if not allow_fallback:
        raise RuntimeError(
            f"expert_attention_backend={requested!r} was requested but {reason}. "
            "Set allow_expert_attention_fallback: true in policy_server_config (or the "
            "deployment's fallback switch) to run the action expert with "
            f"{EXPERT_ATTENTION_BACKEND_SDPA!r} instead."
        )
    logger.warning(
        "Alpamayo action expert attention backend %r is unavailable (%s); falling back to %r",
        requested,
        reason,
        EXPERT_ATTENTION_BACKEND_SDPA,
    )
    return EXPERT_ATTENTION_BACKEND_SDPA


# Inductor picks the launch configuration of its generated Triton reduction
# kernels by benchmarking a few candidates at first use. For the expert's
# RMSNorm-style reductions one of the candidates yields trajectories about a
# metre away from the eager layers (and from the other candidates), so which
# variant a process serves depended on timing at compile time. Inductor's
# deterministic mode replaces the benchmark with a fixed choice; measured cost
# on H100 is within noise. Set to "0" only to reproduce the old behaviour.
EXPERT_COMPILE_DETERMINISTIC = os.getenv("ALPAMAYO_EXPERT_COMPILE_DETERMINISTIC", "1") != "0"
# Triton 3.7.1 miscompiles the Inductor pointwise kernel that builds the
# expert's interleaved-mRoPE angle table for exactly XBLOCK=256 with four
# warps (a lane permutation in one third of the first mRoPE section; every
# other block size and warp count agrees bit for bit). Inductor's pointwise
# autotuner offers 128 and 256 for that kernel and picks by timing, so about
# one process in three rotated q/k with a wrong table and returned trajectories
# about a metre off; with autotuning disabled the single default configuration
# is the bad one. The guard below removes that launch configuration from every
# pointwise candidate list handed to Inductor's autotuner in this process (the
# backbone compiles in the same process and could hit the same bug) and falls
# back to XBLOCK=128 when it was the only candidate. Set
# ALPAMAYO_INDUCTOR_POINTWISE_GUARD=0 to disable. Standalone reproduction:
# perf_study_20260908/triton_repro.
INDUCTOR_POINTWISE_GUARD = os.getenv("ALPAMAYO_INDUCTOR_POINTWISE_GUARD", "1") != "0"
_BAD_POINTWISE_XBLOCK = 256
_BAD_POINTWISE_NUM_WARPS = 4
_FALLBACK_POINTWISE_XBLOCK = 128


def filter_pointwise_configs(configs: Sequence[Any]) -> list[Any]:
    """Drop the miscompiling (XBLOCK=256, num_warps=4) pointwise configuration."""
    kept = [
        cfg
        for cfg in configs
        if not (cfg.kwargs.get("XBLOCK") == _BAD_POINTWISE_XBLOCK and int(cfg.num_warps) == _BAD_POINTWISE_NUM_WARPS)
    ]
    if kept or not configs:
        return kept
    from triton import Config

    first = configs[0]
    return [
        Config(
            {**first.kwargs, "XBLOCK": _FALLBACK_POINTWISE_XBLOCK},
            num_warps=int(first.num_warps),
            num_stages=int(first.num_stages),
        )
    ]


def install_inductor_pointwise_config_guard() -> bool:
    """Wrap Inductor's autotuner entry point so pointwise kernels never use the bad configuration."""
    from torch._inductor.runtime import triton_heuristics
    from torch._inductor.runtime.hints import HeuristicType

    original = triton_heuristics.cached_autotune
    if getattr(original, "_alpamayo_pointwise_guard", False):
        return False

    def guarded_cached_autotune(
        size_hints: Any, configs: Any, triton_meta: Any, heuristic_type: Any, *args: Any, **kwargs: Any
    ) -> Any:
        if heuristic_type == HeuristicType.POINTWISE:
            configs = filter_pointwise_configs(list(configs))
        return original(size_hints, configs, triton_meta, heuristic_type, *args, **kwargs)

    guarded_cached_autotune._alpamayo_pointwise_guard = True  # type: ignore[attr-defined]
    guarded_cached_autotune._alpamayo_original = original  # type: ignore[attr-defined]
    triton_heuristics.cached_autotune = guarded_cached_autotune
    logger.info("Inductor pointwise autotune guard installed (drops XBLOCK=256/num_warps=4 candidates)")
    return True


if INDUCTOR_POINTWISE_GUARD:
    install_inductor_pointwise_config_guard()


def compile_action_expert(expert: nn.Module) -> nn.Module:
    """Compile the expert decoder stack as one Dynamo frame.

    The decoder's layer loop unrolls into a single graph, so every layer's
    ``layer_idx`` is a trace-time constant. With a graph break inside
    attention, ``Cache.update`` and the attention call instead became shared
    per-layer frames whose ``layer_idx`` guards recompiled once per layer,
    exhausting ``torch._dynamo.config.recompile_limit`` and leaving the
    remaining layers eager. ``fullgraph=True`` turns any future graph break
    into a startup error that names the break instead of that silent
    fallback. Shapes are static because ``_run_manual_action_graph`` fixes the
    StaticCache, mask, and action suffix per captured graph; Inductor's own
    CUDA graphs stay off since that method captures the whole integration.
    Inference only: the FA3 op has no backward, and the model runner already
    executes under ``torch.inference_mode``.
    """
    logger.info(
        "Compiling the Alpamayo action expert as a single graph (fullgraph=True, static shapes, deterministic=%s)",
        EXPERT_COMPILE_DETERMINISTIC,
    )
    return torch.compile(
        expert,
        fullgraph=True,
        dynamic=False,
        options={"triton.cudagraphs": False, "deterministic": EXPERT_COMPILE_DETERMINISTIC},
    )


class PinnedHostPixelValuesProcessor(Qwen3VLProcessor):
    """Qwen3-VL processor with a cheaper host hand-off and placeholder check.

    vLLM runs the fast HF image processor on the GPU (``device="cuda"``) and
    then moves its outputs to the host before serializing them for the engine
    process. For the 24 camera frames of one request that output is a 106 MB
    float32 tensor, and the implicit pageable copy took about 57 ms per
    request on an H100 host, most of the frontend-to-engine gap. The vision
    tower casts ``pixel_values`` to its bf16 weight dtype before the first
    kernel, so casting here is the same single rounding step; it halves the
    bytes crossing the process boundary, and a pinned destination turns the
    pageable copy into a few-millisecond DMA. vLLM's own host move then finds
    the tensor already on the host.
    """

    def _check_special_mm_tokens(self, text: list[str], text_inputs: Any, modalities: list[str]) -> None:
        # Same check as ProcessorMixin (the placeholder count in the text must
        # match the token count in input_ids), but counted on plain Python
        # ints. The base implementation iterates the input_ids tensor row into
        # 0-d tensors and calls list.count on them, which took about 20 ms per
        # request for a 4,600-token prompt, two thirds of the processor call.
        input_ids = text_inputs["input_ids"]
        rows = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else [list(ids) for ids in input_ids]
        for modality in modalities:
            token_str = getattr(self, f"{modality}_token", None)
            token_id = getattr(self, f"{modality}_token_id", None)
            if token_str is None or token_id is None:
                continue
            ids_count = [row.count(token_id) for row in rows]
            text_count = [sample.count(token_str) for sample in text]
            if ids_count != text_count:
                raise ValueError(
                    f"Mismatch in `{modality}` token count between text and `input_ids`. "
                    f"Got ids={ids_count} and text={text_count}. Likely due to `truncation='max_length'`. "
                    "Please disable truncation or increase `max_length`."
                )

    def __call__(self, *args: Any, **kwargs: Any) -> BatchFeature:
        outputs = super().__call__(*args, **kwargs)
        for key in ("pixel_values", "pixel_values_videos"):
            values = outputs.get(key) if hasattr(outputs, "get") else None
            if isinstance(values, torch.Tensor) and values.device.type == "cuda":
                outputs[key] = self._to_pinned_host_bf16(values)
        return outputs

    @staticmethod
    def _to_pinned_host_bf16(values: torch.Tensor) -> torch.Tensor:
        staged = values.to(torch.bfloat16) if values.is_floating_point() else values
        # PyTorch's caching host allocator hands back a recycled pinned block
        # after the first request and only reuses it once this tensor is
        # released, so a fresh tensor per call is both cheap and safe for the
        # processor cache that may keep the result alive across requests.
        host = torch.empty(staged.shape, dtype=staged.dtype, pin_memory=True)
        host.copy_(staged, non_blocking=True)
        torch.cuda.current_stream(staged.device).synchronize()
        return host


class Alpamayo2SuperProcessingInfo(Qwen3VLProcessingInfo):
    """Use the processor and extended tokenizer shipped with Super."""

    def get_hf_processor(self, **kwargs: object) -> Any:
        config = self.get_hf_config()
        kwargs.pop("device", None)
        return cached_get_processor(
            config._name_or_path,
            processor_cls=PinnedHostPixelValuesProcessor,
            tokenizer=self.get_tokenizer(),
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )


class Alpamayo2SuperMultiModalProcessor(Qwen3VLMultiModalProcessor):
    """Process image-only requests without the dummy-text detour.

    The NIM sends prompts as token ids, so vLLM calls the HF processor for the
    multimodal data only. For processors with a custom text path (Qwen3-VL
    has one for videos) vLLM synthesizes a dummy prompt with one placeholder per
    image, tokenizes it, expands the placeholders and validates the counts,
    and then discards the text: about 28 ms per request for the 24 camera
    frames, more than ten times the image processing itself. Alpamayo requests
    carry images only, so hand the images straight to the processor; the
    pixel values and grid are identical to the text path's.
    """

    def _apply_hf_processor_mm_only(
        self,
        mm_items: Any,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        counts = mm_items.get_all_counts()
        if {modality for modality, count in counts.items() if count} != {"image"}:
            return super()._apply_hf_processor_mm_only(mm_items, hf_processor_mm_kwargs, tokenization_kwargs)
        processor_data, passthrough_data = self._get_hf_mm_data(mm_items.select({"image"}))
        processed = self.info.ctx.call_hf_processor(
            self.info.get_hf_processor(**hf_processor_mm_kwargs),
            processor_data,
            dict(**hf_processor_mm_kwargs, **tokenization_kwargs),
        )
        processed.update(passthrough_data)
        return processed


@MULTIMODAL_REGISTRY.register_processor(
    Alpamayo2SuperMultiModalProcessor,
    info=Alpamayo2SuperProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Alpamayo2SuperForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """Run Super's VLM in vLLM and its terminal diffusion expert in-model.

    The autoregressive Qwen3-VL prefix stays entirely native to vLLM. Once
    ``<|traj_future_start|>`` is emitted, the model gathers that request's
    paged KV prefix and gives it to the released, non-causal action expert for
    fixed-step flow matching. Parallel samples are separate vLLM requests so
    every trajectory is conditioned on its own sampled VLM reasoning.
    """

    have_multimodal_outputs = True
    needs_runner_kv_cache = True
    # This is a terminal single-stage policy. Only the structured action
    # payload is client-facing; no downstream stage consumes VLM hidden states.
    requires_full_prefix_cached_hidden_states = False
    omni_pooler_payload_include_hidden = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.alpamayo_config = vllm_config.model_config.hf_config

        # The released package owns the action-space math and expert modules.
        # Keeping it an image/model dependency avoids duplicating architecture
        # code in the engine while leaving the VLM on vLLM's native Qwen path.
        from alpamayo2_super.models.expert import ExpertModel, ExpertModelConfig

        expert_config = ExpertModelConfig(**copy.deepcopy(self.alpamayo_config.expert_config))
        policy_config = getattr(self.alpamayo_config, "policy_server_config", None) or {}
        if hasattr(policy_config, "model_dump"):
            policy_config = policy_config.model_dump()
        elif not isinstance(policy_config, Mapping):
            policy_config = vars(policy_config)
        # The diffusion suffix needs a non-causal 4D mask, which both HF SDPA
        # and the registered FA3 path honor. Resolve the backend once so a
        # requested-but-unavailable FA3 is a startup error, not a silent no-op.
        self.expert_attention_backend_requested = str(
            policy_config.get("expert_attention_backend", EXPERT_ATTENTION_BACKEND_SDPA)
        )
        fa3_available, fa3_reason = fa3_expert_attention_available()
        self.expert_attention_backend = resolve_expert_attention_backend(
            self.expert_attention_backend_requested,
            fa3_available=fa3_available,
            allow_fallback=bool(policy_config.get("allow_expert_attention_fallback", False)),
            unavailable_reason=fa3_reason,
        )
        logger.info(
            "Alpamayo action expert attention: requested=%s effective=%s (%s)",
            self.expert_attention_backend_requested,
            self.expert_attention_backend,
            fa3_reason,
        )
        expert_config.llm_config._attn_implementation = self.expert_attention_backend
        # Paged expert KV (see PagedPrefixContext): the runner reserves pages
        # for up to this many concurrent expert rows; make_omni_output records
        # where they are.
        self._paged_expert_kv_enabled = bool(policy_config.get("paged_expert_kv", False))
        self._paged_expert_kv_rows = int(policy_config.get("paged_expert_kv_rows", 8))
        self._runner_reserved_blocks: tuple[int, int] = (0, 0)
        self._paged_expert_kv_fallbacks: set[str] = set()
        self.expert = ExpertModel(
            expert_config,
            dtype=vllm_config.model_config.dtype,
        )
        self._compiled_expert: nn.Module | None = None
        self._action_graphs: dict[tuple[Any, ...], dict[str, Any]] = {}
        # -1 unknown, 0 uniform-decode steps seen only outside a FULL CUDA
        # graph, 1 a FULL-mode uniform-decode forward was observed (capture).
        self._decode_cudagraph_observation: int = -1
        # vLLM's uniform-decode query length: one token per request plus the
        # speculative draft tokens verified alongside it.
        speculative_config = vllm_config.speculative_config
        self._uniform_decode_query_len: int = 1 + (
            int(speculative_config.num_speculative_tokens) if speculative_config is not None else 0
        )
        self._force_future_end_indices: tuple[int, ...] = ()
        self._force_future_start_indices: tuple[int, ...] = ()
        self._mask_text_eos_indices: tuple[int, ...] = ()
        self._pending_policy_groups: dict[str, dict[str, dict[str, Any]]] = {}
        # Navigation CFG rendezvous state: held decode steps per guided group
        # and registered unguided twins keyed by (caller request id, child
        # index). Runner request ids are engine-internal, so twins cannot be
        # matched by the NIM-visible child id they name in ``_nav_cfg_partner``.
        self._policy_group_hold_steps: dict[str, int] = {}
        self._pending_nav_twins: dict[tuple[str, int], dict[str, Any]] = {}
        # Processed prompt length of each twin, measured in-engine on the step
        # its first token is forced (after multimodal placeholder expansion).
        self._nav_twin_prefill_lengths: dict[tuple[str, int], int] = {}
        self._logits_request_count: int | None = None
        # compute_logits index staging (see _stage_logits_indices): row indices
        # travel through a pinned host buffer into a preallocated device buffer
        # instead of per-step pageable ``torch.as_tensor(list, device=cuda)``
        # copies, which block the host until the forward has drained. At most
        # three index lists (EOS mask, forced start, forced end) per step.
        self._logits_index_capacity: int = 3 * max(1, int(vllm_config.scheduler_config.max_num_seqs))
        self._logits_index_host: torch.Tensor | None = None
        self._logits_index_device: torch.Tensor | None = None
        self._logits_index_slot: int = 0
        self._text_eos_token_ids_device: tuple[tuple[int, ...], torch.Tensor] | None = None
        generation_config = vllm_config.model_config.try_get_generation_config()
        text_eos_ids = generation_config.get("eos_token_id")
        if text_eos_ids is None:
            text_eos_ids = getattr(self.alpamayo_config.text_config, "eos_token_id", None)
        if isinstance(text_eos_ids, int):
            text_eos_ids = [text_eos_ids]
        self._text_eos_token_ids = tuple(
            int(token_id) for token_id in (text_eos_ids or []) if int(token_id) != self.future_start_id
        )

    @property
    def future_start_id(self) -> int:
        return int(self.alpamayo_config.traj_ids["future_start"])

    @property
    def speculation_terminal_token_id(self) -> int:
        """Token after which speculative decoding yields to the action step."""
        return self.future_start_id

    @property
    def future_end_id(self) -> int:
        return int(self.alpamayo_config.traj_ids["future_end"])

    def _policy_observations(
        self,
        sampling_extra_args: object,
    ) -> list[Mapping[str, Any] | None]:
        if not isinstance(sampling_extra_args, list):
            return []
        observations: list[Mapping[str, Any] | None] = []
        for extra in sampling_extra_args:
            observation = extra.get("robot_obs") if isinstance(extra, Mapping) else None
            observations.append(observation if isinstance(observation, Mapping) else None)
        return observations

    @staticmethod
    def _request_positions(
        positions: torch.Tensor,
        start: int,
        end: int,
    ) -> torch.Tensor:
        """Select one request's position rows from a flattened runner batch."""
        if positions.ndim == 2 and positions.shape[0] == 3:
            return positions[:, start:end]
        return positions.reshape(-1)[start:end]

    def prepare_runner_inputs(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        req_ids: Sequence[str],
        num_computed_tokens: Sequence[int],
        num_scheduled_tokens: Sequence[int],
        input_ids_buffer: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Keep terminal token IDs visible with multimodal embeddings."""

        del req_ids, num_computed_tokens, num_scheduled_tokens
        if inputs_embeds is not None and input_ids is None and input_ids_buffer is not None:
            input_ids = input_ids_buffer
        return input_ids, positions

    @staticmethod
    def _gather_prefix_cache(
        caches: list[torch.Tensor],
        block_table: torch.Tensor,
        seq_len: int,
    ) -> DynamicCache:
        """Convert one request's paged cache to HF cache layout.

        vLLM has used both a rank-5 layout with an explicit K/V axis and a
        rank-4 layout with K and V concatenated in the final dimension.
        """

        dynamic_cache = DynamicCache()
        for layer_index, cache in enumerate(caches):
            if cache.ndim == 4 and cache.shape[-1] % 2 == 0:
                # FlashAttention: [blocks, kv_heads, block_size, 2 * head_dim].
                block_size = cache.shape[2]
                num_blocks = (seq_len + block_size - 1) // block_size
                block_ids = block_table[:num_blocks].to(dtype=torch.long)
                key_blocks, value_blocks = cache.chunk(2, dim=-1)
                key = key_blocks.index_select(0, block_ids)
                value = value_blocks.index_select(0, block_ids)
                key = key.permute(1, 0, 2, 3).flatten(1, 2)[:, :seq_len]
                value = value.permute(1, 0, 2, 3).flatten(1, 2)[:, :seq_len]
            elif cache.ndim == 5 and (cache.shape[0] == 2 or cache.shape[1] == 2):
                block_size = cache.shape[2]
                num_blocks = (seq_len + block_size - 1) // block_size
                block_ids = block_table[:num_blocks].to(dtype=torch.long)
                if cache.shape[0] == 2:
                    key = cache[0].index_select(0, block_ids).flatten(0, 1)[:seq_len]
                    value = cache[1].index_select(0, block_ids).flatten(0, 1)[:seq_len]
                else:
                    selected = cache.index_select(0, block_ids)
                    key = selected[:, 0].flatten(0, 1)[:seq_len]
                    value = selected[:, 1].flatten(0, 1)[:seq_len]
                key = key.transpose(0, 1)
                value = value.transpose(0, 1)
            else:
                raise RuntimeError(
                    "Alpamayo 2 Super requires a supported vLLM paged KV layout; "
                    f"layer {layer_index} has shape {tuple(cache.shape)}"
                )
            dynamic_cache.update(
                key.unsqueeze(0).contiguous(),
                value.unsqueeze(0).contiguous(),
                layer_index,
            )
        return dynamic_cache

    @staticmethod
    def _repeat_cache(cache: DynamicCache, count: int) -> DynamicCache:
        if count == 1:
            return cache
        # Repeat each layer in place so the one-sample source layer can be
        # released before the next layer is expanded. Building a second cache
        # retained the complete source cache until all repeated layers existed,
        # adding roughly one full dense prefix to the K-sample peak.
        cache.batch_repeat_interleave(count)
        return cache

    @classmethod
    def _gather_prefix_cache_batch(
        cls,
        caches: list[torch.Tensor],
        block_tables: Sequence[torch.Tensor],
        seq_lens: Sequence[int],
    ) -> DynamicCache:
        """Gather distinct paged prefixes directly into one dense batch.

        Gathering every branch into a complete ``DynamicCache`` before
        combining them retains both the six branch caches and the combined
        cache. Constructing the batch one layer at a time keeps only one
        layer's temporary tensors live in addition to the final cache.
        """
        if not block_tables or len(block_tables) != len(seq_lens):
            raise ValueError("Batched prefix block tables and lengths must align")
        max_length = max(int(seq_len) for seq_len in seq_lens)
        combined = DynamicCache()
        for layer_index, paged_layer in enumerate(caches):
            keys: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            for block_table, seq_len in zip(block_tables, seq_lens, strict=True):
                gathered = cls._gather_prefix_cache(
                    [paged_layer],
                    block_table,
                    int(seq_len),
                )
                layer = gathered.layers[0]
                pad_length = max_length - int(seq_len)
                keys.append(torch.nn.functional.pad(layer.keys, (0, 0, 0, pad_length)))
                values.append(torch.nn.functional.pad(layer.values, (0, 0, 0, pad_length)))
            combined.update(
                torch.cat(keys, dim=0),
                torch.cat(values, dim=0),
                layer_index,
            )
        return combined

    @staticmethod
    def _combine_prefix_caches(caches: Sequence[DynamicCache]) -> DynamicCache:
        """Right-pad distinct VLM prefixes into one expert batch."""
        if not caches:
            raise ValueError("At least one VLM prefix is required")
        max_length = max(cache.get_seq_length() for cache in caches)
        combined = DynamicCache()
        for layer_index in range(len(caches[0].layers)):
            keys: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            for cache in caches:
                layer = cache.layers[layer_index]
                pad_length = max_length - layer.keys.shape[-2]
                keys.append(torch.nn.functional.pad(layer.keys, (0, 0, 0, pad_length)))
                values.append(torch.nn.functional.pad(layer.values, (0, 0, 0, pad_length)))
            combined.update(
                torch.cat(keys, dim=0).contiguous(),
                torch.cat(values, dim=0).contiguous(),
                layer_index,
            )
        return combined

    @staticmethod
    def _observation_tensor(value: Any, *, device: torch.device) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=device, dtype=torch.float32)
        while tensor.ndim > 1 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        return tensor

    @staticmethod
    def _action_boundary_position(
        positions: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Return the real terminal position, excluding graph padding."""
        if positions.ndim == 2 and positions.shape[0] == 3:
            return positions[:, :1].to(device)
        return positions.reshape(-1)[:1].repeat(3, 1).to(device)

    @staticmethod
    def _recreate_action_noise(
        extra_args: Sequence[Mapping[str, Any]],
        *,
        action_dims: tuple[int, ...],
        device: torch.device,
    ) -> torch.Tensor:
        """Recreate independently seeded noise after trajectory inference."""
        noise_rows: list[torch.Tensor] = []
        for extra in extra_args:
            sampling_seed = extra.get("_sampling_seed")
            generator = None
            if sampling_seed is not None:
                generator = torch.Generator(device=device)
                generator.manual_seed(int(sampling_seed))
            noise_rows.append(
                torch.randn(
                    1,
                    *action_dims,
                    device=device,
                    dtype=torch.float32,
                    generator=generator,
                )
            )
        return torch.cat(noise_rows, dim=0)

    def _make_static_action_cache(
        self,
        cache: DynamicCache,
        *,
        suffix_length: int,
        max_cache_len: int | None,
    ) -> StaticCache:
        prefix_length = cache.get_seq_length()
        minimum_length = prefix_length + suffix_length
        effective_max_len = max_cache_len or minimum_length
        if effective_max_len < minimum_length:
            raise ValueError(
                f"static action cache is too short: need at least {minimum_length} tokens, got {effective_max_len}"
            )
        static_cache = StaticCache(
            config=self.expert.config.llm_config,
            max_cache_len=effective_max_len,
        )
        cache_position = torch.arange(
            prefix_length,
            device=cache.layers[0].keys.device,
            dtype=torch.long,
        )
        for layer_index, layer in enumerate(cache.layers):
            static_cache.update(
                layer.keys,
                layer.values,
                layer_index,
                cache_kwargs={"cache_position": cache_position},
            )
        return static_cache

    @staticmethod
    def _copy_action_prefix(
        dst: StaticCache,
        src: DynamicCache,
        prefix_length: int,
    ) -> None:
        for dst_layer, src_layer in zip(dst.layers, src.layers, strict=True):
            dst_layer.keys[:, :, :prefix_length].copy_(src_layer.keys)
            dst_layer.values[:, :, :prefix_length].copy_(src_layer.values)
            dst_layer.cumulative_length.fill_(prefix_length)

    @staticmethod
    def _select_paged_blocks(
        cache: torch.Tensor,
        block_ids: torch.Tensor,
        layer_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather one layer's blocks with a single ``index_select``.

        Returns ``(key_blocks, value_blocks)`` as views of the one gathered
        tensor, both shaped ``[blocks, kv_heads, block_size, head_dim]`` for
        every supported paged layout (see ``_gather_prefix_cache``).
        """
        if cache.ndim == 4 and cache.shape[-1] % 2 == 0:
            # FlashAttention: [blocks, kv_heads, block_size, 2 * head_dim].
            key_blocks, value_blocks = cache.index_select(0, block_ids).chunk(2, dim=-1)
            return key_blocks, value_blocks
        if cache.ndim == 5 and cache.shape[0] == 2:
            # [2, blocks, block_size, kv_heads, head_dim].
            selected = cache.index_select(1, block_ids)
            return selected[0].permute(0, 2, 1, 3), selected[1].permute(0, 2, 1, 3)
        if cache.ndim == 5 and cache.shape[1] == 2:
            # [blocks, 2, block_size, kv_heads, head_dim].
            selected = cache.index_select(0, block_ids)
            return selected[:, 0].permute(0, 2, 1, 3), selected[:, 1].permute(0, 2, 1, 3)
        raise RuntimeError(
            "Alpamayo 2 Super requires a supported vLLM paged KV layout; "
            f"layer {layer_index} has shape {tuple(cache.shape)}"
        )

    @staticmethod
    def _copy_block_prefix(
        dst_row: torch.Tensor,
        blocks: torch.Tensor,
        seq_len: int,
        prefix_length: int,
    ) -> None:
        """Write exactly ``seq_len`` gathered positions into one static row.

        ``dst_row`` is one ``[kv_heads, max_cache_len, head_dim]`` row of a
        StaticCache layer and ``blocks`` the ``[blocks, kv_heads, block_size,
        head_dim]`` gather of that row. Whole blocks land in one strided
        ``copy_``, the partial tail block in a second, and the positions up to
        the batch prefix length are zeroed to match right padding. Nothing
        past ``prefix_length`` is touched.
        """
        block_size = blocks.shape[2]
        full_blocks = seq_len // block_size
        full_length = full_blocks * block_size
        if full_blocks:
            dst_row[:, :full_length].unflatten(1, (full_blocks, block_size)).copy_(
                blocks[:full_blocks].permute(1, 0, 2, 3)
            )
        if seq_len > full_length:
            dst_row[:, full_length:seq_len].copy_(blocks[full_blocks, :, : seq_len - full_length])
        if seq_len < prefix_length:
            dst_row[:, seq_len:prefix_length].zero_()

    @classmethod
    def _gather_prefix_cache_into_static(
        cls,
        caches: list[torch.Tensor],
        block_tables: Sequence[torch.Tensor],
        seq_lens: Sequence[int],
        static_cache: StaticCache,
    ) -> None:
        """Gather paged prefixes straight into an existing expert StaticCache.

        This replaces ``_gather_prefix_cache_batch`` followed by
        ``_copy_action_prefix`` once a manual action graph exists: each layer
        is one block ``index_select`` plus a handful of strided copies per
        row, with no dense DynamicCache in between. The written content is
        bitwise identical to the two-step path, including the zero right
        padding of shorter rows, and the cumulative-length cursor is set the
        same way.
        """
        if not block_tables or len(block_tables) != len(seq_lens):
            raise ValueError("Batched prefix block tables and lengths must align")
        if len(static_cache.layers) != len(caches):
            raise ValueError(f"static action cache has {len(static_cache.layers)} layers, expected {len(caches)}")
        lengths = [int(seq_len) for seq_len in seq_lens]
        prefix_length = max(lengths)
        # Block tables are int32 on the device; convert each row once rather
        # than once per layer (dim 2 is block_size in every supported layout).
        block_size = caches[0].shape[2]
        row_block_ids = [
            block_table[: (seq_len + block_size - 1) // block_size].to(dtype=torch.long)
            for block_table, seq_len in zip(block_tables, lengths, strict=True)
        ]
        for layer_index, (paged_layer, static_layer) in enumerate(zip(caches, static_cache.layers, strict=True)):
            dst_keys, dst_values = static_layer.keys, static_layer.values
            if dst_keys.shape[0] != len(block_tables) or dst_keys.shape[2] < prefix_length:
                raise ValueError(
                    f"static action cache layer {layer_index} of shape {tuple(dst_keys.shape)} cannot hold "
                    f"{len(block_tables)} prefixes of up to {prefix_length} tokens"
                )
            for row, (block_ids, seq_len) in enumerate(zip(row_block_ids, lengths, strict=True)):
                key_blocks, value_blocks = cls._select_paged_blocks(paged_layer, block_ids, layer_index)
                cls._copy_block_prefix(dst_keys[row], key_blocks, seq_len, prefix_length)
                cls._copy_block_prefix(dst_values[row], value_blocks, seq_len, prefix_length)
            static_layer.cumulative_length.fill_(prefix_length)

    def _paged_expert_kv_applies(self, caches: list[torch.Tensor], *, row_count: int, suffix_length: int) -> bool:
        """Whether this batch can run in paged mode; otherwise the dense path serves it."""
        key_cache, _ = split_paged_kv_cache(caches[0], 0)
        expert_dtype = self.expert.action_out_proj.weight.dtype
        reason = None
        if key_cache.dtype != expert_dtype:
            reason = f"the engine KV cache is {key_cache.dtype}, the expert {expert_dtype}"
        else:
            try:
                for layer_index, cache in enumerate(caches):
                    paged_kv_flat_rows(cache, layer_index)
            except RuntimeError as exc:
                reason = str(exc)
        if reason is None:
            needed = row_count * self.paged_blocks_per_row(int(key_cache.shape[1]), suffix_length)
            if needed > self._runner_reserved_blocks[1]:
                reason = (
                    f"{row_count} expert rows need {needed} reserved KV pages, the runner reserved "
                    f"{self._runner_reserved_blocks[1]} (paged_expert_kv_rows)"
                )
        if reason is None:
            return True
        if reason not in self._paged_expert_kv_fallbacks:
            self._paged_expert_kv_fallbacks.add(reason)
            logger.warning("paged_expert_kv not applicable, using the dense prefix path: %s", reason)
        return False

    @staticmethod
    def paged_blocks_per_row(block_size: int, suffix_length: int) -> int:
        """Reserved pages one expert row needs: the relocated partial block plus the suffix."""
        return (block_size - 1 + suffix_length + block_size - 1) // block_size

    def runner_kv_cache_reserved_blocks(self, block_size: int) -> int:
        """Pages per layer the runner should allocate beyond the scheduler's pool.

        Called by the model runner before the KV cache is sized. In paged
        expert mode every concurrently served expert row (samples, plus
        navigation twins) owns a few pages for its action tokens.
        """
        if not self._paged_expert_kv_enabled:
            return 0
        suffix_length = int(self.expert.action_space.get_action_space_dims()[0])
        return self._paged_expert_kv_rows * self.paged_blocks_per_row(int(block_size), suffix_length)

    def _paged_prefix_context(
        self,
        caches: list[torch.Tensor],
        block_tables: Sequence[torch.Tensor],
        seq_lens: Sequence[int],
        *,
        nominal_length: int,
        suffix_length: int,
    ) -> PagedPrefixContext:
        """Lay the batch's rows out over the engine's paged cache.

        The prefix part of the page table is padded to the number of blocks
        the configured nominal length needs (a longer prompt widens it and
        creates another graph shape), so captured graphs see one shape per K.
        """
        device = caches[0].device
        key_caches: list[torch.Tensor] = []
        value_caches: list[torch.Tensor] = []
        flat_caches: list[torch.Tensor] = []
        layout: PagedRowLayout | None = None
        for layer_index, cache in enumerate(caches):
            key_cache, value_cache = split_paged_kv_cache(cache, layer_index)
            flat_cache, layer_layout = paged_kv_flat_rows(cache, layer_index)
            if layout is None:
                layout = layer_layout
            elif layer_layout != layout:
                raise RuntimeError("paged_expert_kv needs the same KV layout in every layer")
            key_caches.append(key_cache)
            value_caches.append(value_cache)
            flat_caches.append(flat_cache)
        assert layout is not None
        block_size = int(key_caches[0].shape[1])
        kv_heads = int(key_caches[0].shape[2])
        rows = len(block_tables)
        blocks_per_row = self.paged_blocks_per_row(block_size, suffix_length)
        reserved_start, reserved_count = self._runner_reserved_blocks
        if rows * blocks_per_row > reserved_count:
            raise RuntimeError(
                f"paged_expert_kv: {rows} expert rows need {rows * blocks_per_row} reserved KV pages "
                f"but the runner reserved {reserved_count}; raise paged_expert_kv_rows"
            )
        needed_blocks = max((int(seq_len) + block_size - 1) // block_size for seq_len in seq_lens)
        prefix_width = max((int(nominal_length) + block_size - 1) // block_size, needed_blocks)
        block_table = torch.zeros((rows, prefix_width + blocks_per_row), dtype=torch.int32, device=device)
        seqused = torch.tensor([int(seq_len) + suffix_length for seq_len in seq_lens], dtype=torch.int32, device=device)
        tail_source = torch.zeros(rows, dtype=torch.long, device=device)
        tail_target_host: list[int] = []
        suffix_block_host = torch.empty((rows, suffix_length), dtype=torch.long)
        suffix_slot_host = torch.empty((rows, suffix_length), dtype=torch.long)
        for row, (table, seq_len) in enumerate(zip(block_tables, seq_lens, strict=True)):
            full_blocks, tail = divmod(int(seq_len), block_size)
            reserved = [reserved_start + row * blocks_per_row + offset for offset in range(blocks_per_row)]
            block_table[row, :full_blocks] = table[:full_blocks].to(dtype=torch.int32)
            block_table[row, full_blocks : full_blocks + blocks_per_row] = torch.tensor(
                reserved, dtype=torch.int32, device=device
            )
            tail_target_host.append(reserved[0])
            if tail > 0:
                # Device-to-device: the block id stays on the GPU (no host sync).
                tail_source[row : row + 1].copy_(table[full_blocks : full_blocks + 1].to(dtype=torch.long))
            else:
                tail_source[row] = reserved[0]
            local = torch.arange(tail, tail + suffix_length, dtype=torch.long)
            suffix_block_host[row] = torch.tensor(reserved, dtype=torch.long)[local // block_size]
            suffix_slot_host[row] = local % block_size
        key_rows, value_rows = layout.rows(suffix_block_host, suffix_slot_host, kv_heads)
        return PagedPrefixContext(
            key_caches,
            value_caches,
            flat_caches,
            list(caches),
            layout,
            block_table,
            seqused,
            torch.arange(0, rows * suffix_length + 1, suffix_length, dtype=torch.int32, device=device),
            key_rows.to(device),
            value_rows.to(device),
            tail_source,
            torch.tensor(tail_target_host, dtype=torch.long, device=device),
        )

    @staticmethod
    def _rewind_static_action_cache(
        cache: StaticCache,
        prefix_length: int | torch.Tensor,
    ) -> None:
        # Each expert call overwrites the complete action suffix, so only the
        # write cursor needs to be reset between flow-matching steps.
        for layer in cache.layers:
            if isinstance(prefix_length, torch.Tensor):
                layer.cumulative_length.copy_(prefix_length)
            else:
                layer.cumulative_length.fill_(prefix_length)

    @staticmethod
    def _make_action_attention_mask(
        *,
        sample_count: int,
        prefix_length: int,
        suffix_length: int,
        max_cache_len: int | None,
        device: torch.device,
        dtype: torch.dtype,
        prefix_lengths: Sequence[int] | None = None,
    ) -> torch.Tensor:
        minimum_length = prefix_length + suffix_length
        mask_length = max_cache_len or minimum_length
        if mask_length < minimum_length:
            raise ValueError(
                f"static action cache is too short: need at least {minimum_length} tokens, got {mask_length}"
            )
        attention_mask = torch.full(
            (sample_count, 1, suffix_length, mask_length),
            torch.finfo(dtype).min,
            device=device,
            dtype=dtype,
        )
        if prefix_lengths is None:
            attention_mask[..., :minimum_length] = 0
        else:
            if len(prefix_lengths) != sample_count:
                raise ValueError("Expert prefix lengths must align with the action batch")
            for index, valid_prefix_length in enumerate(prefix_lengths):
                attention_mask[index, ..., :valid_prefix_length] = 0
                attention_mask[
                    index,
                    ...,
                    prefix_length : prefix_length + suffix_length,
                ] = 0
        return attention_mask

    def _run_manual_action_graph(
        self,
        *,
        expert: nn.Module,
        prefix: DynamicCache | None,
        prefix_length: int,
        action: torch.Tensor,
        expert_positions: torch.Tensor,
        attention_mask: torch.Tensor,
        inference_steps: int,
        suffix_length: int,
        static_cache_max_len: int | None,
        guidance_weight: torch.Tensor | None = None,
        paged: PagedPrefixContext | None = None,
    ) -> torch.Tensor:
        """Replay the fixed-shape ten-step expert integration in one graph.

        ``prefix`` is the dense VLM prefix to copy into the graph's StaticCache.
        It is ``None`` when ``_sample_actions_batch`` already gathered the paged
        prefix straight into that cache, which requires the graph state for
        this shape to exist.

        With ``guidance_weight`` (navigation CFG) the cache, positions and mask
        hold two rows per sample, guided first then the unguided twins; every
        step denoises the same action through both rows and blends the two
        velocities as ``(1 - w) * v_unguided + w * v_guided`` inside the
        captured body, with ``w`` a device scalar copied in per replay.
        """

        nav_cfg = guidance_weight is not None
        key = self._action_graph_key(
            action_shape=tuple(action.shape),
            action_dtype=action.dtype,
            device=action.device,
            positions_shape=tuple(expert_positions.shape),
            attention_mask_shape=tuple(attention_mask.shape) if attention_mask is not None else (),
            inference_steps=inference_steps,
            nav_cfg=nav_cfg,
            paged_blocks=int(paged.block_table.shape[1]) if paged is not None else 0,
        )
        state = self._action_graphs.get(key)
        if state is None:
            if paged is not None:
                # The attention wrapper writes the action tokens into the engine's
                # reserved pages; the HF cache object stores nothing.
                static_cache = SuffixPassthroughCache(config=self.expert.config.llm_config, max_cache_len=suffix_length)
                first_position = 0
            else:
                if prefix is None:
                    raise RuntimeError("A dense prefix cache is required to create a new action graph state")
                static_cache = self._make_static_action_cache(
                    prefix,
                    suffix_length=suffix_length,
                    max_cache_len=static_cache_max_len,
                )
                first_position = prefix_length
            state = {
                "cache": static_cache,
                "action": action.clone(),
                "positions": expert_positions.clone(),
                "attention_mask": attention_mask.clone() if attention_mask is not None else None,
                "cache_position": torch.arange(
                    first_position,
                    first_position + suffix_length,
                    device=action.device,
                    dtype=torch.long,
                ),
                "guidance_weight": guidance_weight.clone() if nav_cfg else None,
                "paged": paged.clone_indices() if paged is not None else None,
            }
            sample_count = action.shape[0]

            def graph_body() -> torch.Tensor:
                graph_action = state["action"]
                if state["paged"] is not None:
                    state["paged"].relocate_tails()
                for step in range(inference_steps):
                    timestep = graph_action.new_full(
                        (graph_action.shape[0],) + (1,) * (graph_action.ndim - 1),
                        step / inference_steps,
                    )
                    step_action, step_timestep = graph_action, timestep
                    if nav_cfg:
                        # Guided and unguided rows denoise the same action x.
                        step_action = torch.cat([graph_action, graph_action], dim=0)
                        step_timestep = torch.cat([timestep, timestep], dim=0)
                    with torch.autocast(
                        device_type="cuda",
                        dtype=self.expert.action_out_proj.weight.dtype,
                    ):
                        embeddings = self.expert.action_in_proj(step_action, step_timestep)
                    with paged_prefix_scope(state["paged"]):
                        output = expert(
                            inputs_embeds=embeddings.to(self.expert.action_out_proj.weight.dtype),
                            position_ids=state["positions"],
                            past_key_values=static_cache,
                            attention_mask=state["attention_mask"],
                            cache_position=state["cache_position"],
                            use_cache=True,
                            is_causal=not bool(self.expert.config.expert_non_causal_attention),
                        )
                    self._rewind_static_action_cache(
                        static_cache,
                        state["cache_position"][0],
                    )
                    velocity = self.expert.action_out_proj(output.last_hidden_state[:, -suffix_length:])
                    velocity = velocity.float().view(step_action.shape[0], *graph_action.shape[1:])
                    if nav_cfg:
                        weight = state["guidance_weight"]
                        velocity = (1.0 - weight) * velocity[sample_count:] + weight * velocity[:sample_count]
                    graph_action = graph_action + velocity / inference_steps
                return graph_action

            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                for _ in range(2):
                    graph_body()
                    self._rewind_static_action_cache(
                        static_cache,
                        state["cache_position"][0],
                    )
            torch.cuda.current_stream().wait_stream(warmup_stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_output = graph_body()
            state["graph"] = graph
            state["output"] = graph_output
            self._action_graphs[key] = state

        state["action"].copy_(action)
        state["positions"].copy_(expert_positions)
        if attention_mask is not None:
            state["attention_mask"].copy_(attention_mask)
        if paged is not None:
            state["paged"].copy_indices_from(paged)
        else:
            state["cache_position"].copy_(
                torch.arange(
                    prefix_length,
                    prefix_length + suffix_length,
                    device=action.device,
                    dtype=torch.long,
                )
            )
        if nav_cfg:
            state["guidance_weight"].copy_(guidance_weight)
        if prefix is not None:
            self._copy_action_prefix(state["cache"], prefix, prefix_length)
        state["graph"].replay()
        return state["output"].clone()

    @staticmethod
    def _action_graph_key(
        *,
        action_shape: tuple[int, ...],
        action_dtype: torch.dtype,
        device: torch.device,
        positions_shape: tuple[int, ...],
        attention_mask_shape: tuple[int, ...],
        inference_steps: int,
        nav_cfg: bool = False,
        paged_blocks: int = 0,
    ) -> tuple[Any, ...]:
        """Identify one captured action graph by the shapes it was traced with.

        ``_sample_actions_batch`` derives the same key from the planned shapes
        before any expert tensor exists, so a known shape can gather its paged
        prefix directly into the graph's StaticCache. ``nav_cfg`` separates the
        CFG body (two cache rows per action row, blended velocities) from a
        guided-only body with the same tensor shapes.
        """
        return (
            tuple(action_shape),
            action_dtype,
            device,
            tuple(positions_shape),
            tuple(attention_mask_shape),
            int(inference_steps),
            bool(nav_cfg),
            int(paged_blocks),
        )

    def _sample_actions_batch(
        self,
        *,
        caches: list[torch.Tensor],
        block_tables: Sequence[torch.Tensor],
        seq_lens: Sequence[int],
        positions: Sequence[torch.Tensor],
        observations: Sequence[Mapping[str, Any]],
        extra_args: Sequence[Mapping[str, Any]],
        unguided_block_tables: Sequence[torch.Tensor] | None = None,
        unguided_seq_lens: Sequence[int] | None = None,
        unguided_positions: Sequence[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run flow matching for a batch of guided samples.

        When ``unguided_*`` are given, every guided sample is paired with the
        prefix of its unguided twin and the velocity is combined per step as in
        ``FlowMatching._guided_v`` (``examples/two_gpu_nav_cfg_demo.py``):
        ``v = (1 - w) * v_unguided + w * v_guided``.
        """
        target_cache_layers = int(self.expert.config.llm_config.num_hidden_layers)
        if len(caches) < target_cache_layers:
            raise RuntimeError(
                "Alpamayo action generation received fewer target KV-cache "
                f"layers than expected: got {len(caches)}, expected {target_cache_layers}"
            )
        # vLLM 0.28 appends speculative-draft caches after the target model's
        # layers. The action expert is conditioned only on the target prefix.
        caches = caches[:target_cache_layers]
        device = caches[0].device
        sample_count = len(observations)
        if not (sample_count == len(block_tables) == len(seq_lens) == len(positions) == len(extra_args)):
            raise ValueError("Batched expert inputs must have matching lengths")
        if sample_count < 1:
            raise ValueError("The action expert batch must not be empty")
        first_extra = extra_args[0]
        profile_action = bool(first_extra.get("_profile_action", False)) and device.type == "cuda"
        profile_events: list[torch.cuda.Event] = []

        def record_profile_event() -> None:
            if profile_action:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                profile_events.append(event)

        record_profile_event()
        if any(int(extra.get("num_traj_samples", 1)) != 1 for extra in extra_args):
            raise ValueError("Each parallel VLM branch must request one expert trajectory")
        inference_steps = int(first_extra.get("diffusion_steps", 10))
        temperature = float(first_extra.get("action_temperature", 1.0))
        if inference_steps < 1:
            raise ValueError("num_traj_samples and diffusion_steps must be positive")
        for extra in extra_args[1:]:
            if (
                int(extra.get("diffusion_steps", 10)) != inference_steps
                or float(extra.get("action_temperature", 1.0)) != temperature
            ):
                raise ValueError("Batched expert branches must use identical diffusion settings")
        requested_weights = {extra.get("_nav_guidance_weight") for extra in extra_args}
        if len(requested_weights) != 1:
            raise ValueError("Batched expert branches must share one nav_guidance_weight")
        requested_weight = requested_weights.pop()
        nav_cfg = unguided_block_tables is not None
        if nav_cfg:
            if requested_weight is None:
                raise ValueError("Navigation CFG requires _nav_guidance_weight on every guided branch")
            if (
                unguided_seq_lens is None
                or unguided_positions is None
                or not (len(unguided_block_tables) == len(unguided_seq_lens) == len(unguided_positions) == sample_count)
            ):
                raise ValueError("Navigation CFG needs exactly one unguided prefix per guided sample")
        # The effective weight: 1.0 whenever the expert runs guided-only.
        guidance_weight = float(requested_weight) if nav_cfg else 1.0

        all_block_tables = [*block_tables, *(unguided_block_tables or ())]
        all_seq_lens = [int(seq_len) for seq_len in (*seq_lens, *(unguided_seq_lens or ()))]
        all_positions = [*positions, *(unguided_positions or ())]
        row_count = len(all_block_tables)
        prefix_lengths = all_seq_lens
        prefix_len = max(all_seq_lens)
        if bool(first_extra.get("_batch_action_expert", True)) is False and sample_count > 1:
            raise ValueError("Sequential expert mode must submit one VLM branch per expert call")
        action_dims = tuple(int(dim) for dim in self.expert.action_space.get_action_space_dims())
        suffix_length = action_dims[0]
        static_cache_max_len_value = first_extra.get("_static_expert_cache_max_len")
        static_cache_max_len = int(static_cache_max_len_value) if static_cache_max_len_value is not None else None
        manual_action_cudagraph = bool(first_extra.get("_manual_action_cudagraph", False))
        static_expert_cache = bool(first_extra.get("_static_expert_cache", manual_action_cudagraph))
        # Paged expert KV: the expert attends to the VLM prefix in the engine's
        # paged cache through FA3's paged call and keeps only its own action
        # tokens in a small suffix cache. No prefix gather, no dense copy, no
        # per-row StaticCache; requires the FA3 expert attention backend.
        paged_expert_kv = bool(first_extra.get("_paged_expert_kv", False))
        if paged_expert_kv:
            if self.expert_attention_backend != EXPERT_ATTENTION_BACKEND_FA3:
                raise RuntimeError("paged_expert_kv requires expert_attention_backend=alpamayo_fa3")
            paged_expert_kv = self._paged_expert_kv_applies(caches, row_count=row_count, suffix_length=suffix_length)
        # Navigation CFG doubles the cache rows (guided plus unguided twins) and
        # blends two velocities per step; the captured graph carries that body
        # too (keyed separately), so navigation replays a graph like guided-only
        # requests do. It also keeps the StaticCache: the compiled expert is
        # traced with static shapes, and a DynamicCache sized to each prompt
        # made every new prompt length a recompile (about two minutes per
        # length on H100).
        if static_expert_cache and static_cache_max_len is not None:
            # The configured length is the reusable latency profile, not a
            # hard request limit. Longer prompts remain correct and simply
            # create a second graph shape.
            static_cache_max_len = max(
                static_cache_max_len,
                prefix_len + suffix_length,
            )

        # Materialize the VLM prefix. Once the manual graph for this shape
        # exists, the paged blocks are gathered straight into its StaticCache;
        # the dense DynamicCache is built only for every other expert mode and
        # for the first request of a new graph shape (which creates the state).
        prefix: DynamicCache | None = None
        graph_state: dict[str, Any] | None = None
        paged_context: PagedPrefixContext | None = None
        if paged_expert_kv:
            paged_context = self._paged_prefix_context(
                caches,
                all_block_tables,
                all_seq_lens,
                nominal_length=(static_cache_max_len or (prefix_len + suffix_length)),
                suffix_length=suffix_length,
            )
        elif manual_action_cudagraph and static_expert_cache:
            graph_state = self._action_graphs.get(
                self._action_graph_key(
                    action_shape=(sample_count, *action_dims),
                    action_dtype=torch.float32,
                    device=device,
                    positions_shape=(3, row_count, suffix_length),
                    attention_mask_shape=(
                        row_count,
                        1,
                        suffix_length,
                        static_cache_max_len if static_cache_max_len is not None else prefix_len + suffix_length,
                    ),
                    inference_steps=inference_steps,
                    nav_cfg=nav_cfg,
                )
            )
        if paged_expert_kv:
            pass  # nothing to materialize: the prefix stays in the engine's paged cache
        elif graph_state is not None:
            self._gather_prefix_cache_into_static(caches, all_block_tables, all_seq_lens, graph_state["cache"])
        else:
            prefix = self._gather_prefix_cache_batch(caches, all_block_tables, all_seq_lens)
        record_profile_event()

        # One noise row per trajectory sample; the unguided twin rows share
        # their guided sample's action state.
        noise_rows: list[torch.Tensor] = []
        for extra in extra_args:
            generator = None
            if (sampling_seed := extra.get("_sampling_seed")) is not None:
                generator = torch.Generator(device=device)
                generator.manual_seed(int(sampling_seed))
            noise_rows.append(
                torch.randn(
                    1,
                    *action_dims,
                    device=device,
                    dtype=torch.float32,
                    generator=generator,
                )
            )
        action = torch.cat(noise_rows, dim=0)
        return_initial_noise = any(bool(extra.get("_return_action_noise", False)) for extra in extra_args)
        if return_initial_noise and any(extra.get("_sampling_seed") is None for extra in extra_args):
            raise ValueError("Action-noise diagnostics require a sampling seed")
        action = action * temperature

        last_position = torch.cat(
            [self._action_boundary_position(value, device=device) for value in all_positions],
            dim=1,
        )
        expert_positions = (
            last_position[:, :, None] + 1 + torch.arange(suffix_length, device=device)[None, None, :]
        ).expand(3, row_count, suffix_length)
        weight_dtype = self.expert.action_out_proj.weight.dtype
        attention_mask: torch.Tensor | None = None
        if not paged_expert_kv:
            attention_mask = self._make_action_attention_mask(
                sample_count=row_count,
                prefix_length=prefix_len,
                suffix_length=suffix_length,
                max_cache_len=(static_cache_max_len if static_expert_cache else None),
                device=device,
                dtype=weight_dtype,
                prefix_lengths=prefix_lengths,
            )
        # Every non-graph mode below runs on a dense prefix; the manual-graph
        # path may arrive here with prefix None because it gathered directly
        # into the graph state's StaticCache.
        expert_cache: DynamicCache | StaticCache | None = prefix
        expert_cache_position = None
        if paged_expert_kv and not manual_action_cudagraph:
            expert_cache = SuffixPassthroughCache(config=self.expert.config.llm_config, max_cache_len=suffix_length)
            expert_cache_position = torch.arange(suffix_length, device=device, dtype=torch.long)
        elif static_expert_cache and not manual_action_cudagraph:
            expert_cache = self._make_static_action_cache(
                prefix,
                suffix_length=suffix_length,
                max_cache_len=static_cache_max_len,
            )
            expert_cache_position = torch.arange(
                prefix_len,
                prefix_len + suffix_length,
                device=device,
                dtype=torch.long,
            )
        record_profile_event()

        expert = self.expert.expert
        if bool(first_extra.get("_compile_expert", False)):
            if self._compiled_expert is None:
                self._compiled_expert = compile_action_expert(self.expert.expert)
            expert = self._compiled_expert
        if manual_action_cudagraph:
            action = self._run_manual_action_graph(
                expert=expert,
                prefix=prefix,
                prefix_length=prefix_len,
                action=action,
                expert_positions=expert_positions,
                attention_mask=attention_mask,
                inference_steps=inference_steps,
                suffix_length=suffix_length,
                static_cache_max_len=static_cache_max_len,
                guidance_weight=(
                    torch.tensor(guidance_weight, device=device, dtype=torch.float32) if nav_cfg else None
                ),
                paged=paged_context,
            )
        else:
            assert expert_cache is not None, "eager expert modes require a dense DynamicCache prefix"
            if paged_context is not None:
                paged_context.relocate_tails()
            for step in range(inference_steps):
                timestep = action.new_full(
                    (sample_count,) + (1,) * len(action_dims),
                    step / inference_steps,
                )
                step_action, step_timestep = action, timestep
                if nav_cfg:
                    # Guided and unguided rows denoise the same action x.
                    step_action = torch.cat([action, action], dim=0)
                    step_timestep = torch.cat([timestep, timestep], dim=0)
                with torch.autocast(device_type=device.type, dtype=weight_dtype):
                    embeddings = self.expert.action_in_proj(step_action, step_timestep)
                with paged_prefix_scope(paged_context):
                    outputs = expert(
                        inputs_embeds=embeddings.to(weight_dtype),
                        position_ids=expert_positions,
                        past_key_values=expert_cache,
                        attention_mask=attention_mask,
                        cache_position=expert_cache_position,
                        use_cache=True,
                        is_causal=not bool(self.expert.config.expert_non_causal_attention),
                    )
                if isinstance(expert_cache, StaticCache):
                    self._rewind_static_action_cache(expert_cache, 0 if paged_expert_kv else prefix_len)
                else:
                    expert_cache.crop(prefix_len)
                velocity = self.expert.action_out_proj(outputs.last_hidden_state[:, -suffix_length:])
                velocity = velocity.float().view(row_count, *action_dims)
                if nav_cfg:
                    guided_velocity = velocity[:sample_count]
                    unguided_velocity = velocity[sample_count:]
                    velocity = (1.0 - guidance_weight) * unguided_velocity + guidance_weight * guided_velocity
                action = action + velocity / inference_steps
        record_profile_event()

        history_xyz_rows = [
            self._observation_tensor(observation["ego_history_xyz"], device=device) for observation in observations
        ]
        history_rot_rows = [
            self._observation_tensor(observation["ego_history_rot"], device=device) for observation in observations
        ]
        if any(value.ndim != 2 or value.shape[-1] != 3 for value in history_xyz_rows):
            raise ValueError("Every ego_history_xyz must have shape (T, 3)")
        if any(value.ndim != 3 or value.shape[-2:] != (3, 3) for value in history_rot_rows):
            raise ValueError("Every ego_history_rot must have shape (T, 3, 3)")
        repeated_history_xyz = torch.stack(history_xyz_rows)
        repeated_history_rot = torch.stack(history_rot_rows)
        pred_xyz, pred_rot = self.expert.action_space.action_to_traj(
            action,
            repeated_history_xyz,
            repeated_history_rot,
        )
        record_profile_event()
        result = {
            "pred_trajectories": pred_xyz,
            "pred_rotations": pred_rot,
            "actions": pred_xyz,
            "rotations": pred_rot,
            "normalized_controls": action,
        }
        if profile_action:
            # Synchronize only for explicitly profiled requests. The four
            # intervals are KV materialization, action setup, diffusion, and
            # action-space trajectory conversion respectively.
            profile_events[-1].synchronize()
            result["action_profile_ms"] = torch.tensor(
                [
                    profile_events[index].elapsed_time(profile_events[index + 1])
                    for index in range(len(profile_events) - 1)
                ],
                device=device,
                dtype=torch.float32,
            )
        if return_initial_noise:
            # Recreate the receipt only after trajectory computation. Keeping
            # the original GPU tensor alive changes allocator reuse during the
            # expert pass and can perturb otherwise deterministic trajectories.
            result["action_noise"] = self._recreate_action_noise(
                extra_args,
                action_dims=action_dims,
                device=device,
            )
        # Counts trajectory samples per expert call; CFG doubles the prefix
        # rows without changing the sample count.
        result["action_expert_invocation_batch_size"] = torch.full(
            (sample_count,),
            sample_count,
            device=device,
            dtype=torch.int32,
        )
        # Whether this invocation ran the expert's attention through FA3. The
        # same predicate gates the kernel; several rows (CFG) additionally need
        # the page-aligned StaticCache that the two-segment paged call requires,
        # otherwise the attention wrapper falls back to SDPA and this reports 0.
        multi_row_fa3_layout = (
            paged_expert_kv
            or row_count == 1
            or (
                static_expert_cache
                and static_cache_max_len is not None
                and static_cache_max_len % EXPERT_FA3_PAGE_SIZE == 0
            )
        )
        fa3_supported = expert_fa3_supports(
            batch_rows=row_count,
            dtype=weight_dtype,
            device_type=device.type,
            is_causal=not bool(self.expert.config.expert_non_causal_attention),
        )
        fa3_used = (
            self.expert_attention_backend == EXPERT_ATTENTION_BACKEND_FA3 and multi_row_fa3_layout and fa3_supported
        )
        result["action_expert_attention_fa3"] = torch.full(
            (sample_count,),
            int(fa3_used),
            device=device,
            dtype=torch.int32,
        )
        # Request-level scalars (identical for every sample of one request).
        result["nav_cfg_applied"] = torch.tensor(nav_cfg, device=device)
        result["nav_guidance_weight"] = torch.tensor(guidance_weight, device=device, dtype=torch.float32)
        result["action_expert_attention_backend_configured"] = torch.tensor(
            int(self.expert_attention_backend == EXPERT_ATTENTION_BACKEND_FA3),
            device=device,
            dtype=torch.int32,
        )
        result["decode_full_cudagraph_observed"] = torch.tensor(
            self._decode_cudagraph_observation,
            device=device,
            dtype=torch.int32,
        )
        return result

    def _sample_actions(
        self,
        *,
        caches: list[torch.Tensor],
        block_table: torch.Tensor,
        seq_len: int,
        positions: torch.Tensor,
        observation: Mapping[str, Any],
        extra_args: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Compatibility wrapper for one sequential expert invocation."""
        return self._sample_actions_batch(
            caches=caches,
            block_tables=[block_table],
            seq_lens=[seq_len],
            positions=[positions],
            observations=[observation],
            extra_args=[extra_args],
        )

    @staticmethod
    def _parallel_group_info(
        request_id: str,
        extra_args: Mapping[str, Any],
    ) -> tuple[str, int, int]:
        expected = int(extra_args.get("_parallel_sample_count", 1))
        if expected <= 1:
            return request_id, 0, 1
        child_index, separator, parent_id = request_id.partition("_")
        if not separator or not child_index.isdigit() or not parent_id:
            raise RuntimeError("Parallel Alpamayo request IDs must use vLLM's '<index>_<parent>' form")
        index = int(child_index)
        if index >= expected:
            raise RuntimeError("Parallel Alpamayo child index exceeds its sample count")
        return parent_id, index, expected

    @staticmethod
    def _split_batched_policy_output(
        output: Mapping[str, torch.Tensor],
        index: int,
    ) -> dict[str, torch.Tensor]:
        return {
            key: value[index : index + 1] if key in _BATCHED_POLICY_OUTPUT_KEYS else value
            for key, value in output.items()
        }

    @staticmethod
    def _concat_batched_policy_outputs(
        outputs: Sequence[Mapping[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Join independently evaluated expert microbatches in sample order."""
        if not outputs:
            raise ValueError("At least one action-expert output is required")
        keys = set(outputs[0])
        if any(set(output) != keys for output in outputs[1:]):
            raise RuntimeError("Action-expert microbatches returned different fields")
        combined: dict[str, torch.Tensor] = {}
        for key in keys:
            values = [output[key] for output in outputs]
            if key in _BATCHED_POLICY_OUTPUT_KEYS:
                combined[key] = torch.cat(values, dim=0)
            elif key == "action_profile_ms":
                combined[key] = torch.stack(values).sum(dim=0)
            else:
                combined[key] = values[0]
        return combined

    def _attn_metadata_is_uniform_decode(self, attn_metadata: Any) -> bool:
        """Return whether the forward context's attention metadata is decode-shaped.

        vLLM sizes every uniform-decode step, captured or real, with
        ``max_query_len`` equal to the uniform-decode query length. The metadata
        maps layer names to per-layer metadata (a list of such maps under
        dual-batch overlap); only host-side ints are read, never device tensors.
        """
        groups = attn_metadata if isinstance(attn_metadata, list | tuple) else (attn_metadata,)
        for group in groups:
            if not isinstance(group, Mapping):
                continue
            for metadata in group.values():
                max_query_len = getattr(metadata, "max_query_len", None)
                if isinstance(max_query_len, int) and max_query_len == self._uniform_decode_query_len:
                    return True
        return False

    def _observe_decode_cudagraph_mode(self) -> None:
        """Record whether uniform-decode steps run inside a FULL CUDA graph.

        vLLM executes this Python forward during graph capture with the runtime
        mode ``FULL`` and a ``uniform`` batch descriptor; graph replays skip
        Python entirely. Piecewise or eager decode runs Python every step under
        ``PIECEWISE``/``NONE`` with the descriptor relaxed to non-uniform, so a
        step also counts as decode-shaped when its attention metadata carries
        the uniform-decode query length. Observing ``FULL`` once proves full
        decode graphs were captured and is sticky. This is diagnostics only and
        never raises.
        """
        if get_forward_context is None or self._decode_cudagraph_observation == 1:
            return
        try:
            context = get_forward_context()
            mode = getattr(context, "cudagraph_runtime_mode", None)
            if mode is None:
                return
            descriptor = getattr(context, "batch_descriptor", None)
            uniform = getattr(descriptor, "uniform", None)
            if uniform is None:
                uniform = getattr(descriptor, "uniform_decode", False)
            if not bool(uniform) and not self._attn_metadata_is_uniform_decode(getattr(context, "attn_metadata", None)):
                return
            mode_name = getattr(mode, "name", str(mode))
            if mode_name == "FULL":
                self._decode_cudagraph_observation = 1
            elif mode_name in ("PIECEWISE", "NONE"):
                self._decode_cudagraph_observation = 0
        except Exception:  # Diagnostics must never break a decode step.
            return

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors | OmniOutput:
        kwargs.pop("sampling_extra_args", None)
        kwargs.pop("runner_kv_cache_context", None)
        self._observe_decode_cudagraph_mode()
        return super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    @staticmethod
    def _microbatch_children(
        ordered: Sequence[tuple[str, dict[str, Any]]],
        max_batch_size: int,
        *,
        rows_per_sample: int,
    ) -> list[Sequence[tuple[str, dict[str, Any]]]]:
        """Chunk guided children so each expert call stays within the row budget.

        ``_action_expert_max_batch_size`` counts expert prefix rows. A guided
        sample and its unguided CFG twin always share one call, so with CFG
        each pair costs two rows; a single pair is never split.
        """
        per_chunk = max(1, max_batch_size // rows_per_sample)
        return [ordered[begin : begin + per_chunk] for begin in range(0, len(ordered), per_chunk)]

    @staticmethod
    def _nav_twin_key(extra: Mapping[str, Any]) -> tuple[str, int]:
        """Return the (caller request id, child index) a twin pairs with."""
        group = extra.get("_nav_cfg_group")
        partner_index = extra.get("_nav_cfg_partner_index")
        if not group or partner_index is None:
            raise RuntimeError("Unguided CFG twins must carry _nav_cfg_group and _nav_cfg_partner_index")
        return str(group), int(partner_index)

    def make_omni_output(
        self,
        hidden_states: torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]],
        *,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor,
        sampling_extra_args: object = None,
        runner_kv_cache_context: RunnerKVCacheContext | None = None,
        request_token_spans: Sequence[tuple[int, int]] | None = None,
        request_first_token_ids: Sequence[int] | None = None,
        **_: Any,
    ) -> IntermediateTensors | OmniOutput:
        """Detect the reasoning boundary and run the action expert.

        ``request_first_token_ids`` carries each request's first scheduled
        token for this step as host ints (the runner reads them from its CPU
        token table under synchronous scheduling). When given, the
        ``future_start`` trigger is decided without touching ``input_ids``, so
        the hook no longer blocks on the forward every decode step. When
        ``None`` (async scheduling placeholders, older runners) the trigger
        falls back to reading the first token of each span from ``input_ids``.
        """
        if isinstance(hidden_states, IntermediateTensors):
            return hidden_states
        observations = self._policy_observations(sampling_extra_args)
        # The released OSS model does not allow an ordinary text EOS to end
        # trajectory reasoning. Generation must continue until future_start,
        # which transfers control to the action expert. Keep this per request
        # so text-only tasks retain their normal EOS behavior in mixed batches.
        self._mask_text_eos_indices = tuple(
            index for index, observation in enumerate(observations) if observation is not None
        )
        self._logits_request_count = len(observations) if observations else None
        multimodal_outputs: dict[str, list[torch.Tensor | None]] = {}
        triggered_indices: set[int] = set()
        if observations and input_ids is not None:
            request_count = len(observations)
            if request_token_spans is None:
                if request_count != 1:
                    raise RuntimeError("Super batched policy inference requires request_token_spans")
                request_token_spans = [(0, input_ids.numel())]
            if len(request_token_spans) != request_count:
                raise RuntimeError("Super request token spans and runner requests must align")
            if request_first_token_ids is not None and len(request_first_token_ids) != request_count:
                raise RuntimeError("Super request first token ids and runner requests must align")

            # Host path: the runner already knows each span's first token.
            # Fallback: a device->host read per request that waits for the
            # forward to finish.
            flat_input_ids = input_ids.reshape(-1) if request_first_token_ids is None else None
            extras = sampling_extra_args if isinstance(sampling_extra_args, list) else []
            trigger_mask = [
                observation is not None
                and end > start
                and (
                    int(request_first_token_ids[index])
                    if request_first_token_ids is not None
                    else int(flat_input_ids[start])
                )
                == self.future_start_id
                for index, (observation, (start, end)) in enumerate(zip(observations, request_token_spans, strict=True))
            ]
            if any(trigger_mask):
                if runner_kv_cache_context is None:
                    raise RuntimeError("Super action generation requires runner KV-cache access")
                if len(runner_kv_cache_context.request_ids) != request_count:
                    raise RuntimeError("Super policy arguments and runner requests must align")
                start, count = getattr(runner_kv_cache_context, "reserved_blocks", (0, 0))
                self._runner_reserved_blocks = (int(start), int(count))
            per_request_outputs: list[dict[str, torch.Tensor] | None] = [None] * request_count
            touched_groups: set[str] = set()
            touched_twins: set[str] = set()
            forced_first_token_indices: set[int] = set()
            for index, (observation, span) in enumerate(zip(observations, request_token_spans, strict=True)):
                start, end = span
                extra = extras[index] if index < len(extras) else None
                if (
                    observation is not None
                    and isinstance(extra, Mapping)
                    and extra.get("_force_first_token") is not None
                    and not trigger_mask[index]
                ):
                    # An unguided CFG twin that has not produced its first
                    # token yet (its prompt is still being prefilled). Force
                    # future_start so its next span triggers the rendezvous.
                    if int(extra["_force_first_token"]) != self.future_start_id:
                        raise ValueError("_force_first_token must be the future_start token")
                    forced_first_token_indices.add(index)
                    if extra.get("_nav_cfg_role") == "unguided" and runner_kv_cache_context is not None:
                        # Full processed prompt length; the last chunk of a
                        # chunked prefill overwrites earlier partial values.
                        self._nav_twin_prefill_lengths[self._nav_twin_key(extra)] = int(
                            runner_kv_cache_context.sequence_lengths[index]
                        )
                # A speculative verification block can contain an unaccepted
                # future_start after its first token. The accepted boundary is
                # fed back as the first token of a draft-free request span.
                if not trigger_mask[index]:
                    continue
                assert observation is not None
                assert runner_kv_cache_context is not None
                if not isinstance(extra, Mapping):
                    raise RuntimeError("Super policy sampling arguments must be mappings")
                request_id = runner_kv_cache_context.request_ids[index]
                if extra.get("_nav_cfg_role") == "unguided":
                    # Twins never run the expert themselves; they only lend
                    # their unguided prefix to the guided partner's group.
                    partner = self._nav_twin_key(extra)
                    seq_len = int(runner_kv_cache_context.sequence_lengths[index])
                    twin = self._pending_nav_twins.get(partner)
                    if twin is None:
                        prefill_len = self._nav_twin_prefill_lengths.pop(partner, None)
                        if prefill_len is None or seq_len != prefill_len + 1:
                            # future_start was not the first token produced
                            # after the prompt (the forcing step was skipped,
                            # see compute_logits), so the prefix no longer
                            # ends at the guided reasoning boundary.
                            logger.warning(
                                "Unguided CFG twin %s reached future_start at length %d, expected %s; "
                                "its guided partner %s will run without navigation guidance",
                                request_id,
                                seq_len,
                                "unknown" if prefill_len is None else prefill_len + 1,
                                extra.get("_nav_cfg_partner", partner),
                            )
                            self._pending_nav_twins[partner] = {"request_id": request_id, "invalid": True}
                            triggered_indices.add(index)
                            touched_twins.add(partner)
                            continue
                        self._pending_nav_twins[partner] = {
                            "request_id": request_id,
                            "block_table": runner_kv_cache_context.block_table[index].clone(),
                            "seq_len": seq_len,
                            "positions": self._request_positions(positions, start, end).clone(),
                        }
                    elif twin["request_id"] != request_id:
                        raise RuntimeError(f"Guided child {partner} has more than one unguided CFG twin")
                    touched_twins.add(partner)
                    continue
                group_id, child_index, expected = self._parallel_group_info(
                    request_id,
                    extra,
                )
                needs_twin = extra.get("_nav_guidance_weight") is not None
                if not needs_twin and (expected == 1 or not bool(extra.get("_batch_action_expert", True))):
                    per_request_outputs[index] = self._sample_actions(
                        caches=runner_kv_cache_context.caches,
                        block_table=runner_kv_cache_context.block_table[index],
                        seq_len=runner_kv_cache_context.sequence_lengths[index],
                        positions=self._request_positions(positions, start, end),
                        observation=observation,
                        extra_args=extra,
                    )
                    triggered_indices.add(index)
                    continue

                group = self._pending_policy_groups.setdefault(group_id, {})
                group.setdefault(
                    request_id,
                    {
                        "child_index": child_index,
                        "expected": expected,
                        "block_table": runner_kv_cache_context.block_table[index].clone(),
                        "seq_len": runner_kv_cache_context.sequence_lengths[index],
                        "positions": self._request_positions(positions, start, end).clone(),
                        "observation": observation,
                        "extra_args": extra,
                        "needs_twin": needs_twin,
                        # (caller request id, child index): the key its twin uses.
                        "nav_key": (str(extra.get("_nav_cfg_group") or group_id), child_index),
                    },
                )
                touched_groups.add(group_id)

            request_index_by_id = (
                {request_id: index for index, request_id in enumerate(runner_kv_cache_context.request_ids)}
                if runner_kv_cache_context is not None
                else {}
            )
            waiting_indices: set[int] = set()
            for group_id in touched_groups:
                group = self._pending_policy_groups[group_id]
                expected_values = {int(entry["expected"]) for entry in group.values()}
                if len(expected_values) != 1:
                    raise RuntimeError("Parallel Alpamayo children disagree on sample count")
                expected = expected_values.pop()
                all_members_scheduled = all(request_id in request_index_by_id for request_id in group)
                if len(group) < expected or not all_members_scheduled:
                    waiting_indices.update(
                        request_index_by_id[request_id] for request_id in group if request_id in request_index_by_id
                    )
                    continue

                # Navigation CFG: every guided child that requested guidance
                # must also have its unguided twin scheduled in this batch.
                twin_partners = [entry["nav_key"] for entry in group.values() if entry["needs_twin"]]
                twins = {nav_key: self._pending_nav_twins.get(nav_key) for nav_key in twin_partners}
                apply_nav_cfg = bool(twin_partners)
                if twin_partners:
                    if any(twin is not None and twin.get("invalid") for twin in twins.values()):
                        apply_nav_cfg = False
                    elif not all(
                        twin is not None and twin["request_id"] in request_index_by_id for twin in twins.values()
                    ):
                        hold_steps = self._policy_group_hold_steps.get(group_id, 0) + 1
                        self._policy_group_hold_steps[group_id] = hold_steps
                        max_hold_steps = max(
                            int(entry["extra_args"].get("_nav_cfg_max_hold_steps", _NAV_CFG_DEFAULT_MAX_HOLD_STEPS))
                            for entry in group.values()
                        )
                        if hold_steps < max_hold_steps:
                            waiting_indices.update(request_index_by_id[request_id] for request_id in group)
                            waiting_indices.update(
                                request_index_by_id[twin["request_id"]]
                                for twin in twins.values()
                                if twin is not None and twin["request_id"] in request_index_by_id
                            )
                            continue
                        logger.warning(
                            "Navigation CFG twin missing for %s after %d held decode steps; "
                            "running the action expert guided-only",
                            group_id,
                            hold_steps,
                        )
                        apply_nav_cfg = False

                ordered = sorted(group.items(), key=lambda item: int(item[1]["child_index"]))
                configured_batch_sizes = {
                    int(entry["extra_args"].get("_action_expert_max_batch_size", expected)) for _, entry in ordered
                }
                if len(configured_batch_sizes) != 1:
                    raise RuntimeError("Parallel Alpamayo children disagree on expert batch size")
                max_batch_size = configured_batch_sizes.pop()
                if max_batch_size < 1:
                    raise ValueError("action_expert_max_batch_size must be positive")
                if not all(bool(entry["extra_args"].get("_batch_action_expert", True)) for _, entry in ordered):
                    max_batch_size = 1
                microbatch_outputs = []
                for chunk in self._microbatch_children(
                    ordered,
                    max_batch_size,
                    rows_per_sample=2 if apply_nav_cfg else 1,
                ):
                    batch_kwargs: dict[str, Any] = {
                        "caches": runner_kv_cache_context.caches,
                        "block_tables": [entry["block_table"] for _, entry in chunk],
                        "seq_lens": [int(entry["seq_len"]) for _, entry in chunk],
                        "positions": [entry["positions"] for _, entry in chunk],
                        "observations": [entry["observation"] for _, entry in chunk],
                        "extra_args": [entry["extra_args"] for _, entry in chunk],
                    }
                    if apply_nav_cfg:
                        chunk_twins = [twins[entry["nav_key"]] for _, entry in chunk]
                        batch_kwargs.update(
                            unguided_block_tables=[twin["block_table"] for twin in chunk_twins],
                            unguided_seq_lens=[int(twin["seq_len"]) for twin in chunk_twins],
                            unguided_positions=[twin["positions"] for twin in chunk_twins],
                        )
                    microbatch_outputs.append(self._sample_actions_batch(**batch_kwargs))
                batched_output = self._concat_batched_policy_outputs(microbatch_outputs)
                for batch_index, (request_id, _) in enumerate(ordered):
                    request_index = request_index_by_id[request_id]
                    per_request_outputs[request_index] = self._split_batched_policy_output(
                        batched_output,
                        batch_index,
                    )
                    triggered_indices.add(request_index)
                for nav_key in twin_partners:
                    # Twins end with the group (forced future_end, empty payload).
                    self._nav_twin_prefill_lengths.pop(nav_key, None)
                    twin = self._pending_nav_twins.pop(nav_key, None)
                    if twin is not None and twin["request_id"] in request_index_by_id:
                        triggered_indices.add(request_index_by_id[twin["request_id"]])
                if apply_nav_cfg:
                    logger.info(
                        "Navigation CFG applied for group %s after %d held decode steps (K=%d, weight=%.2f)",
                        group_id,
                        self._policy_group_hold_steps.get(group_id, 0),
                        expected,
                        float(ordered[0][1]["extra_args"]["_nav_guidance_weight"]),
                    )
                del self._pending_policy_groups[group_id]
                self._policy_group_hold_steps.pop(group_id, None)

            for partner in touched_twins:
                twin = self._pending_nav_twins.get(partner)
                if twin is None:
                    continue
                partner_pending = any(
                    entry.get("nav_key") == partner
                    for group in self._pending_policy_groups.values()
                    for entry in group.values()
                )
                if twin.get("invalid"):
                    # Already ended; keep the marker only while its group can
                    # still consume it.
                    if not partner_pending:
                        del self._pending_nav_twins[partner]
                    continue
                twin_index = request_index_by_id.get(twin["request_id"])
                if twin_index is None or twin_index in triggered_indices:
                    continue
                if partner_pending:
                    # The guided group is still assembling; hold the twin too.
                    waiting_indices.add(twin_index)
                    continue
                # The partner already finished (timeout fallback or abort); a
                # twin must never run the expert or wait on its own.
                logger.warning(
                    "Unguided CFG twin %s has no pending guided partner %s; ending it without an expert run",
                    twin["request_id"],
                    partner,
                )
                del self._pending_nav_twins[partner]
                triggered_indices.add(twin_index)

            output_keys = {key for output in per_request_outputs if output is not None for key in output}
            multimodal_outputs = {
                key: [output.get(key) if output is not None else None for output in per_request_outputs]
                for key in output_keys
            }
            self._force_future_end_indices = tuple(sorted(triggered_indices))
            self._force_future_start_indices = tuple(sorted(waiting_indices | forced_first_token_indices))
        if isinstance(hidden_states, list | tuple):
            text_hidden_states = hidden_states[0]
            aux_hidden_states = hidden_states[1]
        else:
            text_hidden_states = hidden_states
            aux_hidden_states = None
        return OmniOutput(
            text_hidden_states=text_hidden_states,
            multimodal_outputs=multimodal_outputs,
            aux_hidden_states=aux_hidden_states,
        )

    def _text_eos_token_ids_on(self, device: torch.device) -> torch.Tensor:
        """Return the masked text EOS ids as a device tensor, built once per device."""
        cached = self._text_eos_token_ids_device
        if cached is None or cached[0] != self._text_eos_token_ids or cached[1].device != device:
            cached = (
                self._text_eos_token_ids,
                torch.tensor(self._text_eos_token_ids, dtype=torch.long, device=device),
            )
            self._text_eos_token_ids_device = cached
        return cached[1]

    def _stage_logits_indices(
        self,
        device: torch.device,
        index_lists: Sequence[Sequence[int]],
    ) -> list[torch.Tensor]:
        """Upload logits row-index lists to ``device`` with one asynchronous copy.

        The lists are written into a pinned host buffer and copied into a
        preallocated device buffer with ``non_blocking=True``; the returned
        views alias that buffer and are valid until the next call. Two host
        slots alternate so a step never overwrites the staging area of the
        copy issued by the previous step, the same reuse discipline vLLM
        applies to its own per-step input buffers. Buffers grow (rarely, and
        only on the host-visible allocation path) when a step needs more rows
        than the scheduler's request budget.
        """
        total = sum(len(indices) for indices in index_lists)
        host = self._logits_index_host
        device_buffer = self._logits_index_device
        if host is None or device_buffer is None or host.shape[1] < total or device_buffer.device != device:
            capacity = max(self._logits_index_capacity, total)
            host = torch.empty(2, capacity, dtype=torch.long, pin_memory=device.type == "cuda")
            device_buffer = torch.empty(capacity, dtype=torch.long, device=device)
            self._logits_index_host = host
            self._logits_index_device = device_buffer
        slot = self._logits_index_slot
        self._logits_index_slot = 1 - slot
        staged = host[slot]
        staged_np = staged.numpy()
        views: list[torch.Tensor] = []
        offset = 0
        for indices in index_lists:
            count = len(indices)
            staged_np[offset : offset + count] = indices
            views.append(device_buffer[offset : offset + count])
            offset += count
        device_buffer[:total].copy_(staged[:total], non_blocking=True)
        return views

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput) -> torch.Tensor:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        logits = super().compute_logits(hidden_states)
        force_end_indices = self._force_future_end_indices
        force_start_indices = self._force_future_start_indices
        mask_text_eos_indices = self._mask_text_eos_indices
        request_count = self._logits_request_count
        self._force_future_end_indices = ()
        self._force_future_start_indices = ()
        self._mask_text_eos_indices = ()
        self._logits_request_count = None
        traj_ids = self.alpamayo_config.traj_ids
        start = min(int(traj_ids["history_id0"]), int(traj_ids["future_id0"]))
        logits[..., start : start + int(self.alpamayo_config.traj_vocab_size)] = -torch.inf
        if not (mask_text_eos_indices and self._text_eos_token_ids):
            mask_text_eos_indices = ()
        if (
            (force_start_indices or force_end_indices)
            and request_count is not None
            and int(logits.shape[0]) != request_count
        ):
            # Speculative verification steps carry one logits row per draft
            # position, so request indices do not address rows. Model-owned
            # boundaries are only forced on draft-free steps (a request that
            # sampled future_start disables drafting for the next step); never
            # force a token onto another request's row.
            logger.warning(
                "Skipping forced trajectory boundaries: %d logits rows for %d requests",
                int(logits.shape[0]),
                request_count,
            )
            force_start_indices = ()
            force_end_indices = ()
        if not (mask_text_eos_indices or force_start_indices or force_end_indices):
            return logits
        # One host->device copy per step for every index list; the row views
        # index the logits directly so no host synchronization happens here.
        mask_rows, start_rows, end_rows = self._stage_logits_indices(
            logits.device,
            (mask_text_eos_indices, force_start_indices, force_end_indices),
        )
        if mask_text_eos_indices:
            logits[mask_rows[:, None], self._text_eos_token_ids_on(logits.device)] = -torch.inf
        for rows, token_id in (
            (start_rows, self.future_start_id),
            (end_rows, self.future_end_id),
        ):
            if rows.numel() == 0:
                continue
            logits[rows] = -torch.inf
            logits[rows, token_id] = 0
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        vlm_weights: list[tuple[str, torch.Tensor]] = []
        local_weights: list[tuple[str, torch.Tensor]] = []
        for name, weight in weights:
            if name.startswith("vlm."):
                vlm_weights.append((name.removeprefix("vlm."), weight))
            elif name.startswith("expert."):
                local_weights.append((name, weight))
        loaded = set(super().load_weights(vlm_weights))
        parameters = dict(self.named_parameters())
        for name, weight in local_weights:
            parameter = parameters.get(name)
            if parameter is None:
                continue
            loader = getattr(parameter, "weight_loader", default_weight_loader)
            loader(parameter, weight)
            loaded.add(name)
        return loaded
