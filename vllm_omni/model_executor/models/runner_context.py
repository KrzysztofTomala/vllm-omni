# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Explicit model-runner context exposed to opt-in Omni models."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RunnerKVCacheContext:
    """Read-only runner views needed by a model-side terminal output hook.

    Models opting into ``needs_runner_kv_cache`` receive this context from
    ``make_omni_output``. The tensors remain owned by the runner and must not
    be mutated or retained after the hook returns.
    """

    caches: list[torch.Tensor]
    block_table: torch.Tensor
    sequence_lengths: tuple[int, ...]
    request_ids: tuple[str, ...]
    # Blocks (first id, count) the runner allocated in every cache tensor
    # beyond the scheduler's pool, for a model that asked for them through
    # ``runner_kv_cache_reserved_blocks``; (0, 0) when none.
    reserved_blocks: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        if self.block_table.ndim != 2:
            raise ValueError("runner KV block_table must be rank 2")
        if len(self.sequence_lengths) != len(self.request_ids):
            raise ValueError("runner KV sequence lengths and request IDs must align")
        if self.block_table.shape[0] < len(self.request_ids):
            raise ValueError("runner KV block_table has fewer rows than requests")
