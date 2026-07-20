# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Materialization of Alpamayo's tokenizer extension."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from transformers import AutoTokenizer

from .processing import extend_tokenizer


def ensure_extended_tokenizer(
    backbone: str = "nvidia/Cosmos-Reason2-8B",
    *,
    cache_root: str | os.PathLike[str] | None = None,
) -> str:
    """Return a persistent local tokenizer directory usable by vLLM.

    The Alpamayo checkpoint contains only model weights and configuration.
    Its official loader extends the backbone tokenizer in memory, whereas
    vLLM accepts a tokenizer name/path at engine construction time.  Saving an
    idempotently extended copy bridges those two loading conventions.
    """

    root = Path(cache_root or os.getenv("VLLM_OMNI_CACHE", Path.home() / ".cache" / "vllm-omni"))
    digest = hashlib.sha256(backbone.encode()).hexdigest()[:12]
    target = root / "tokenizers" / f"alpamayo1_5-{digest}"
    complete = target / ".complete"
    if complete.is_file():
        return str(target)

    target.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    extend_tokenizer(tokenizer)
    if len(tokenizer) != 155697:
        raise RuntimeError(f"Alpamayo tokenizer must contain 155697 tokens, got {len(tokenizer)}")
    tokenizer.save_pretrained(target)
    complete.touch()
    return str(target)
