# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage input processor for Alpamayo 2 Super."""

from __future__ import annotations

from typing import Any


def reasoning2expert(source_outputs: list[Any], prompt: Any, requires_multimodal_data: bool) -> dict[str, Any]:
    """Build the expert stage's prompt.

    The expert reads only the transferred KV prefix and its own sampling
    params, so the camera frames are not forwarded.
    """
    del source_outputs, prompt, requires_multimodal_data
    return {"prompt": ""}
