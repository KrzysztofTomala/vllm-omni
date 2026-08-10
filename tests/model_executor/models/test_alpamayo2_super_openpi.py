# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest

from vllm_omni.model_executor.models.alpamayo2_super.openpi import (
    Alpamayo2SuperOpenPIRequestAdapter,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_super_vision_cache_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("NIM_ALPAMAYO_VISION_EMBED_CACHE", raising=False)
    adapter = object.__new__(Alpamayo2SuperOpenPIRequestAdapter)

    assert adapter._image_uuids(
        {"image_uuids": ["content-a", "content-b"]},
        request_id="request-7",
        image_count=2,
    ) == ["request-7:image:0", "request-7:image:1"]


def test_super_vision_cache_uses_content_uuids_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("NIM_ALPAMAYO_VISION_EMBED_CACHE", "1")
    adapter = object.__new__(Alpamayo2SuperOpenPIRequestAdapter)

    assert adapter._image_uuids(
        {"image_uuids": ["content-a", "content-b"]},
        request_id="request-7",
        image_count=2,
    ) == ["content-a", "content-b"]


def test_super_vision_cache_requires_one_uuid_per_image(monkeypatch) -> None:
    monkeypatch.setenv("NIM_ALPAMAYO_VISION_EMBED_CACHE", "true")
    adapter = object.__new__(Alpamayo2SuperOpenPIRequestAdapter)

    with pytest.raises(ValueError, match="one content UUID per image"):
        adapter._image_uuids(
            {"image_uuids": ["content-a"]},
            request_id="request-7",
            image_count=2,
        )
