# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Model-owned request adapters for OpenPI policy serving."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class OpenPIEngineRequest:
    """The generic request shape consumed by ``AsyncOmni.generate``."""

    prompt: Any
    sampling_params: Any
    request_id: str


class OpenPIRequestAdapter(Protocol):
    """Convert a model-specific OpenPI observation into an engine request."""

    def build_request(
        self,
        observation: dict[str, Any],
        *,
        request_id: str,
        session_id: str,
        reset: bool,
    ) -> OpenPIEngineRequest: ...


def load_openpi_request_adapter(
    adapter_path: str | None,
    policy_config: dict[str, Any],
) -> OpenPIRequestAdapter | None:
    """Load a trusted model adapter declared by ``policy_server_config``.

    The declaration uses ``module.path:ClassName`` syntax. Keeping the class
    path in model configuration lets the generic OpenPI serving layer support
    autoregressive and diffusion policies without importing model packages.
    """

    if not adapter_path:
        return None
    module_name, separator, class_name = adapter_path.partition(":")
    if not separator or not module_name or not class_name:
        raise ValueError(
            "policy_server_config.request_adapter must use "
            "'module.path:ClassName' syntax"
        )
    module = importlib.import_module(module_name)
    adapter_class = getattr(module, class_name, None)
    if adapter_class is None:
        raise ValueError(f"OpenPI request adapter {adapter_path!r} was not found")
    adapter = adapter_class(policy_config)
    if not callable(getattr(adapter, "build_request", None)):
        raise TypeError(f"OpenPI request adapter {adapter_path!r} has no build_request method")
    return adapter
