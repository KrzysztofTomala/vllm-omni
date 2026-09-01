# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Serving layer for robot policy inference via `/v1/realtime/robot/openpi`.

Flow: raw obs → engine request → actions.
The loaded policy model owns dataset transforms inside its pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import count
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from vllm.logger import init_logger

from vllm_omni.entrypoints.openpi.request_adapters import (
    OpenPIEngineRequest,
    load_openpi_request_adapter,
)

logger = init_logger(__name__)

ActionOutput = np.ndarray | dict[str, np.ndarray]
PolicyOutput = ActionOutput | dict[str, Any]

_PARALLEL_POLICY_BATCH_KEYS = frozenset(
    {
        "pred_trajectories",
        "pred_rotations",
        "actions",
        "rotations",
        "normalized_controls",
        "action_noise",
        "action_expert_invocation_batch_size",
    }
)


def _completion_multimodal_outputs(result: Any) -> list[Mapping[str, Any]]:
    completions = getattr(result, "outputs", None) or []
    outputs = [
        output
        for completion in completions
        if isinstance((output := getattr(completion, "multimodal_output", None)), Mapping)
    ]
    if outputs:
        return outputs
    output = getattr(result, "multimodal_output", None)
    return [output] if isinstance(output, Mapping) else []


def _concatenate_policy_values(values: list[Any]) -> Any:
    try:
        import torch

        if all(isinstance(value, torch.Tensor) for value in values):
            return torch.cat(values, dim=0)
    except ImportError:  # pragma: no cover - torch is a runtime dependency
        pass
    if all(isinstance(value, np.ndarray) for value in values):
        return np.concatenate(values, axis=0)
    arrays = [np.asarray(value) for value in values]
    return np.concatenate(arrays, axis=0)


def _aggregate_multimodal_outputs(result: Any) -> Mapping[str, Any] | None:
    outputs = _completion_multimodal_outputs(result)
    if not outputs:
        return None
    if len(outputs) == 1:
        return outputs[0]
    keys = set(outputs[0])
    if any(set(output) != keys for output in outputs[1:]):
        raise RuntimeError("Parallel policy completions returned different output fields")
    aggregated: dict[str, Any] = {}
    for key in keys:
        values = [output[key] for output in outputs]
        aggregated[key] = _concatenate_policy_values(values) if key in _PARALLEL_POLICY_BATCH_KEYS else values
    return aggregated


def _multimodal_output(result: Any) -> Mapping[str, Any] | None:
    return _aggregate_multimodal_outputs(result)


def _to_builtin_container(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Mapping):
        return {key: _to_builtin_container(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_builtin_container(item) for item in value]
    return value


def _to_policy_wire_value(value: Any) -> Any:
    """Convert engine tensors recursively to OpenPI-msgpack-compatible values."""

    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().to(device="cpu", dtype=torch.float32).numpy()
    except ImportError:  # pragma: no cover - torch is a runtime dependency
        pass
    if isinstance(value, np.ndarray) and value.dtype.kind == "f":
        return np.asarray(value, dtype=np.float32)
    if isinstance(value, Mapping):
        return {str(key): _to_policy_wire_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_policy_wire_value(item) for item in value]
    return value


@dataclass(frozen=True)
class PolicyServerConfig:
    """OpenPI policy server handshake config.

    Values are model-specific and must be provided by the loaded policy model.
    """

    values: dict[str, Any]

    @classmethod
    def from_model_config(cls, model_config: Any) -> PolicyServerConfig:
        if isinstance(model_config, Mapping):
            raw_config = model_config.get("policy_server_config")
            hf_config = model_config.get("hf_config")
        else:
            raw_config = getattr(model_config, "policy_server_config", None)
            hf_config = getattr(model_config, "hf_config", None)

        if raw_config is None and hf_config is not None:
            if isinstance(hf_config, Mapping):
                raw_config = hf_config.get("policy_server_config")
            else:
                raw_config = getattr(hf_config, "policy_server_config", None)

        if raw_config is None:
            raise ValueError("Robot OpenPI serving requires policy_server_config.")
        if isinstance(raw_config, cls):
            return raw_config
        if not isinstance(raw_config, Mapping):
            raise ValueError("Robot OpenPI serving requires policy_server_config.")
        return cls(_to_builtin_container(raw_config))

    def to_dict(self) -> dict[str, Any]:
        return _to_builtin_container(self.values)

    @property
    def structured_policy_result(self) -> bool:
        """Whether the wire response preserves policy metadata.

        The historical vLLM-Omni OpenPI endpoint returns only the value under
        ``multimodal_output["actions"]``.  Reasoning VLAs may additionally
        produce useful per-action metadata (for example a chain of
        causation), so models can opt into the upstream-style result envelope
        through their handshake config without changing existing policies.
        """

        return bool(self.values.get("structured_policy_result", False))

    @property
    def engine_request_type(self) -> str:
        """Engine input path used by the policy (``diffusion`` or ``ar``)."""

        return str(self.values.get("engine_request_type", "diffusion"))


class ServingRealtimeRobotOpenPI:
    """Robot policy serving layer for OpenPI protocol.

    Model-specific transform/state lives in the diffusion pipeline.
    """

    def __init__(
        self,
        engine_client: Any,
        model_name: str | None = None,
    ) -> None:
        self.engine_client = engine_client
        self.model_name = model_name
        self.policy_server_config = self._get_policy_server_config(engine_client)
        self._request_counter = count()
        self.request_adapter = load_openpi_request_adapter(
            self.policy_server_config.values.get("request_adapter"),
            self.policy_server_config.values,
        )
        if self.policy_server_config.engine_request_type != "diffusion" and self.request_adapter is None:
            raise ValueError("Non-diffusion OpenPI policies must declare policy_server_config.request_adapter")

    @classmethod
    def create_policy_server(
        cls,
        engine_client: Any,
        model_name: str | None = None,
    ) -> ServingRealtimeRobotOpenPI | None:
        try:
            return cls(engine_client=engine_client, model_name=model_name)
        except ValueError as exc:
            if "policy_server_config" not in str(exc):
                raise
            logger.info("Robot OpenPI serving disabled for model %s", model_name)
            return None

    @staticmethod
    def _get_policy_server_config(engine_client: Any) -> PolicyServerConfig:
        model_config = None
        get_od_config = getattr(engine_client, "get_diffusion_od_config", None)
        if callable(get_od_config):
            od_config = get_od_config()
            model_config = getattr(od_config, "model_config", None)

        if model_config is None:
            for stage_config in getattr(engine_client, "stage_configs", []) or []:
                if getattr(stage_config, "stage_type", None) != "diffusion":
                    continue
                engine_args = getattr(stage_config, "engine_args", None)
                model_config = getattr(engine_args, "model_config", None)
                if model_config is not None:
                    break

        if model_config is None:
            for stage_config in getattr(engine_client, "stage_configs", []) or []:
                engine_args = getattr(stage_config, "engine_args", None)
                model_config = getattr(engine_args, "model_config", None)
                if model_config is not None:
                    break

        if model_config is None:
            od_config = getattr(engine_client, "od_config", None)
            model_config = getattr(od_config, "model_config", None)

        if model_config is None:
            model_config = getattr(engine_client, "model_config", None)
        return PolicyServerConfig.from_model_config(model_config)

    def reset(self, obs: dict) -> None:
        """Compatibility hook; per-connection state lives in RobotRealtimeConnection."""

    async def infer(self, obs: dict, *, session_id: str, reset: bool) -> PolicyOutput:
        """raw obs → engine → actions."""
        # Build request, run inference through AsyncOmni
        request = self._build_request(obs, session_id=session_id, reset=reset)
        result = None
        # OpenPI policy serving is one request -> one action reply. AsyncOmni
        # exposes an async iterator, so consume it to completion and use the
        # final output, matching other non-streaming OpenAI serving paths.
        async for output in self.engine_client.generate(
            prompt=request.prompt,
            request_id=request.request_id,
            sampling_params_list=[request.sampling_params],
        ):
            result = output
        if result is None:
            raise RuntimeError("Robot OpenPI request produced no output.")

        return self._extract_policy_output(result)

    def _next_request_id(self, session_id: str) -> str:
        return f"robot-{session_id}-{next(self._request_counter)}"

    def _build_request(self, obs: dict, *, session_id: str, reset: bool) -> Any:
        """Build engine request from raw robot obs.

        Returns an `OmniDiffusionRequest` payload consumed by
        `AsyncOmni.generate()` and routed to the diffusion stage.
        """
        request_id = self._next_request_id(session_id)
        if self.request_adapter is not None:
            return self.request_adapter.build_request(
                obs,
                request_id=request_id,
                session_id=session_id,
                reset=reset,
            )

        extra_args = {
            "reset": reset,
            "session_id": session_id,
            "robot_obs": obs,
        }

        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        prompt = obs.get("prompt", "")
        sampling_params = OmniDiffusionSamplingParams(extra_args=extra_args)
        request = OmniDiffusionRequest(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
        )
        return OpenPIEngineRequest(
            prompt=request.prompt,
            sampling_params=request.sampling_params,
            request_id=request.request_id,
        )

    def _extract_actions(self, result: Any) -> ActionOutput:
        """Extract actions from engine result."""
        multimodal_output = _multimodal_output(result)
        if not isinstance(multimodal_output, Mapping):
            raise RuntimeError("Missing multimodal_output in robot policy result")

        actions = multimodal_output.get("actions")
        if actions is None:
            raise RuntimeError("Missing multimodal_output['actions'] in robot policy result")
        if isinstance(actions, Mapping):
            return {str(key): np.asarray(value, dtype=np.float32) for key, value in actions.items()}
        return np.asarray(actions, dtype=np.float32)

    def _extract_policy_output(self, result: Any) -> PolicyOutput:
        """Return actions alone or an opt-in structured policy result."""

        if not self.policy_server_config.structured_policy_result:
            return self._extract_actions(result)

        multimodal_output = _multimodal_output(result)
        if not isinstance(multimodal_output, Mapping):
            raise RuntimeError("Missing multimodal_output in robot policy result")
        if "actions" not in multimodal_output:
            raise RuntimeError("Missing multimodal_output['actions'] in robot policy result")

        output = _to_policy_wire_value(multimodal_output)
        actions = output["actions"]
        if isinstance(actions, Mapping):
            output["actions"] = {str(key): np.asarray(value, dtype=np.float32) for key, value in actions.items()}
        else:
            output["actions"] = np.asarray(actions, dtype=np.float32)
        completions = getattr(result, "outputs", None) or []
        reasoning = [completion.text for completion in completions if getattr(completion, "text", None)]
        if reasoning:
            output.setdefault("reasoning", reasoning[0] if len(reasoning) == 1 else reasoning)
        return output
