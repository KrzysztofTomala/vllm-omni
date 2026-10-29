# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""KV handoff for SamplingParams.n > 1: one transfer per sample, all of them to the diffusion request."""

from types import SimpleNamespace

import pytest
import torch
from vllm import SamplingParams

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import OmniKVCacheConfig, OmniKVTransferManager
from vllm_omni.engine.orchestrator import Orchestrator
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _runner_with_global_id(req_id: str, global_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        requests={req_id: object()},
        model_intermediate_buffer={req_id: {"global_request_id": [global_id]}},
    )


@pytest.mark.parametrize(
    "req_id, global_id, expected",
    [
        ("req-a", "req-a", "req-a"),
        # vLLM names the children of an n > 1 request "{index}_{parent}".
        ("0_req-a", "req-a", "0_req-a"),
        ("5_req-a", "req-a", "5_req-a"),
        # An engine-internal id that is not a child resolves to the global id.
        ("req-a-1f2e", "req-a", "req-a"),
    ],
)
def test_each_sample_transfers_kv_under_its_own_id(req_id, global_id, expected):
    runner = _runner_with_global_id(req_id, global_id)
    assert GPUARModelRunner._resolve_global_request_id(runner, req_id) == expected


def test_diffusion_params_follow_source_sample_count():
    params = OmniDiffusionSamplingParams()
    sampled = Orchestrator._diffusion_params_for_samples(SamplingParams(n=4), params)
    assert sampled.num_outputs_per_prompt == 4
    assert params.num_outputs_per_prompt == 1  # the request's params are not mutated
    assert Orchestrator._diffusion_params_for_samples(SamplingParams(n=1), params) is params
    llm_params = SamplingParams()
    assert Orchestrator._diffusion_params_for_samples(SamplingParams(n=4), llm_params) is llm_params


def _receiver(monkeypatch, available: set[str]) -> OmniKVTransferManager:
    manager = OmniKVTransferManager(
        OmniKVCacheConfig(connector_config={"type": "mock"}, stage_id=1, engine_input_source=[0], need_recv_cache=True)
    )

    def fake_receive(request_id, target_device=None, **kwargs):
        if request_id not in available:
            return None, 0
        layer = torch.full((2, 1, 4), float(request_id[0]))
        data = {"layer_blocks": {"key_cache": [layer], "value_cache": [layer]}, "metadata": {"key": request_id}}
        return data, 1

    monkeypatch.setattr(manager, "receive_kv_cache_for_request", fake_receive)
    return manager


def test_diffusion_request_receives_every_sample(monkeypatch):
    manager = _receiver(monkeypatch, {"0_req", "1_req", "2_req"})
    req = OmniDiffusionRequest(
        prompt="", sampling_params=OmniDiffusionSamplingParams(num_outputs_per_prompt=3), request_id="req"
    )

    assert manager.receive_multi_kv_cache(req)
    caches = req.sampling_params.past_key_values
    assert [cache.key_cache[0][0, 0, 0].item() for cache in caches] == [0.0, 1.0, 2.0]
    assert [metadata["key"] for metadata in req.sampling_params.kv_metadata] == ["0_req", "1_req", "2_req"]


def test_missing_sample_fails_the_receive(monkeypatch):
    manager = _receiver(monkeypatch, {"0_req", "2_req"})
    req = OmniDiffusionRequest(
        prompt="", sampling_params=OmniDiffusionSamplingParams(num_outputs_per_prompt=3), request_id="req"
    )

    assert not manager.receive_multi_kv_cache(req)
    assert req.sampling_params.past_key_values is None
