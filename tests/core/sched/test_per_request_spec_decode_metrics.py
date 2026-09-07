from types import SimpleNamespace

import pytest

from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin
from vllm_omni.metrics.spec_decode import RequestSpecDecodeMetrics

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Scheduler(OmniSchedulerMixin):
    def __init__(self, level: str):
        self.spec_decode_metrics_level = level


def test_request_spec_decode_metrics_summary() -> None:
    metrics = RequestSpecDecodeMetrics.new(3)
    metrics.observe(num_draft_tokens=3, num_accepted=2)
    metrics.observe(num_draft_tokens=3, num_accepted=0)

    assert metrics.to_dict() == {
        "mean_acceptance_length": 2.0,
        "draft_acceptance_rate": 1 / 3,
        "acceptance_histogram": [1, 0, 1, 0],
        "num_spec_steps": 2,
        "num_accepted_draft_tokens": 2,
        "num_draft_tokens": 6,
        "num_spec_tokens": 3,
    }


def test_request_spec_decode_metrics_detailed() -> None:
    metrics = RequestSpecDecodeMetrics.new(2)
    metrics.observe(num_draft_tokens=2, num_accepted=1, detailed=True)

    payload = metrics.to_dict()

    assert payload["per_step_accepted"] == [1]
    assert payload["per_step_drafted"] == [2]


def test_scheduler_observation_excludes_invalid_draft_tokens() -> None:
    scheduler = _Scheduler("summary")
    request = SimpleNamespace(spec_decode_metrics=RequestSpecDecodeMetrics.new(3))

    scheduler._observe_per_request_spec_decode_metrics(
        request,
        num_draft_tokens=3,
        num_accepted_tokens=1,
        num_invalid_spec_tokens=1,
    )

    assert scheduler._per_request_spec_decode_metrics_snapshot(request, finished=True) == {
        "mean_acceptance_length": 2.0,
        "draft_acceptance_rate": 0.5,
        "acceptance_histogram": [0, 1, 0, 0],
        "num_spec_steps": 1,
        "num_accepted_draft_tokens": 1,
        "num_draft_tokens": 2,
        "num_spec_tokens": 3,
    }


def test_scheduler_emits_metrics_only_when_finished() -> None:
    scheduler = _Scheduler("summary")
    request = SimpleNamespace(spec_decode_metrics=RequestSpecDecodeMetrics.new(3))

    assert scheduler._per_request_spec_decode_metrics_snapshot(request, finished=False) is None
    assert (
        scheduler._per_request_spec_decode_metrics_snapshot(SimpleNamespace(spec_decode_metrics=None), finished=True)
        is None
    )
