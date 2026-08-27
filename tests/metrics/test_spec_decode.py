from vllm_omni.metrics.spec_decode import RequestSpecDecodeMetrics


def test_summary_spec_decode_metrics() -> None:
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


def test_detailed_spec_decode_metrics() -> None:
    metrics = RequestSpecDecodeMetrics.new(2)
    metrics.observe(num_draft_tokens=2, num_accepted=1, detailed=True)

    payload = metrics.to_dict()
    assert payload["per_step_accepted"] == [1]
    assert payload["per_step_drafted"] == [2]
