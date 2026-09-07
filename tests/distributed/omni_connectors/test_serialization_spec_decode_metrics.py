import pytest
from vllm.outputs import CompletionOutput

from vllm_omni.distributed.omni_connectors.utils.serialization import (
    OmniMsgpackDecoder,
    OmniMsgpackEncoder,
)
from vllm_omni.metrics.spec_decode import RequestSpecDecodeMetrics

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_completion_spec_decode_metrics_roundtrip() -> None:
    output = CompletionOutput(
        index=0,
        text="result",
        token_ids=[1, 2],
        cumulative_logprob=None,
        logprobs=None,
        finish_reason="stop",
        stop_reason=None,
    )
    metrics = RequestSpecDecodeMetrics.new(7)
    metrics.observe(num_draft_tokens=7, num_accepted=4)
    metrics.observe(num_draft_tokens=7, num_accepted=6)
    output.spec_decode_metrics = metrics

    decoded = OmniMsgpackDecoder().decode(OmniMsgpackEncoder().encode(output))

    assert isinstance(decoded, CompletionOutput)
    assert decoded.spec_decode_metrics == metrics.to_dict()
