# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RequestSpecDecodeMetrics:
    """Per-request speculative-decoding acceptance statistics."""

    num_spec_tokens: int
    histogram: list[int] = field(default_factory=list)
    num_draft_tokens: int = 0
    per_step_accepted: list[int] = field(default_factory=list)
    per_step_drafted: list[int] = field(default_factory=list)

    @classmethod
    def new(cls, num_spec_tokens: int) -> "RequestSpecDecodeMetrics":
        return cls(
            num_spec_tokens=num_spec_tokens,
            histogram=[0] * (num_spec_tokens + 1),
        )

    def observe(
        self,
        num_draft_tokens: int,
        num_accepted: int,
        detailed: bool = False,
    ) -> None:
        self.histogram[num_accepted] += 1
        self.num_draft_tokens += num_draft_tokens
        if detailed:
            self.per_step_accepted.append(num_accepted)
            self.per_step_drafted.append(num_draft_tokens)

    def to_dict(self) -> dict[str, Any]:
        num_spec_steps = sum(self.histogram)
        num_accepted = sum(accepted * count for accepted, count in enumerate(self.histogram))
        result: dict[str, Any] = {
            "mean_acceptance_length": (1.0 + num_accepted / num_spec_steps if num_spec_steps else 1.0),
            "draft_acceptance_rate": (num_accepted / self.num_draft_tokens if self.num_draft_tokens else 0.0),
            "acceptance_histogram": list(self.histogram),
            "num_spec_steps": num_spec_steps,
            "num_accepted_draft_tokens": num_accepted,
            "num_draft_tokens": self.num_draft_tokens,
            "num_spec_tokens": self.num_spec_tokens,
        }
        if self.per_step_accepted:
            result["per_step_accepted"] = list(self.per_step_accepted)
            result["per_step_drafted"] = list(self.per_step_drafted)
        return result
