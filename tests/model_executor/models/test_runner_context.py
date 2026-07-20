# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.model_executor.models.runner_context import RunnerKVCacheContext

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_runner_kv_cache_context_validates_request_alignment():
    with pytest.raises(ValueError, match="must align"):
        RunnerKVCacheContext(
            caches=[],
            block_table=torch.zeros(1, 1, dtype=torch.int32),
            sequence_lengths=(10,),
            request_ids=("first", "second"),
        )
