# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-stage policy pipeline for Alpamayo 2 Super."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

ALPAMAYO2_SUPER_PIPELINE = PipelineConfig(
    model_type="alpamayo2_super",
    model_arch="Alpamayo2Super",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="policy",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="latent",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="latent",
            # Token budgets and stop tokens are request-specific: policy
            # inference stops at future_end, meta-action at future_start, and
            # auto-labeling/grounding must be allowed to finish text output.
            sampling_constraints={
                "detokenize": True,
            },
        ),
    ),
)
