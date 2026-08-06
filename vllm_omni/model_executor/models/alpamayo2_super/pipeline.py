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
            sampling_constraints={
                "detokenize": True,
                "stop_token_ids": [155683],
                "max_tokens": 128,
            },
        ),
    ),
)
