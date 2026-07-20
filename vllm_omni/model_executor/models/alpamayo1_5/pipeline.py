# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-stage native vLLM pipeline for Alpamayo 1.5."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

ALPAMAYO1_5_PIPELINE = PipelineConfig(
    model_type="alpamayo1_5",
    model_arch="Alpamayo1_5",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="policy",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            # Treat the structured trajectory as the terminal payload while
            # retaining the AR completion text as reasoning metadata.
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
