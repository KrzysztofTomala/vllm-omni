# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Alpamayo 1.5 single-stage policy topology."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

ALPAMAYO1_5_PIPELINE = PipelineConfig(
    model_type="alpamayo1_5",
    model_arch="Alpamayo1_5Pipeline",
    hf_architectures=("Alpamayo1_5",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="actions",
            model_arch="Alpamayo1_5Pipeline",
        ),
    ),
)
