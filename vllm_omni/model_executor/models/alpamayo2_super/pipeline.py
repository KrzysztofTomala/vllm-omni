# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Alpamayo 2 Super single-stage policy topology."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

ALPAMAYO2_SUPER_PIPELINE = PipelineConfig(
    model_type="alpamayo2_super",
    default_deploy_config_name="alpamayo2_super.yaml",
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
        ),
    ),
)
