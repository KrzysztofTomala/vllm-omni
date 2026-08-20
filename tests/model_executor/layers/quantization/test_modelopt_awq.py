# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import torch
from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8Config

from vllm_omni.model_executor.layers.quantization.modelopt_awq import (
    ModelOptW4A16AwqConfig,
    _modelopt_uint8_to_int32,
    install_modelopt_w4a16_awq_patch,
)


def test_modelopt_twos_complement_conversion() -> None:
    # Logical signed [N, K] values in the exact range representable by INT4.
    signed = (torch.arange(16 * 8, dtype=torch.int16).reshape(16, 8) % 16) - 8
    twos_complement = (signed & 0x0F).to(torch.uint8)
    # ModelOpt 0.37: even output in low nibble, odd output in high nibble.
    modelopt = (twos_complement[0::2] | (twos_complement[1::2] << 4)).contiguous()

    packed = _modelopt_uint8_to_int32(modelopt)
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    offset_binary = ((packed.unsqueeze(1) >> shifts[None, :, None]) & 0x0F).reshape(8, 16).T

    assert torch.equal(offset_binary.to(torch.int16) - 8, signed)


def test_modelopt_w4a16_config_dispatch_and_visual_exclusion() -> None:
    install_modelopt_w4a16_awq_patch()
    config = ModelOptFp8Config.from_config(
        {
            "quantization": {
                "quant_algo": "W4A16_AWQ",
                "group_size": 128,
                "has_zero_point": False,
                "pre_quant_scale": True,
                "exclude_modules": ["model.layers.visual*", "lm_head"],
            }
        }
    )

    assert isinstance(config, ModelOptW4A16AwqConfig)
    assert config.group_size == 128
    assert config.pre_quant_scale is True
    assert config.is_layer_excluded("vlm.visual.blocks.0.attn.qkv")
    assert config.is_layer_excluded("vlm.language_model.lm_head")
    assert not config.is_layer_excluded("vlm.language_model.model.layers.0.self_attn.qkv_proj")
