# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""ModelOpt W4A16 AWQ compatibility for vLLM.

ModelOpt exports ``W4A16_AWQ`` weights in a layout that upstream vLLM 0.27
does not yet recognize: two two's-complement INT4 output channels per uint8,
with per-group scales stored as ``[out_features, in_features / group_size]``.
This module loads that native checkpoint layout, converts it once to vLLM's
standard GPTQ-like packing, and delegates inference to the existing mixed-
precision kernel selection (Marlin on supported NVIDIA GPUs).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from vllm.model_executor.kernels.linear import (
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptFp8Config,
    ModelOptQuantConfigBase,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    PackedvLLMParameter,
    RowvLLMParameter,
)
from vllm.scalar_type import scalar_types

logger = logging.getLogger(__name__)

_QUANT_ALGO = "W4A16_AWQ"
_PACK_FACTOR = 8  # INT4 values per int32 in vLLM's kernel-facing layout.


def _noop_weight_loader(*args: Any, **kwargs: Any) -> None:
    """Loader for parameters created after checkpoint loading has finished."""


def _modelopt_uint8_to_int32(weight: torch.Tensor) -> torch.Tensor:
    """Convert ModelOpt ``[N/2, K]`` bytes to standard ``[K/8, N]`` INT4.

    ModelOpt 0.37 places the even output channel in the low nibble and the odd
    output channel in the high nibble, using two's-complement INT4. vLLM's
    Hopper kernels accept ``uint4b8`` (offset-binary), so toggle the sign bit
    while repacking. For four-bit values, ``twos_complement XOR 8`` is exactly
    ``signed_value + 8``.
    """
    if weight.dtype != torch.uint8 or weight.ndim != 2:
        raise ValueError(
            "ModelOpt W4A16_AWQ weight must be a 2-D uint8 tensor, got "
            f"shape={tuple(weight.shape)}, dtype={weight.dtype}."
        )

    packed_out, size_k = weight.shape
    size_n = packed_out * 2
    if size_k % _PACK_FACTOR:
        raise ValueError(f"ModelOpt W4A16_AWQ in_features ({size_k}) must be divisible by {_PACK_FACTOR}.")

    # Build the output a nibble-plane at a time. This avoids materializing a
    # full int32 [K, N] tensor (4x the uncompressed INT4 weights) at startup.
    converted = torch.zeros(
        size_k // _PACK_FACTOR,
        size_n,
        dtype=torch.int32,
        device=weight.device,
    )
    for index in range(_PACK_FACTOR):
        modelopt_bytes = weight[:, index::_PACK_FACTOR].T
        shift = index * 4
        even = ((modelopt_bytes & 0x0F) ^ 0x08).to(torch.int32)
        odd = ((modelopt_bytes >> 4) ^ 0x08).to(torch.int32)
        converted[:, 0::2] |= even << shift
        converted[:, 1::2] |= odd << shift
    return converted.contiguous()


class ModelOptW4A16AwqConfig(ModelOptQuantConfigBase):
    """Configuration for ModelOpt's symmetric, group-wise INT4 format."""

    LinearMethodCls: type = None  # type: ignore[assignment]

    def __init__(
        self,
        group_size: int,
        exclude_modules: list[str],
        *,
        has_zero_point: bool,
        pre_quant_scale: bool,
    ) -> None:
        super().__init__(exclude_modules)
        if has_zero_point:
            raise ValueError("ModelOpt W4A16_AWQ checkpoints with zero points are not supported.")
        if group_size <= 0:
            raise ValueError(f"ModelOpt W4A16_AWQ requires a positive group_size, got {group_size}.")
        self.group_size = group_size
        self.has_zero_point = has_zero_point
        self.pre_quant_scale = pre_quant_scale
        self.LinearMethodCls = ModelOptW4A16AwqLinearMethod

    def get_name(self):
        return "modelopt"

    def is_layer_excluded(self, prefix: str) -> bool:
        # The released Qwen3-VL export names its exclusion
        # ``model.layers.visual*``. vLLM hoists that subtree to ``visual``
        # (and Alpamayo may add an outer ``vlm`` prefix), so the literal
        # ModelOpt wildcard no longer matches after architecture mapping.
        if "visual" in prefix.split("."):
            return True
        return super().is_layer_excluded(prefix)

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def _from_config(
        cls,
        *,
        quant_method: str,
        exclude_modules: list[str],
        original_config: dict[str, Any],
        group_size: int | None,
        **kwargs: Any,
    ) -> ModelOptW4A16AwqConfig:
        if quant_method != _QUANT_ALGO:
            raise ValueError(f"Expected {_QUANT_ALGO}, got {quant_method}.")
        quantization = original_config.get("quantization", original_config)
        if group_size is None:
            raise ValueError("ModelOpt W4A16_AWQ config is missing group_size.")
        return cls(
            group_size,
            exclude_modules,
            has_zero_point=bool(quantization.get("has_zero_point", False)),
            pre_quant_scale=bool(quantization.get("pre_quant_scale", False)),
        )


class ModelOptW4A16AwqLinearMethod(LinearMethodBase):
    """Load ModelOpt AWQ tensors and execute them through vLLM MP kernels."""

    _kernel_backends_being_used: set[str] = set()

    def __init__(self, quant_config: ModelOptW4A16AwqConfig) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        group_size = self.quant_config.group_size
        if input_size_per_partition % group_size:
            raise ValueError(
                f"in_features per partition ({input_size_per_partition}) must "
                f"be divisible by ModelOpt AWQ group_size ({group_size})."
            )
        if output_size_per_partition % 2:
            raise ValueError(f"out_features per partition ({output_size_per_partition}) must be even.")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        kernel_config = MPLinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=(
                input_size_per_partition,
                output_size_per_partition,
            ),
            weight_type=scalar_types.uint4b8,
            act_type=params_dtype,
            group_size=group_size,
            zero_points=False,
            has_g_idx=False,
        )
        kernel_type = choose_mp_linear_kernel(kernel_config)
        if kernel_type.__name__ not in self._kernel_backends_being_used:
            logger.info(
                "Using %s for ModelOpt W4A16_AWQ linear layers.",
                kernel_type.__name__,
            )
            self._kernel_backends_being_used.add(kernel_type.__name__)

        # Native ModelOpt layout: nibbles are packed along the output dim.
        weight = PackedvLLMParameter(
            data=torch.empty(
                output_size_per_partition // 2,
                input_size_per_partition,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            packed_dim=0,
            packed_factor=2,
            weight_loader=weight_loader,
        )
        # Native ModelOpt scale layout mirrors the logical [N, K] weight.
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // group_size,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)
        layer.register_parameter("weight_scale", weight_scale)

        if self.quant_config.pre_quant_scale:
            # Only row-parallel o_proj/down_proj tensors are present in the
            # released checkpoint. Other layers retain the identity default.
            pre_quant_scale = RowvLLMParameter(
                data=torch.ones(input_size_per_partition, dtype=params_dtype),
                input_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("pre_quant_scale", pre_quant_scale)

        self.kernel = kernel_type(
            kernel_config,
            w_q_param_name="weight",
            w_s_param_name="weight_scale",
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        converted_weight = PackedvLLMParameter(
            data=_modelopt_uint8_to_int32(layer.weight.data),
            input_dim=0,
            output_dim=1,
            packed_dim=0,
            packed_factor=_PACK_FACTOR,
            weight_loader=_noop_weight_loader,
        )
        converted_scale = GroupQuantScaleParameter(
            data=layer.weight_scale.data.T.contiguous(),
            input_dim=0,
            output_dim=1,
            weight_loader=_noop_weight_loader,
        )
        layer.weight = converted_weight
        layer.weight_scale = converted_scale
        self.kernel.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pre_quant_scale = getattr(layer, "pre_quant_scale", None)
        if pre_quant_scale is not None:
            x = x * pre_quant_scale
        return self.kernel.apply_weights(layer, x, bias)


def install_modelopt_w4a16_awq_patch() -> None:
    """Teach the pinned vLLM ModelOpt registry about ``W4A16_AWQ``."""
    import vllm.model_executor.layers.quantization.modelopt as modelopt
    from vllm.transformers_utils.model_arch_config_convertor import (
        ModelArchConfigConvertorBase,
    )

    if _QUANT_ALGO not in modelopt.QUANT_ALGOS:
        modelopt.QUANT_ALGOS.append(_QUANT_ALGO)

    current_from_config = ModelOptFp8Config._from_config.__func__
    if not getattr(current_from_config, "_vllm_omni_w4a16_awq", False):
        original_from_config = current_from_config

        def _from_config(cls, *, quant_method: str, **kwargs: Any):
            if quant_method == _QUANT_ALGO:
                return ModelOptW4A16AwqConfig._from_config(quant_method=quant_method, **kwargs)
            return original_from_config(cls, quant_method=quant_method, **kwargs)

        _from_config._vllm_omni_w4a16_awq = True  # type: ignore[attr-defined]
        ModelOptFp8Config._from_config = classmethod(_from_config)

    current_normalize = ModelArchConfigConvertorBase._normalize_quantization_config
    if not getattr(current_normalize, "_vllm_omni_w4a16_awq", False):
        original_normalize = current_normalize

        def _normalize_quantization_config(self, config):
            quant_cfg = getattr(config, "quantization_config", None)
            if isinstance(quant_cfg, dict):
                producer = quant_cfg.get("producer", {}).get("name")
                nested = quant_cfg.get("quantization", {})
                algo = nested.get("quant_algo") if isinstance(nested, dict) else None
                if producer == "modelopt" and str(algo).upper() == _QUANT_ALGO:
                    quant_cfg["quant_method"] = "modelopt"
                    return quant_cfg
            return original_normalize(self, config)

        _normalize_quantization_config._vllm_omni_w4a16_awq = True  # type: ignore[attr-defined]
        ModelArchConfigConvertorBase._normalize_quantization_config = _normalize_quantization_config

    current_override = ModelOptFp8Config.override_quantization_method.__func__
    if not getattr(current_override, "_vllm_omni_w4a16_awq", False):
        original_override = current_override

        def _override(cls, hf_quant_cfg, user_quant, hf_config=None):
            algo = cls._extract_modelopt_quant_algo(hf_quant_cfg)
            if algo == _QUANT_ALGO and user_quant in (None, "modelopt"):
                return "modelopt"
            return original_override(cls, hf_quant_cfg, user_quant, hf_config)

        _override._vllm_omni_w4a16_awq = True  # type: ignore[attr-defined]
        ModelOptFp8Config.override_quantization_method = classmethod(_override)

    logger.info("Installed ModelOpt W4A16_AWQ compatibility support.")
