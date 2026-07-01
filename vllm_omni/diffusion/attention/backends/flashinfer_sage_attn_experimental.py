# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental FlashInfer SageAttention diffusion backend.

On SM100 this adapter targets FlashInfer's TRT-LLM ragged DiT kernel through
``trtllm_ragged_attention_deepseek``.  The Sage variant consumes INT8 Q/K with
per-token/per-block dequantization factors and FP8 E4M3 V.  vLLM-Omni supplies
BF16/FP16 BSHD tensors, so this implementation smooths and quantizes them with
either a conservative PyTorch path or fused Triton preprocessing. SM100 keeps
native GQA and physically pads odd K/V sequence lengths to use 16-token K
quantization blocks.

The available SM100 Sage cubins are non-causal, head-dimension-128 kernels.
Unsupported calls (including Cosmos3's causal understanding attention) use
the regular dense FlashInfer backend unless
``VLLM_OMNI_FLASHINFER_SAGE_STRICT=1`` is set.

The backend is intentionally opt-in and currently supports SM100 only.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.backends.flashinfer_attn import (
    FlashInferAttentionImpl,
)

logger = init_logger(__name__)

try:
    import flashinfer

    HAS_FLASHINFER = True
except Exception as exc:  # pragma: no cover - environment dependent
    flashinfer = None
    HAS_FLASHINFER = False
    logger.warning("FlashInfer import failed for experimental Sage backend: %s", exc)

try:
    from vllm_omni.diffusion.attention.backends.flashinfer_sage_preprocess_triton import (
        SageQuantBuffers,
        allocate_sage_quant_buffers,
        preprocess_sage_sm100,
    )

    HAS_SAGE_TRITON_PREPROCESS = True
except Exception as exc:  # pragma: no cover - environment dependent
    SageQuantBuffers = None  # type: ignore[assignment,misc]
    HAS_SAGE_TRITON_PREPROCESS = False
    logger.warning("Triton Sage preprocessing is unavailable: %s", exc)


_STRICT_ENV = "VLLM_OMNI_FLASHINFER_SAGE_STRICT"
_DIAGNOSTICS_ENV = "VLLM_OMNI_FLASHINFER_SAGE_DIAGNOSTICS"
_PREPROCESS_ENV = "VLLM_OMNI_FLASHINFER_SAGE_PREPROCESS"
_WORKSPACE_RESET_ENV = "VLLM_OMNI_FLASHINFER_SAGE_WORKSPACE_RESET"
_REQUIRE_TESTED_VERSION_ENV = "VLLM_OMNI_FLASHINFER_SAGE_REQUIRE_TESTED_VERSION"
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FP8_E4M3 = torch.float8_e4m3fn
_TESTED_FLASHINFER_VERSIONS = {"0.6.14"}


class FlashInferSageAttentionExperimentalBackend(AttentionBackend):
    """Experimental FlashInfer TRT-LLM SageAttention for SM100."""

    accept_output_buffer: bool = True

    @classmethod
    def supports_attention_mask(cls) -> bool:
        # Explicit masks are handled by the dense FlashInfer fallback.
        return True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [128]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_SAGE_ATTN_EXPERIMENTAL"

    @staticmethod
    def get_impl_cls() -> type["FlashInferSageAttentionExperimentalImpl"]:
        return FlashInferSageAttentionExperimentalImpl


class FlashInferSageAttentionExperimentalImpl(AttentionImpl):
    """Adapt vLLM-Omni BSHD attention to FlashInfer Sage kernels."""

    _warned_fallback_reasons: set[str] = set()
    _logged_kernel_paths: set[str] = set()
    _diagnostic_call_count: int = 0
    _diagnostic_dumped_nan: bool = False
    # The TRT-LLM launcher uses workspace only during a call. Sharing one
    # allocation per CUDA stream avoids allocating 256 MiB in every model layer.
    _workspace_by_stream: dict[tuple[int, int], torch.Tensor] = {}
    # Keep at most one shape allocation per CUDA stream. Dynamic workloads
    # replace the entry instead of growing a shape-keyed cache indefinitely.
    _triton_buffers_by_stream: dict[tuple[int, int], tuple[tuple, object]] = {}

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        backend_kwargs: dict | None = None,
        **extra_impl_args,
    ) -> None:
        self.prefix = prefix or "<unknown>"
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_size = head_size
        self.softmax_scale = softmax_scale
        self.causal = causal
        backend_kwargs = backend_kwargs or {}
        env_strict = os.environ.get(_STRICT_ENV, "").strip().lower() in _TRUE_VALUES
        self.strict = bool(backend_kwargs.get("strict", env_strict))
        env_diagnostics = os.environ.get(_DIAGNOSTICS_ENV, "").strip().lower() in _TRUE_VALUES
        self.diagnostics = bool(backend_kwargs.get("diagnostics", env_diagnostics))
        self.smooth_k = bool(backend_kwargs.get("smooth_k", True))
        self.workspace_size = int(backend_kwargs.get("workspace_size", 256 * 1024 * 1024))
        env_require_version = (
            os.environ.get(_REQUIRE_TESTED_VERSION_ENV, "").strip().lower() in _TRUE_VALUES
        )
        self.require_tested_flashinfer_version = bool(
            backend_kwargs.get("require_tested_flashinfer_version", env_require_version)
        )
        if HAS_FLASHINFER and self._flashinfer_version() not in _TESTED_FLASHINFER_VERSIONS:
            logger.warning_once(
                "Experimental FlashInfer SageAttention was tested with FlashInfer %s; "
                "installed version is %s",
                ", ".join(sorted(_TESTED_FLASHINFER_VERSIONS)),
                self._flashinfer_version(),
            )
        self.preprocess = str(
            backend_kwargs.get("preprocess", os.environ.get(_PREPROCESS_ENV, "torch"))
        ).lower()
        if self.preprocess not in {"torch", "triton", "auto"}:
            raise ValueError(
                f"Sage preprocess must be 'torch', 'triton', or 'auto', got {self.preprocess!r}"
            )
        self.workspace_reset = str(
            backend_kwargs.get(
                "workspace_reset", os.environ.get(_WORKSPACE_RESET_ENV, "always")
            )
        ).lower()
        if self.workspace_reset not in {"always", "counter", "once"}:
            raise ValueError(
                "Sage workspace_reset must be 'always', 'counter', or 'once', "
                f"got {self.workspace_reset!r}"
            )
        self._dense_fallback = FlashInferAttentionImpl(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
            **extra_impl_args,
        )

    @staticmethod
    def _sm100_api_available() -> bool:
        return bool(
            HAS_FLASHINFER
            and flashinfer is not None
            and hasattr(flashinfer, "prefill")
            and hasattr(flashinfer.prefill, "trtllm_ragged_attention_deepseek")
        )

    @staticmethod
    def _flashinfer_version() -> str:
        return str(getattr(flashinfer, "__version__", "unknown"))

    @classmethod
    def _workspace(cls, device: torch.device, size: int) -> torch.Tensor:
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        stream_id = int(torch.cuda.current_stream(device).cuda_stream)
        key = (device_index, stream_id)
        workspace = cls._workspace_by_stream.get(key)
        if workspace is None or workspace.numel() < size:
            # The TRT-LLM persistent scheduler requires zeroed counters on the
            # first call. Successful launches restore them to zero.
            workspace = torch.zeros(size, dtype=torch.uint8, device=device)
            cls._workspace_by_stream[key] = workspace
        return workspace

    @classmethod
    def _triton_buffers(
        cls, query: torch.Tensor, key: torch.Tensor
    ) -> "SageQuantBuffers":
        device_index = query.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        stream_id = int(torch.cuda.current_stream(query.device).cuda_stream)
        stream_key = (device_index, stream_id)
        shape_key = (
            query.dtype,
            tuple(query.shape),
            tuple(key.shape),
        )
        cached = cls._triton_buffers_by_stream.get(stream_key)
        if cached is None or cached[0] != shape_key:
            buffers = allocate_sage_quant_buffers(query, key)
            cls._triton_buffers_by_stream[stream_key] = (shape_key, buffers)
        else:
            buffers = cached[1]
        return buffers  # type: ignore[return-value]

    @classmethod
    def clear_runtime_caches(cls) -> None:
        """Release stream-local workspaces and quantization buffers."""
        cls._workspace_by_stream.clear()
        cls._triton_buffers_by_stream.clear()

    @staticmethod
    def _quantize_int8_blocked(
        tensor: torch.Tensor, block_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Match FlashInfer's SM100 Sage test scale layout on the GPU."""
        tokens, heads, head_dim = tensor.shape
        blocks = tensor.float().reshape(tokens // block_size, block_size, heads, head_dim)
        amax = blocks.abs().amax(dim=-1, keepdim=True).amax(dim=1, keepdim=True)
        amax = amax.clamp_min(1e-12)
        quant_scale = 127.0 / amax
        quantized = (blocks * quant_scale).round().clamp(-128, 127)
        quantized = quantized.reshape(tokens, heads, head_dim).to(torch.int8)
        # TRT-LLM expects [head, token-block] flattened, not token-major scales.
        inv_scale = (amax / 127.0).reshape(tokens // block_size, heads)
        inv_scale = inv_scale.T.flatten().contiguous()
        return quantized, inv_scale

    @staticmethod
    def _quantize_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Match ``_to_float8`` in FlashInfer's ragged DiT test."""
        finfo = torch.finfo(_FP8_E4M3)
        amax = tensor.float().abs().amax().clamp_min(1e-12)
        quant_scale = finfo.max / amax * 0.1
        quantized = (tensor.float() * quant_scale).clamp(finfo.min, finfo.max)
        return quantized.to(_FP8_E4M3), quant_scale.float().reciprocal()

    def _common_ineligibility_reason(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> str | None:
        if not HAS_FLASHINFER:
            return "FlashInfer is not importable"
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            return "expected rank-4 BSHD query/key/value tensors"
        if query.dtype not in (torch.float16, torch.bfloat16):
            return f"Sage preprocessing requires FP16/BF16 input, got {query.dtype}"
        if key.dtype != query.dtype or value.dtype != query.dtype:
            return "query/key/value dtypes must match"
        if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
            return "query/key/value batch sizes must match"
        if query.shape[3] != key.shape[3] or query.shape[3] != value.shape[3]:
            return "query/key/value head dimensions must match"
        if key.shape[2] != value.shape[2]:
            return "key/value head counts must match"
        if query.shape[2] % key.shape[2] != 0:
            return f"query heads ({query.shape[2]}) must be divisible by KV heads ({key.shape[2]})"
        if attn_metadata is not None and attn_metadata.attn_mask is not None:
            return "explicit attention masks are not supported by the Sage paths"
        return None

    def _sm100_ineligibility_reason(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> str | None:
        if not self._sm100_api_available():
            return "installed FlashInfer does not expose trtllm_ragged_attention_deepseek"
        version = self._flashinfer_version()
        if self.require_tested_flashinfer_version and version not in _TESTED_FLASHINFER_VERSIONS:
            return (
                f"FlashInfer {version} is untested; expected one of "
                f"{sorted(_TESTED_FLASHINFER_VERSIONS)}"
            )
        if self.causal:
            return "FlashInfer's SM100 Sage cubins do not include a causal-mask kernel"
        if query.shape[3] != 128:
            return f"FlashInfer's SM100 Sage kernel requires head dimension 128, got {query.shape[3]}"
        return None

    def _fallback(
        self,
        reason: str,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> torch.Tensor:
        if self.strict:
            raise RuntimeError(f"Experimental FlashInfer SageAttention unavailable: {reason}")
        if reason not in type(self)._warned_fallback_reasons:
            logger.warning(
                "Experimental FlashInfer SageAttention falling back to dense "
                "FLASHINFER_ATTN: %s. Set %s=1 to forbid fallback.",
                reason,
                _STRICT_ENV,
            )
            type(self)._warned_fallback_reasons.add(reason)
        out = self._dense_fallback.forward_cuda(query, key, value, attn_metadata)
        if self.diagnostics:
            self._log_tensor_diagnostics(
                path="dense-fallback",
                query=query,
                key=key,
                value=value,
                out=out,
                reference=None,
            )
        return out

    @staticmethod
    def _finite_and_absmax(tensor: torch.Tensor) -> tuple[bool, float]:
        tensor_float = tensor.float()
        return bool(torch.isfinite(tensor_float).all()), float(tensor_float.abs().max())

    def _log_tensor_diagnostics(
        self,
        *,
        path: str,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        out: torch.Tensor,
        reference: torch.Tensor | None,
        quantized: tuple[torch.Tensor, ...] = (),
    ) -> None:
        type(self)._diagnostic_call_count += 1
        call = type(self)._diagnostic_call_count
        q_finite, q_max = self._finite_and_absmax(query)
        k_finite, k_max = self._finite_and_absmax(key)
        v_finite, v_max = self._finite_and_absmax(value)
        out_finite, out_max = self._finite_and_absmax(out)
        quant_finite = all(bool(torch.isfinite(t.float()).all()) for t in quantized)
        dump_path = os.environ.get("VLLM_OMNI_FLASHINFER_SAGE_NAN_DUMP", "").strip()
        if (
            dump_path
            and not type(self)._diagnostic_dumped_nan
            and q_finite
            and k_finite
            and v_finite
            and not out_finite
        ):
            torch.save(
                {
                    "path": path,
                    "prefix": self.prefix,
                    "query": query.detach().cpu(),
                    "key": key.detach().cpu(),
                    "value": value.detach().cpu(),
                    "out": out.detach().cpu(),
                    "reference": None if reference is None else reference.detach().cpu(),
                    "quantized": tuple(t.detach().cpu() for t in quantized),
                    "softmax_scale": self.softmax_scale,
                    "smooth_k": self.smooth_k,
                },
                dump_path,
            )
            type(self)._diagnostic_dumped_nan = True
            logger.warning("SAGE_DIAG saved first finite-input/nonfinite-output payload to %s", dump_path)
        if reference is None:
            logger.warning(
                "SAGE_DIAG call=%d path=%s prefix=%s q=%s/%.6g k=%s/%.6g "
                "v=%s/%.6g quant=%s out=%s/%.6g",
                call,
                path,
                self.prefix,
                q_finite,
                q_max,
                k_finite,
                k_max,
                v_finite,
                v_max,
                quant_finite,
                out_finite,
                out_max,
            )
            return

        ref_finite, ref_max = self._finite_and_absmax(reference)
        out_float = out.float()
        ref_float = reference.float()
        dot = torch.dot(out_float.flatten(), ref_float.flatten())
        denom = torch.linalg.vector_norm(out_float) * torch.linalg.vector_norm(ref_float)
        cosine = float(dot / denom.clamp_min(1e-20))
        error = (out_float - ref_float).abs()
        logger.warning(
            "SAGE_DIAG call=%d path=%s prefix=%s q=%s/%.6g k=%s/%.6g "
            "v=%s/%.6g quant=%s out=%s/%.6g ref=%s/%.6g "
            "cos=%.8f mae=%.8g maxerr=%.8g",
            call,
            path,
            self.prefix,
            q_finite,
            q_max,
            k_finite,
            k_max,
            v_finite,
            v_max,
            quant_finite,
            out_finite,
            out_max,
            ref_finite,
            ref_max,
            cosine,
            float(error.mean()),
            float(error.max()),
        )

    def _forward_sm100(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> torch.Tensor:
        original_key = key
        original_value = value
        batch_size, qo_len, num_heads, head_dim = query.shape
        kv_len = key.shape[1]
        num_kv_heads = key.shape[2]
        kv_repeat = num_heads // key.shape[2]

        # The cubin supports native GQA, so retain the original KV head count.
        # Pad only physical storage to select the coarser K-block-16 variant;
        # kv_lens below preserves the logical length and masks these values.
        physical_kv_len = (kv_len + 15) // 16 * 16
        sage_blk_q = 1
        sage_blk_k = 16
        sage_blk_v = 1
        use_triton = HAS_SAGE_TRITON_PREPROCESS and self.preprocess in {"triton", "auto"}
        if use_triton and not self.smooth_k:
            logger.warning_once(
                "Triton Sage preprocessing always smooths K; using the PyTorch "
                "path because smooth_k=False"
            )
            use_triton = False
        if self.preprocess == "triton" and not HAS_SAGE_TRITON_PREPROCESS:
            logger.warning_once(
                "Triton Sage preprocessing was requested but is unavailable; using PyTorch"
            )

        if use_triton:
            buffers = self._triton_buffers(query, key)
            q_int8, k_int8, v_fp8, q_sfs, k_sfs, v_sfs = preprocess_sage_sm100(
                query, key, value, buffers
            )
            # Per-channel V scales are consumed directly by the Sage cubin.
            v_inv_scale: float | torch.Tensor = 1.0
            preprocess_name = "triton"
        else:
            # SageAttention's default preprocessing subtracts the sequence-wise
            # K mean. For non-causal attention this shifts every logit in a row
            # equally, leaving softmax invariant while removing shared offsets.
            if self.smooth_k:
                key_mean = key.float().mean(dim=1, keepdim=True).to(key.dtype)
                key = key - key_mean
            kv_pad = physical_kv_len - kv_len
            if kv_pad:
                key = F.pad(key, (0, 0, 0, 0, 0, kv_pad))
                value = F.pad(value, (0, 0, 0, 0, 0, kv_pad))
            q_flat = query.reshape(batch_size * qo_len, num_heads, head_dim).contiguous()
            k_flat = key.reshape(
                batch_size * physical_kv_len, num_kv_heads, head_dim
            ).contiguous()
            v_flat = value.reshape(
                batch_size * physical_kv_len, num_kv_heads, head_dim
            ).contiguous()
            q_int8, q_sfs = self._quantize_int8_blocked(q_flat, sage_blk_q)
            k_int8, k_sfs = self._quantize_int8_blocked(k_flat, sage_blk_k)
            v_fp8, v_inv_scale = self._quantize_fp8(v_flat)
            # Non-None V scales select the V=FP8 Sage cubin. Global V
            # dequantization remains bmm2_scale on the legacy path.
            v_sfs = torch.ones(
                num_kv_heads * head_dim, dtype=torch.float32, device=query.device
            )
            preprocess_name = "torch"

        q_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=query.device) * qo_len
        kv_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=query.device) * physical_kv_len
        kv_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=query.device)
        workspace = self._workspace(query.device, self.workspace_size)
        if self.workspace_reset == "always":
            workspace.zero_()
        elif self.workspace_reset == "counter":
            # FlashInfer reserves 8192 batches * 256 heads * int32 counters.
            workspace[: 8192 * 256 * 4].zero_()

        if "sm100" not in type(self)._logged_kernel_paths:
            logger.info(
                "Executing FlashInfer TRT-LLM SM100 SageAttention kernel "
                "(shape=%s, kv_len=%d, physical_kv_len=%d, kv_heads=%d, "
                "kv_repeat=%d, q_block=%d, k_block=%d, smooth_k=%s, "
                "preprocess=%s, workspace_reset=%s)",
                tuple(query.shape),
                kv_len,
                physical_kv_len,
                num_kv_heads,
                kv_repeat,
                sage_blk_q,
                sage_blk_k,
                self.smooth_k,
                preprocess_name,
                self.workspace_reset,
            )
            type(self)._logged_kernel_paths.add("sm100")

        out = flashinfer.prefill.trtllm_ragged_attention_deepseek(
            q_int8,
            k_int8,
            v_fp8,
            workspace,
            kv_lens,
            qo_len,
            physical_kv_len,
            self.softmax_scale,
            v_inv_scale,
            -1,
            batch_size,
            -1,
            q_indptr,
            kv_indptr,
            False,
            False,
            False,
            sage_attn_sfs=(q_sfs, k_sfs, None, v_sfs),
            num_elts_per_sage_attn_blk=(sage_blk_q, sage_blk_k, 0, sage_blk_v),
        )
        out = out.reshape(batch_size, qo_len, num_heads, head_dim)
        out = out if out.dtype == query.dtype else out.to(query.dtype)
        if self.diagnostics:
            reference = self._dense_fallback.forward_cuda(
                query, original_key, original_value, attn_metadata
            )
            self._log_tensor_diagnostics(
                path="sm100-sage",
                query=query,
                key=original_key,
                value=original_value,
                out=out,
                reference=reference,
                quantized=(q_int8, k_int8, v_fp8, q_sfs, k_sfs, v_sfs),
            )
        return out

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        reason = self._common_ineligibility_reason(query, key, value, attn_metadata)
        if reason is not None:
            return self._fallback(reason, query, key, value, attn_metadata)

        capability = torch.cuda.get_device_capability(query.device)
        if capability == (10, 0):
            reason = self._sm100_ineligibility_reason(query, key, value)
            if reason is None:
                return self._forward_sm100(query, key, value, attn_metadata)
        else:
            reason = f"no experimental FlashInfer Sage path for sm{capability[0]}{capability[1]}"
        return self._fallback(reason, query, key, value, attn_metadata)
