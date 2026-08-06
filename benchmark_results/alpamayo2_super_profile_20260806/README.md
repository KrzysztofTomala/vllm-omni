# Alpamayo 2 Super vLLM-Omni fast-path benchmark

Date: 2026-08-06

## Configuration

- GPU: NVIDIA H100 NVL (single GPU)
- Precision: BF16
- Request: the same Alpamayo 2 Super sample and checkpoint for every variant
- Sampling: seed 42, `num_traj_samples=1`
- Prompt/generation length: 4,580 prompt tokens and 15 generated tokens
- Warm statistics: six requests unless noted otherwise
- vLLM-Omni commit: `2c2a2829` (`alpamayo2-super-vllm-omni-fast`)
- NIM integration branch: `ktomala/alpamayo2-super-vllm-omni`

The vLLM wall measurements include the complete warm gRPC request. The OSS reference was
measured in-process, so comparing vLLM gRPC wall time with OSS core time is conservative.

## Warm latency results

| Variant | gRPC wall mean | Backend | vLLM generate | Expert diffusion | Output vs dynamic |
|---|---:|---:|---:|---:|---|
| Dynamic baseline | 1.860 s | 1.839 s | 1.754 s | 595.8 ms | Reference |
| Static cache, 8,256 | 1.872 s | - | - | 601.0 ms | Bitwise exact |
| CUDA graph, 8,256 | 1.721 s | - | - | 449.2 ms | Bitwise exact |
| `torch.compile` + graph, 8,256 | 1.620 s | - | - | 352.6 ms | Max XYZ delta 0.0312 m |
| CUDA graph, 4,800 | **1.582 s** | 1.562 s | 1.478 s | **312.0 ms** | **Bitwise exact** |
| `torch.compile` + graph, 4,800 | **1.502 s** | 1.482 s | 1.397 s | **229.6 ms** | ADE 0.0114 m; FDE/max 0.0352 m |

The selected exact-output default is the 4,800-token CUDA-graph path. It falls back to a
larger shape for longer prompts. The compiled 4,800-token path is an optional faster mode;
its first request incurred about 67 seconds of compilation/capture and it introduces small
numeric drift (maximum rotation-matrix absolute difference `1.064e-4`).

The exact path was also repeated twice with seed 43: the repetitions were bitwise identical
to each other and differed from seed 42 as expected.

## OSS BF16 reference and estimated speedup

Five warm direct in-process OSS measurements on the same H100 NVL produced:

| OSS component | Mean | Std. dev. |
|---|---:|---:|
| Input preparation | 0.3771 s | 0.0108 s |
| VLM rollout | 1.7886 s | 0.0413 s |
| Expert diffusion | 0.6591 s | 0.0250 s |
| Model total | 2.4521 s | 0.0616 s |
| Core total | 2.8292 s | 0.0649 s |

Relative to the OSS core total:

| vLLM mode | Comparable latency | Speedup | Latency reduction |
|---|---:|---:|---:|
| Exact graph 4,800, complete gRPC request | 1.582 s | **1.79x** | **44.1%** |
| Compiled graph 4,800, complete gRPC request | 1.502 s | **1.88x** | **46.9%** |

At the component level, the compiled vLLM generate stage is 1.76x faster than the OSS model
total (1.397 vs 2.452 seconds), while compiled expert diffusion is 2.87x faster than the OSS
expert (0.230 vs 0.659 seconds). The exact graph expert is 2.11x faster (0.312 seconds).

These ratios are estimates from one sample and one H100, not a throughput benchmark. A
multi-example evaluation is still required before treating them as release-level numbers.

## Historical TensorRT-LLM NIM reference

The prior TensorRT-LLM NIM measurement on this H100 was approximately 1.432 seconds warm gRPC
latency with approximately 204 ms in expert diffusion. Thus the compiled vLLM path is within
about 70 ms (4.9%) end to end, while the selected exact-output path is about 150 ms (10.5%)
slower. This historical result was not rerun as part of this matrix.

## Raw artifacts

The full request outputs and per-request timing payloads remain local in this directory:

- `dynamic_baseline.json`
- `static_cold.json`, `static_warm.json`
- `graph_only_cold.json`, `graph_only_warm.json`
- `fast_cold.json`, `fast_warm.json`
- `graph_4800_cold.json`, `graph_4800_warm.json`
- `compiled_4800_cold.json`, `compiled_4800_warm.json`
- `graph_seed43_a.json`, `graph_seed43_b.json`

Only this compact report and `summary.json` are tracked in Git because the raw files contain
large trajectory and rotation payloads.
