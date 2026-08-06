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

## TensorRT-LLM NIM comparison

The TensorRT-LLM NIM was rerun on the same H100 on 2026-08-06 using the same
checkpoint, scene, BF16 precision, K=1, top-k=1, and seed 42. Seven stable warm
requests were retained after discarding the first post-readiness client request,
which had a one-time transport/setup delay despite normal server timing.

| TensorRT-LLM component | Mean | Std. dev. |
|---|---:|---:|
| Request decode | 7.204 ms | 0.222 ms |
| Model-input preparation | 12.544 ms | 0.179 ms |
| VLM rollout | 1,134.569 ms | 4.832 ms |
| Native KV capture | 34.544 ms | 0.258 ms |
| Expert diffusion | 204.111 ms | 1.622 ms |
| Trajectory postprocess | 2.035 ms | 0.052 ms |
| Inference core | 1,371.223 ms | 5.692 ms |
| Backend total | 1,378.454 ms | 5.737 ms |
| gRPC wall | **1,398.571 ms** | **6.241 ms** |

The TRT VLM TTFT was 795.205 ms and its post-first-token decode span was
314.155 ms. The expert's ten diffusion steps took 183.855 ms inside a 204.111
ms expert total; static-cache setup took 13.162 ms.

Against TRT end to end, the exact vLLM path is 183.4 ms (13.1%) slower and the
compiled vLLM path is 103.4 ms (7.4%) slower. The largest exact-path gaps are
request preparation (about 65 ms) and expert diffusion (107.9 ms); VLM rollout
is only about 18.4 ms slower. The compiled expert closes its gap to 25.5 ms.

## TRT-gap optimization follow-up

The request-preparation gap was traced to the container runtime rather than the
shared decoder code. On the same 24 JPEGs, upstream torchvision 0.26 in the
vLLM image took 80.25 ms, while the NVIDIA torchvision build in the TRT image
took 5.52 ms. The NIM integration now uses `nvImageCodec` with an automatic
torchvision/CPU fallback. Its isolated vLLM-image time was 8.40 ms including
code-stream construction and contiguous CHW conversion, with pixels exactly
equal to the torchvision CUDA result.

The expert cache profile was also reduced from 4,800 to 4,660 tokens, matching
the standard 4,594-token expert prefix plus its 64-token suffix. Seven stable
warm requests with the combined nvImageCodec, compiled expert, manual CUDA
graph, and 4,660-token cache produced:

| Optimized vLLM component | Mean |
|---|---:|
| Scene preparation | 9.192 ms |
| Adapter build | 1.732 ms |
| TTFT | 802.500 ms |
| Expert KV materialization | 10.734 ms |
| Expert diffusion | 225.732 ms |
| vLLM generate | 1,382.411 ms |
| Backend total | 1,393.858 ms |
| gRPC wall | **1,409 ms** |

This is 10.4 ms (0.74%) slower than the same-node TRT gRPC mean of 1,398.6 ms.
The observed vLLM range was 1,401--1,425 ms and the TRT range was
1,390--1,407 ms. The decoder and cache-size changes did not alter the compiled
output: its delta from the exact path remains 0.011449 m ADE, 0.035210 m
FDE/maximum point error, and `1.063868e-4` maximum rotation-matrix difference.

Cold TRT startup for this fresh local cache was 532.6 seconds through startup
warmup, including 75.4 seconds for the first VLM export, 97.2 seconds in the
FlashInfer-prewarm stage, 289.6 seconds for TRT initialization, and 66.4 seconds
for startup warmup/initial expert graph capture. These costs are excluded from
the warm measurements.

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
