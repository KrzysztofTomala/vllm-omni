# Alpamayo 1.5

This integration runs the Cosmos/Qwen3-VL reasoning rollout and Alpamayo action
expert in one vLLM-Omni worker. The initial target is one GPU with BF16 weights.

Prepare an NPZ containing `image_frames`, `camera_indices`,
`ego_history_xyz` (16 × 3), and `ego_history_rot` (16 × 3 × 3), then run:

```bash
python examples/offline_inference/alpamayo1_5/end2end.py \
  --input observation.npz
```

For visual question answering, add `--question "What is blocking the lane?"`.
The standard OpenAI `/v1/chat/completions` endpoint can also be used for VQA.

To benchmark against the sample shipped with the Alpamayo NIM repository:

```bash
python examples/offline_inference/alpamayo1_5/benchmark_sample.py \
  --sample-dir ../cosmos-genai-nim-alpamayo-nim/alpamayo-tensorrt-llm/sample_data \
  --out alpamayo-benchmark.json --runs 3
```

The benchmark uses the matched BF16, seed-42, top-k-1 settings. Add
`--no-stable-uuids --vary-images` to measure the live-camera cache-miss path.
Trajectory requests also mirror the CUDA RNG consumed by the reference HF VLM
sampling loop before drawing the initial action-expert noise. The expanded
multimodal prompt length is captured during prefill, so the number of RNG
advances stays aligned when reasoning lengths differ. Use
`--return-action-noise` to record the initial-noise SHA-256 and first values in
the sample benchmark JSON.
The JSON result records the hostname, GPU UUID, driver, PyTorch/CUDA versions,
and container identifier when available. Compare warmed runs using identical
image-cache semantics; the first request includes one-time processor and kernel
initialization and is not representative of steady-state latency.

On the `alpamayo-1.5-nim-fast` branch, add `--compile-actions
--profile-actions` to compile the action expert and record its KV extraction,
sampler setup, expert integration, and trajectory-decoder times. Compilation is
expensive on the first request; compare warmed requests only.
Add `--manual-action-cudagraph` to capture and replay the fixed-shape ten-step
action integration using persistent input and KV-cache buffers. The fast path
pads the expert cache and attention mask to 3,328 tokens, matching the NIM
default, so one graph can serve different reasoning-prefix lengths. Override
this with `--static-expert-cache-max-len` when benchmarking a different maximum
sequence length.
The fast branch's bundled deploy config enables both optimizations for OpenPI
serving. Remove `compile_actions` and `manual_action_cudagraph` from
`policy_server_config` to run the eager action path.

Online policy serving uses the OpenPI-compatible websocket endpoint:

```bash
vllm serve nvidia/Alpamayo-1.5-10B --omni \
  --deploy-config vllm_omni/deploy/alpamayo1_5.yaml
```

Connect to `/v1/realtime/robot/openpi`. Each observation uses the same four NPZ
fields (as ndarray values), with optional `nav_text` and
`num_frames_per_camera`. The structured reply contains `actions` (six 64 × 3
ego-frame trajectory samples by default), `rotations`, `normalized_controls`,
and the generated reasoning text.

Current scope: single request at a time, TP=1, BF16, trajectory generation and
VQA. Classifier-free guidance and tensor-parallel action-expert execution are
not enabled in this first implementation. Experimental model-specific action
compilation and fused preprocessing optimizations are intentionally kept out of
this upstream-oriented implementation.
