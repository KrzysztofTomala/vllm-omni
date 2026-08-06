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
The JSON result records the hostname, GPU UUID, driver, PyTorch/CUDA versions,
and container identifier when available. Compare warmed runs using identical
image-cache semantics; the first request includes one-time processor and kernel
initialization and is not representative of steady-state latency.

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
