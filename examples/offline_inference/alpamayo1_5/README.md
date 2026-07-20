# Alpamayo 1.5

This integration runs the Cosmos/Qwen3-VL reasoning rollout and Alpamayo action
expert in one vLLM-Omni worker. The initial target is one GPU with BF16 weights.

Prepare an NPZ containing `image_frames`, `camera_indices`,
`ego_history_xyz` (16 × 3), and `ego_history_rot` (16 × 3 × 3), then run:

```bash
python examples/offline_inference/alpamayo1_5/end2end.py \
  --input observation.npz
```

To benchmark against the sample shipped with the Alpamayo NIM repository:

```bash
python examples/offline_inference/alpamayo1_5/benchmark_sample.py \
  --sample-dir ../cosmos-genai-nim-alpamayo-nim/alpamayo-tensorrt-llm/sample_data \
  --out alpamayo-benchmark.json --runs 3
```

The benchmark uses matched BF16, seed-42, top-k-1 settings and changes one
pixel per frame on every run. The JSON records environment details, reasoning,
trajectories, JPEG decode time, and end-to-end latency. Compare warmed requests;
the first request includes one-time processor and kernel initialization.

Online policy serving uses the OpenPI-compatible websocket endpoint:

```bash
vllm serve nvidia/Alpamayo-1.5-10B --omni \
  --deploy-config vllm_omni/deploy/alpamayo1_5.yaml
```

Connect to `/v1/realtime/robot/openpi`. Each observation uses the same four NPZ
fields (as ndarray values), with optional `nav_text` and
`num_frames_per_camera`. The reply contains `actions` (six 64 × 3 ego-frame
trajectory samples by default). Offline output additionally preserves rotations,
normalized controls, and generated reasoning in `multimodal_output`.

Current scope: single request at a time, TP=1, BF16, and trajectory generation.
The Transformers rollout and dynamic KV cache stay inside the Alpamayo policy
pipeline, following the existing GR00T policy integration. Classifier-free
guidance and tensor-parallel action-expert execution are not enabled.
