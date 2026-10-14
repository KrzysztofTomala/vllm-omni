# Alpamayo 2 Super — H200

> NVIDIA Alpamayo 2 Super driving policy: chain-of-thought reasoning and ego
> trajectory prediction from multi-camera video on one H200

## Summary

- Vendor: NVIDIA
- Model: `nvidia/Alpamayo2-Super`
- Task: Ego trajectory prediction with chain-of-thought reasoning
- Mode: Offline inference (`Omni`)
- Hardware: 1x NVIDIA H200 NVL 141 GB
- Maintainer: Community

## When to use this recipe

Use this recipe to predict a 6.4-second ego trajectory, together with the
model's driving reasoning, from a few seconds of multi-camera driving video and
the recent ego motion.

## Supported model contract

The Qwen3-VL backbone decodes reasoning natively in vLLM. When it emits
`<|traj_future_start|>`, the model runs the flow-matching action expert over the
request's paged KV cache, returns the trajectory as multimodal output, and ends
the request with `<|traj_future_end|>`.

| Item | Contract |
| --- | --- |
| Images | Up to 7 cameras x 4 frames, camera-major, cameras sorted by index (0 cross left, 1 front wide, 2 cross right, 3 rear left, 4 rear tele, 5 rear right, 6 front tele); resized to the checkpoint's 163,840–196,608 pixel budget |
| Ego history | 16 poses at 10 Hz ending at t0, in the ego frame at t0: `ego_history_xyz` `(16, 3)`, `ego_history_rot` `(16, 3, 3)` |
| Output | `actions` `(64, 3)` ego positions and `rotations` `(64, 3, 3)` at 10 Hz in the same frame, plus the reasoning tokens |
| Deployment | BF16, TP=1, `max_num_seqs: 1`, eager ([`alpamayo2_super.yaml`](../../vllm_omni/deploy/alpamayo2_super.yaml)) |

## References

- Model card: <https://huggingface.co/nvidia/Alpamayo2-Super>
- Reference implementation and data loader: <https://github.com/NVlabs/alpamayo2>
- Dataset: <https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles>
- Model: [`vllm_omni/model_executor/models/alpamayo2_super`](../../vllm_omni/model_executor/models/alpamayo2_super)
- Tests: [`tests/model_executor/models/test_alpamayo2_super.py`](../../tests/model_executor/models/test_alpamayo2_super.py)

## Hardware

- Accelerator model and per-device memory: NVIDIA H200 NVL, 141 GB
- Number of devices: 1
- Qualification scope: BF16, one request at a time

## Software environment

- OS: Linux (x86_64)
- Python: 3.12
- Driver / runtime: CUDA 13.0
- vLLM version: 0.31.0
- vLLM-Omni version or commit: `vllm/vllm-omni:nightly` image

## Command

### Input data

The examples load one scene from the PhysicalAI-AV dataset with the
NVlabs/alpamayo2 loader, which streams the clip from Hugging Face (accept the
dataset terms and run `hf auth login` first) and decodes four frames per camera
before `t0_us`:

```bash
pip install git+https://github.com/NVlabs/alpamayo2
```

```python
from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset

scene = load_physical_aiavdataset("030c760c-ae38-49aa-9ad8-f5650a545d26", t0_us=5_100_000)
frames = scene["image_frames"]  # (7, 4, 3, H, W) uint8
history_xyz = scene["ego_history_xyz"].reshape(16, 3)
history_rot = scene["ego_history_rot"].reshape(16, 3, 3)
```

The package is only used to load data; the model runs entirely in vLLM-Omni.

### Offline inference

```python
from vllm import SamplingParams

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.model_executor.models.alpamayo2_super import Alpamayo2SuperConfig
from vllm_omni.model_executor.models.alpamayo2_super.processing import (
    build_alpamayo2_super_tokenizer,
    build_trajectory_prompt,
)

MODEL = "nvidia/Alpamayo2-Super"
config = Alpamayo2SuperConfig.from_pretrained(MODEL)
tokenizer = build_alpamayo2_super_tokenizer(MODEL, config.history_vocab_size, config.future_vocab_size)
prompt = {
    "prompt_token_ids": build_trajectory_prompt(
        tokenizer,
        config,
        frames=frames,
        camera_indices=scene["camera_indices"].tolist(),
        ego_history_xyz=history_xyz,
    ),
    "multi_modal_data": {"image": list(frames.flatten(0, 1))},
}
params = SamplingParams(
    temperature=0.6,
    top_p=0.98,
    max_tokens=256,
    seed=42,
    # Trajectory tokens are outside the stage tokenizer; decode reasoning with `tokenizer`.
    detokenize=False,
    stop_token_ids=[config.traj_ids["future_end"]],
    extra_args={
        "robot_obs": {"ego_history_xyz": history_xyz.tolist(), "ego_history_rot": history_rot.tolist()},
        "action_seed": 42,
    },
)

omni = Omni(model=MODEL)
output = omni.generate(prompt, [params], use_tqdm=False)[0].outputs[0]
print(tokenizer.decode(output.token_ids, skip_special_tokens=True))
print(output.multimodal_output["actions"][-1])  # position 6.4 s ahead
omni.close()
```

Each request returns one trajectory. Sample more trajectories with more
requests and different seeds.

## Verification

The reasoning is one or two short driving sentences, and the trajectory has
64 finite points whose last position is a plausible distance ahead of the ego
vehicle for the scene (meters, x forward).

## Notes

- Memory usage: about 67 GiB of BF16 weights; the deploy config leaves the rest
  of `gpu_memory_utilization: 0.9` to the KV cache.
- Attention backend: FlashAttention or Triton. FlashInfer is not supported,
  because the action expert reads the request's KV through the attention
  backend's block table.
- Multiple trajectories: each request returns one trajectory, so sampling K
  trajectories takes K requests. They run one after another and reuse the
  cached image and prompt prefill, so latency grows linearly with K.
- Known limitations: one request at a time; the navigation-instruction
  (classifier-free guidance) and text-task modes of the reference
  implementation are not supported.

## Supported features

| Feature | Status |
| --- | --- |
| Offline inference | Supported |
| Online serving | Not supported |
| Tensor parallelism | Not supported (TP=1) |
| CUDA graphs | Not supported (eager) |
| Quantization | Not supported (BF16) |
