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

The policy runs as two stages on one GPU. Stage 0 decodes the Qwen3-VL
reasoning natively in vLLM and ends the request after `<|traj_future_start|>`.
Its KV cache is handed to stage 1, a diffusion stage that runs the
flow-matching action expert over it and returns the trajectory. Requests that
reach stage 1 together share one expert batch.

| Item | Contract |
| --- | --- |
| Images | Up to 7 cameras x 4 frames, camera-major, cameras sorted by index (0 cross left, 1 front wide, 2 cross right, 3 rear left, 4 rear tele, 5 rear right, 6 front tele); resized to the checkpoint's 163,840–196,608 pixel budget |
| Ego history | 16 poses at 10 Hz ending at t0, in the ego frame at t0: `ego_history_xyz` `(16, 3)`, `ego_history_rot` `(16, 3, 3)` |
| Output | Stage 0: the reasoning tokens. Stage 1: `actions` `(64, 3)` ego positions and `rotations` `(64, 3, 3)` at 10 Hz in the same frame |
| Deployment | BF16, TP=1, both stages on one GPU, up to 6 requests in flight, eager ([`alpamayo2_super.yaml`](../../vllm_omni/deploy/alpamayo2_super.yaml)) |

## References

- Model card: <https://huggingface.co/nvidia/Alpamayo2-Super>
- Reference implementation and data loader: <https://github.com/NVlabs/alpamayo2>
- Dataset: <https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles>
- Reasoning stage: [`vllm_omni/model_executor/models/alpamayo2_super`](../../vllm_omni/model_executor/models/alpamayo2_super)
- Expert stage: [`vllm_omni/diffusion/models/alpamayo2_super`](../../vllm_omni/diffusion/models/alpamayo2_super)
- Tests: [`tests/model_executor/models/test_alpamayo2_super.py`](../../tests/model_executor/models/test_alpamayo2_super.py)

## Hardware

- Accelerator model and per-device memory: NVIDIA H200 NVL, 141 GB
- Number of devices: 1
- Device interconnect: not used (single GPU)
- Qualification scope: BF16, both stages on one GPU

## Software environment

- OS: Linux (x86_64)
- Python: 3.12
- Driver / runtime: CUDA 13.0
- vLLM version: 0.31.0
- vLLM-Omni version or commit: `main` at `d93f97dc` or later

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
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
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
reasoning_params = SamplingParams(
    temperature=0.6,
    top_p=0.98,
    max_tokens=256,
    seed=42,
    # Trajectory tokens are outside the stage tokenizer; decode reasoning with `tokenizer`.
    detokenize=False,
    stop_token_ids=[config.traj_ids["future_end"]],
)
expert_params = OmniDiffusionSamplingParams(
    seed=42,
    num_inference_steps=10,
    extra_args={"robot_obs": {"ego_history_xyz": history_xyz.tolist(), "ego_history_rot": history_rot.tolist()}},
)

omni = Omni(model=MODEL)
outputs = {output.stage_id: output for output in omni.generate(prompt, [reasoning_params, expert_params], use_tqdm=False)}
print(tokenizer.decode(outputs[0].outputs[0].token_ids, skip_special_tokens=True))
trajectory = outputs[1].multimodal_output["actions"]
print(trajectory[-1])  # position 6.4 s ahead
omni.close()
```

Each request returns one trajectory. To sample K trajectories, submit K
requests with different seeds; requests in flight decode together and share
expert batches.

## Verification

After the offline example, check the output:

```python
assert trajectory.shape == (64, 3) and trajectory.isfinite().all()
assert outputs[1].multimodal_output["rotations"].shape == (64, 3, 3)
```

The decoded reasoning is one or two short driving sentences, for example
"Stop to keep distance to the lead vehicle.", and the last position is a
plausible distance ahead of the ego vehicle for the scene (meters, x forward).

## Notes

- Memory usage: about 67 GiB of BF16 weights for stage 0, which uses
  `gpu_memory_utilization: 0.6` including its KV cache; stage 1 holds the
  2.4B-parameter expert and the prefixes of its current batch.
- KV handoff: the deploy config uses `NixlConnector`, which moves each
  request's KV prefix (about 0.8 GB for 3k tokens) GPU to GPU with UCX's CUDA
  IPC transport. It sets `UCX_CUDA_IPC_ENABLE_GET_ZCOPY=y` for both stages;
  without it UCX falls back to TCP on GPUs without NVLink, and each handoff
  takes about a second instead of a millisecond. NIXL with its UCX backend is
  required.
- Known limitations: the navigation-instruction (classifier-free guidance) and
  text-task modes of the reference implementation are not supported.

## Supported features

| Feature | Status |
| --- | --- |
| Offline inference | Supported |
| Online serving | Not supported |
| Tensor parallelism | Not supported (TP=1) |
| CUDA graphs | Not supported (eager) |
| Quantization | Not supported (BF16) |

See [feature compatibility](../../docs/user_guide/feature_compatibility.md)
for how these features combine across models.
