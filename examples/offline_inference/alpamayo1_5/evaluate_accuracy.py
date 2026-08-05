#!/usr/bin/env python3
"""Evaluate native-vLLM Alpamayo MinADE/MinFDE on the NIM corpus."""

from __future__ import annotations

import argparse
import csv
import json
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml
from torchvision.io import ImageReadMode, decode_image
from transformers import AutoTokenizer
from vllm import SamplingParams

from vllm_omni import Omni
from vllm_omni.model_executor.models.alpamayo1_5.processing import (
    create_policy_messages,
    fuse_history_tokens,
)
from vllm_omni.model_executor.models.alpamayo1_5.tokenizer import (
    ensure_extended_tokenizer,
)

CAMERAS = ("cross_left", "front_wide", "cross_right", "front_tele")
CAMERA_INDICES = (0, 1, 2, 6)
HORIZON = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int, default=200)
    parser.add_argument("--model", default="nvidia/Alpamayo-1.5-10B")
    parser.add_argument("--deploy-config", default="vllm_omni/deploy/alpamayo1_5.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile-actions", action="store_true")
    parser.add_argument("--manual-action-cudagraph", action="store_true")
    parser.add_argument("--profile-actions", action="store_true")
    parser.add_argument("--processor-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--static-expert-cache-max-len", type=int, default=3328)
    return parser.parse_args()


def image_paths(scene_dir: Path) -> list[Path]:
    paths = []
    for camera in CAMERAS:
        for frame in range(4):
            stem = scene_dir / "images" / f"{camera}_t{frame}"
            path = next(
                (candidate for suffix in (".jpg", ".png") if (candidate := stem.with_suffix(suffix)).is_file()),
                None,
            )
            if path is None:
                raise FileNotFoundError(f"missing {stem}.jpg or {stem}.png")
            paths.append(path)
    return paths


def trajectory_errors(actions: np.ndarray, ground_truth: np.ndarray) -> tuple[float, float]:
    displacement = np.linalg.norm(actions[:, :HORIZON, :2] - ground_truth[None, :HORIZON, :2], axis=-1)
    return float(displacement.mean(axis=1).min()), float(displacement[:, -1].min())


def as_numpy(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def decode_scene(scene_dir: Path, pool: ThreadPoolExecutor) -> tuple[list[torch.Tensor], list[Path]]:
    paths = image_paths(scene_dir)
    encoded = [torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint8) for path in paths]
    images = list(pool.map(lambda item: decode_image(item, mode=ImageReadMode.RGB), encoded))
    return images, paths


def main() -> None:
    args = parse_args()
    scenes = sorted(path for path in args.datadir.iterdir() if path.is_dir())[: args.max_scenes]
    if not scenes:
        raise ValueError(f"no scenes found in {args.datadir}")

    tokenizer = AutoTokenizer.from_pretrained(ensure_extended_tokenizer())
    print(f"Evaluating {len(scenes)} scenes K=1 top_k=1", flush=True)
    startup_started = time.perf_counter()
    omni = Omni(model=args.model, deploy_config=args.deploy_config, log_stats=False)
    startup_s = time.perf_counter() - startup_started
    rows: list[dict] = []
    args.workdir.mkdir(parents=True, exist_ok=True)
    csv_path = args.workdir / "per_scene.csv"
    csv_output = csv_path.open("w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(
        csv_output,
        fieldnames=(
            "scene",
            "min_ade_6s",
            "min_fde_6s",
            "latency_s",
            "preparation_s",
            "end_to_end_s",
            "process_inputs_s",
            "output_tokens",
            "action_profile_ms",
        ),
    )
    csv_writer.writeheader()
    csv_output.flush()
    processor_times: list[float] = []
    input_processor = omni.engine.input_processor
    assert input_processor is not None
    original_process_inputs = input_processor.process_inputs

    def timed_process_inputs(*process_args: object, **process_kwargs: object) -> object:
        started = time.perf_counter()
        result = original_process_inputs(*process_args, **process_kwargs)
        processor_times.append(time.perf_counter() - started)
        return result

    input_processor.process_inputs = timed_process_inputs  # type: ignore[method-assign]
    pool = ThreadPoolExecutor(max_workers=16)
    try:
        for index, scene_dir in enumerate(scenes, 1):
            request_started = time.perf_counter()
            images, paths = decode_scene(scene_dir, pool)
            egomotion = json.loads((scene_dir / "egomotion.json").read_text(encoding="utf-8"))
            history_xyz = np.asarray(egomotion["ego_history_xyz"], dtype=np.float32)
            history_rot = np.asarray(egomotion["ego_history_rot"], dtype=np.float32)
            messages = create_policy_messages(images, camera_indices=CAMERA_INDICES, navigation=None)
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
            )
            prompt_ids = fuse_history_tokens(
                torch.tensor(tokenizer.encode(prompt)),
                torch.from_numpy(history_xyz),
            ).tolist()
            prompt_input = {
                "prompt_token_ids": prompt_ids,
                "multi_modal_data": {"image": images},
                "mm_processor_kwargs": {"device": args.processor_device},
                "multi_modal_uuids": {"image": [f"alpamayo-eval:{scene_dir.name}:{path.stem}" for path in paths]},
            }
            sampling = SamplingParams(
                temperature=1.0,
                top_p=1.0,
                top_k=1,
                max_tokens=256,
                stop_token_ids=[155683],
                seed=args.seed,
                extra_args={
                    "robot_obs": {
                        "ego_history_xyz": history_xyz.tolist(),
                        "ego_history_rot": history_rot.tolist(),
                    },
                    "num_traj_samples": 1,
                    "diffusion_steps": 10,
                    "action_temperature": 1.0,
                    "_sampling_seed": args.seed,
                    "_nim_action_rng_compat": True,
                    "_profile_timings": args.profile_actions,
                    "_compile_expert": args.compile_actions,
                    "_manual_action_cudagraph": args.manual_action_cudagraph,
                    "_static_expert_cache_max_len": args.static_expert_cache_max_len,
                },
            )
            preparation_s = time.perf_counter() - request_started
            inference_started = time.perf_counter()
            outputs = omni.generate(prompt_input, sampling_params_list=[sampling])
            inference_s = time.perf_counter() - inference_started
            if not outputs:
                raise RuntimeError(f"{scene_dir.name}: request produced no output")
            output = getattr(outputs[0], "request_output", outputs[0])
            candidate = output.outputs[0]
            multimodal = getattr(output, "multimodal_output", None)
            if multimodal is None:
                multimodal = getattr(candidate, "multimodal_output", None)
            if multimodal is None:
                multimodal = outputs[0].multimodal_output
            actions = as_numpy(multimodal["actions"])
            output_tokens = len(getattr(candidate, "token_ids", ()) or ())
            action_profile = (
                as_numpy(multimodal["profile_timings_ms"]).tolist()
                if "profile_timings_ms" in multimodal
                else None
            )
            ground_truth = np.asarray(
                json.loads((scene_dir / "gt_trajectory.json").read_text(encoding="utf-8"))["ego_future_xyz"],
                dtype=np.float32,
            )
            min_ade, min_fde = trajectory_errors(actions, ground_truth)
            row = {
                "scene": scene_dir.name,
                "min_ade_6s": min_ade,
                "min_fde_6s": min_fde,
                "latency_s": inference_s,
                "preparation_s": preparation_s,
                "end_to_end_s": time.perf_counter() - request_started,
                "process_inputs_s": processor_times[-1],
                "output_tokens": output_tokens,
                "action_profile_ms": action_profile,
            }
            rows.append(row)
            csv_writer.writerow(row)
            csv_output.flush()
            print(
                f"[{index}/{len(scenes)}] MinADE={min_ade:.3f}m MinFDE={min_fde:.3f}m "
                f"inference={inference_s:.3f}s prep={preparation_s:.3f}s",
                flush=True,
            )
    finally:
        pool.shutdown()
        csv_output.close()
        omni.close()

    warm_rows = rows[1:] if len(rows) > 1 else rows
    summary = {
        "mean_minade_6s": float(np.mean([row["min_ade_6s"] for row in rows])),
        "mean_minfde_6s": float(np.mean([row["min_fde_6s"] for row in rows])),
        "success_rate": 1.0,
        "mean_latency_s": float(np.mean([row["latency_s"] for row in rows])),
        "mean_warm_latency_s": float(np.mean([row["latency_s"] for row in warm_rows])),
        "mean_preparation_s": float(np.mean([row["preparation_s"] for row in rows])),
        "mean_end_to_end_s": float(np.mean([row["end_to_end_s"] for row in rows])),
        "mean_warm_process_inputs_s": float(np.mean([row["process_inputs_s"] for row in warm_rows])),
        "mean_warm_output_tokens": float(np.mean([row["output_tokens"] for row in warm_rows])),
    }
    profiles = [row["action_profile_ms"] for row in warm_rows if row["action_profile_ms"] is not None]
    if profiles:
        summary["mean_warm_action_profile_ms"] = as_numpy(profiles).mean(axis=0).tolist()
    try:
        gpu_details = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], text=True
        ).splitlines()
    except (OSError, subprocess.SubprocessError):
        gpu_details = None
    result = {
        "benchmark_name": "alpamayo-1.5-accuracy",
        "backend": "vllm-omni-nim-fast",
        "configuration": {
            "k": 1,
            "top_k": 1,
            "seed": args.seed,
            "horizon_steps": HORIZON,
            "n_scenes": len(rows),
            "dtype": "bfloat16",
            "nim_action_rng_compat": True,
            "compile_expert": args.compile_actions,
            "manual_action_cudagraph": args.manual_action_cudagraph,
            "static_expert_cache_max_len": args.static_expert_cache_max_len,
            "processor_device": args.processor_device,
            "profile_actions": args.profile_actions,
        },
        "environment": {
            "startup_s": startup_s,
            "hostname": socket.gethostname(),
            "gpu_details": gpu_details,
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
        },
        "summary": summary,
        "per_scene": rows,
    }
    (args.workdir / "result.yaml").write_text(yaml.safe_dump(result, sort_keys=False), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
