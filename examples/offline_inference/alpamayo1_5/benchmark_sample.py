#!/usr/bin/env python3
"""Benchmark Alpamayo on the scene distributed with the NIM implementation."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torchvision.io import ImageReadMode, decode_image

from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

CAMERAS = ("cross_left", "front_wide", "cross_right", "front_tele")
CAMERA_INDICES = (0, 1, 2, 6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--model", default="nvidia/Alpamayo-1.5-10B")
    parser.add_argument("--deploy-config", default="vllm_omni/deploy/alpamayo1_5.yaml")
    parser.add_argument("--navigation", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = []
    for camera in CAMERAS:
        for frame in range(4):
            stem = args.sample_dir / "images" / f"{camera}_t{frame}"
            path = next(
                (candidate for suffix in (".jpg", ".png") if (candidate := stem.with_suffix(suffix)).is_file()),
                None,
            )
            if path is None:
                raise FileNotFoundError(f"missing {stem}.jpg or {stem}.png")
            image_paths.append(path)
    encoded = [torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint8) for path in image_paths]
    with (args.sample_dir / "egomotion.json").open(encoding="utf-8") as file:
        egomotion = json.load(file)
    decode_pool = ThreadPoolExecutor(max_workers=len(encoded))
    sampling = OmniDiffusionSamplingParams(
        extra_args={
            "num_traj_samples": 1,
            "diffusion_steps": 10,
            "action_temperature": 1.0,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 1,
            "max_tokens": 256,
            "seed": 42,
        }
    )
    startup = time.perf_counter()
    omni = Omni(model=args.model, deploy_config=args.deploy_config, log_stats=True)
    startup_s = time.perf_counter() - startup
    rows = []
    try:
        for run_index in range(args.runs):
            request_started = time.perf_counter()
            decode_started = time.perf_counter()
            images = list(
                decode_pool.map(
                    lambda item: decode_image(item, mode=ImageReadMode.RGB),
                    encoded,
                )
            )
            jpeg_decode_s = time.perf_counter() - decode_started
            sampling.extra_args["robot_obs"] = {
                "image_frames": torch.stack(images),
                "camera_indices": CAMERA_INDICES,
                "ego_history_xyz": egomotion["ego_history_xyz"],
                "ego_history_rot": egomotion["ego_history_rot"],
                "navigation": args.navigation,
            }
            infer_started = time.perf_counter()
            outputs = omni.generate(args.navigation or "", sampling_params_list=[sampling])
            infer_s = time.perf_counter() - infer_started
            end_to_end_s = time.perf_counter() - request_started
            if not outputs:
                raise RuntimeError("request produced no trajectory")
            multimodal = outputs[0].multimodal_output or {}
            rows.append(
                {
                    "inference_s": infer_s,
                    "end_to_end_s": end_to_end_s,
                    "jpeg_decode_s": jpeg_decode_s,
                    "reasoning_text": multimodal.get("reasoning", ""),
                    "actions": np.asarray(multimodal["actions"], dtype=np.float32).tolist(),
                }
            )
            print(
                f"run={run_index} end_to_end_s={end_to_end_s:.6f} "
                f"jpeg_decode_s={jpeg_decode_s:.6f} inference_s={infer_s:.6f}",
                flush=True,
            )
    finally:
        decode_pool.shutdown()
        omni.close()

    try:
        gpu_details = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
            text=True,
        ).splitlines()
    except (OSError, subprocess.SubprocessError):
        gpu_details = None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "startup_s": startup_s,
                "hostname": socket.gethostname(),
                "gpu_details": gpu_details,
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "container_image": os.environ.get("NVIDIA_BUILD_ID") or os.environ.get("VLLM_IMAGE_TAG"),
                "runs": rows,
            },
            file,
            indent=2,
        )


if __name__ == "__main__":
    main()
