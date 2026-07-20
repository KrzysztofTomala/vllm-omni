#!/usr/bin/env python3
"""Benchmark the NIM sample request through the offline Omni API."""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.io import ImageReadMode, decode_jpeg
from transformers import AutoTokenizer
from vllm import SamplingParams

from vllm_omni import Omni
from vllm_omni.model_executor.models.alpamayo1_5.processing import (
    create_policy_messages,
    fuse_history_tokens,
)
from vllm_omni.model_executor.models.alpamayo1_5.tokenizer import ensure_extended_tokenizer

CAMERAS = ("cross_left", "front_wide", "cross_right", "front_tele")
CAMERA_INDICES = (0, 1, 2, 6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--stable-uuids", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--vary-images",
        action="store_true",
        help="Change one pixel per run to measure live-frame cache misses.",
    )
    parser.add_argument("--profile-input", action="store_true")
    parser.add_argument("--profile-actions", action="store_true")
    parser.add_argument("--compile-actions", action="store_true")
    parser.add_argument(
        "--image-input",
        choices=("pil", "cpu-tensor", "jpeg-cpu-tensor"),
        default="pil",
        help=(
            "Representation passed to vLLM. jpeg-cpu-tensor includes threaded "
            "JPEG decode in every recorded request wall time."
        ),
    )
    parser.add_argument(
        "--processor-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device used by the HF image processor.",
    )
    parser.add_argument("--model", default="nvidia/Alpamayo-1.5-10B")
    parser.add_argument("--deploy-config", default="vllm_omni/deploy/alpamayo1_5.yaml")
    return parser.parse_args()


def float_array(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def main() -> None:
    args = parse_args()
    image_paths = [args.sample_dir / "images" / f"{camera}_t{frame}.jpg" for camera in CAMERAS for frame in range(4)]
    encoded_images: list[torch.Tensor] | None = None
    decode_pool: ThreadPoolExecutor | None = None
    if args.image_input == "pil":
        images = []
        for path in image_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
    else:
        encoded_images = [
            torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint8)
            for path in image_paths
        ]
        decode_pool = ThreadPoolExecutor(max_workers=len(encoded_images))
        images = list(
            decode_pool.map(
                lambda item: decode_jpeg(item, mode=ImageReadMode.RGB),
                encoded_images,
            )
        )
        if args.image_input == "cpu-tensor":
            decode_pool.shutdown()
            decode_pool = None
    with (args.sample_dir / "egomotion.json").open(encoding="utf-8") as file:
        egomotion = json.load(file)
    history_xyz = np.asarray(egomotion["ego_history_xyz"], dtype=np.float32)
    history_rot = np.asarray(egomotion["ego_history_rot"], dtype=np.float32)

    tokenizer = AutoTokenizer.from_pretrained(ensure_extended_tokenizer())
    messages = create_policy_messages(
        images,
        camera_indices=CAMERA_INDICES,
        navigation="Drive forward.",
    )
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )
    prompt_ids = fuse_history_tokens(
        torch.tensor(tokenizer.encode(prompt)), torch.from_numpy(history_xyz)
    ).tolist()
    prompt_input = {
        "prompt_token_ids": prompt_ids,
        "multi_modal_data": {"image": images},
        "mm_processor_kwargs": {"device": args.processor_device},
    }
    if args.stable_uuids:
        prompt_input["multi_modal_uuids"] = {
            "image": [f"alpamayo-sample:{path.stem}" for path in image_paths]
        }
    sampling = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        max_tokens=256,
        stop_token_ids=[155683],
        seed=42,
        extra_args={
            "robot_obs": {
                "ego_history_xyz": history_xyz.tolist(),
                "ego_history_rot": history_rot.tolist(),
            },
            "num_traj_samples": 1,
            "diffusion_steps": 10,
            "action_temperature": 1.0,
            "_profile_timings": args.profile_actions,
            "_compile_expert": args.compile_actions,
        },
    )

    startup = time.perf_counter()
    omni = Omni(model=args.model, deploy_config=args.deploy_config, log_stats=True)
    startup_s = time.perf_counter() - startup
    gpu_memory_mib = None
    try:
        memory_output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        gpu_memory_mib = sum(int(value.strip()) for value in memory_output.splitlines() if value.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    processor_times: list[float] = []
    input_processor = omni.engine.input_processor
    assert input_processor is not None
    original_process_inputs = input_processor.process_inputs

    def timed_process_inputs(*process_args: object, **process_kwargs: object) -> object:
        started = time.perf_counter()
        if args.profile_input and not processor_times:
            profiler = cProfile.Profile()
            result = profiler.runcall(original_process_inputs, *process_args, **process_kwargs)
            report = io.StringIO()
            pstats.Stats(profiler, stream=report).sort_stats("cumulative").print_stats(40)
            print(report.getvalue(), flush=True)
        else:
            result = original_process_inputs(*process_args, **process_kwargs)
        processor_times.append(time.perf_counter() - started)
        return result

    input_processor.process_inputs = timed_process_inputs  # type: ignore[method-assign]
    rows = []
    try:
        for run_index in range(args.runs):
            request_started = time.perf_counter()
            jpeg_decode_s = 0.0
            request_images = images
            if args.image_input == "jpeg-cpu-tensor":
                assert encoded_images is not None and decode_pool is not None
                decode_started = time.perf_counter()
                request_images = list(
                    decode_pool.map(
                        lambda item: decode_jpeg(item, mode=ImageReadMode.RGB),
                        encoded_images,
                    )
                )
                jpeg_decode_s = time.perf_counter() - decode_started
            run_prompt = prompt_input
            if args.vary_images:
                run_images = []
                for image_index, image in enumerate(request_images):
                    if isinstance(image, torch.Tensor):
                        varied = image.clone()
                        channel = image_index % int(varied.shape[0])
                        varied[channel, 0, 0] = (
                            int(varied[channel, 0, 0]) + run_index + image_index + 1
                        ) % 256
                    else:
                        varied = image.copy()
                        pixel = list(varied.getpixel((0, 0)))
                        channel = image_index % len(pixel)
                        pixel[channel] = (pixel[channel] + run_index + image_index + 1) % 256
                        varied.putpixel((0, 0), tuple(pixel))
                    run_images.append(varied)
                run_prompt = {
                    "prompt_token_ids": prompt_ids,
                    "multi_modal_data": {"image": run_images},
                    "mm_processor_kwargs": {"device": args.processor_device},
                }
                if args.stable_uuids:
                    # Unique IDs bypass expensive content hashing while still
                    # forcing a cache miss for every live frame on every run.
                    run_prompt["multi_modal_uuids"] = {
                        "image": [
                            f"alpamayo-sample:{path.stem}:run-{run_index}"
                            for path in image_paths
                        ]
                    }
            started = time.perf_counter()
            outputs = list(omni.generate([run_prompt], [sampling], use_tqdm=False))
            wall_s = time.perf_counter() - started
            end_to_end_s = time.perf_counter() - request_started
            output = getattr(outputs[-1], "request_output", outputs[-1])
            candidate = output.outputs[0]
            multimodal = getattr(output, "multimodal_output", None)
            if multimodal is None:
                multimodal = getattr(candidate, "multimodal_output", None)
            if multimodal is None:
                raise RuntimeError("request produced no trajectory")
            rows.append(
                {
                    "wall_s": wall_s,
                    "end_to_end_s": end_to_end_s,
                    "jpeg_decode_s": jpeg_decode_s,
                    "process_inputs_s": processor_times[-1],
                    "reasoning_text": candidate.text,
                    "actions": float_array(multimodal["actions"]).tolist(),
                    "action_profile_ms": (
                        float_array(multimodal["profile_timings_ms"]).tolist()
                        if "profile_timings_ms" in multimodal
                        else None
                    ),
                }
            )
            print(
                f"run={run_index} end_to_end_s={end_to_end_s:.6f} "
                f"jpeg_decode_s={jpeg_decode_s:.6f} wall_s={wall_s:.6f} "
                f"process_inputs_s={processor_times[-1]:.6f}",
                flush=True,
            )
    finally:
        if decode_pool is not None:
            decode_pool.shutdown()
        omni.close()

    gpu_details = None
    try:
        gpu_details = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).splitlines()
    except (OSError, subprocess.SubprocessError):
        pass

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "startup_s": startup_s,
                "hostname": socket.gethostname(),
                "gpu_details": gpu_details,
                "gpu_memory_mib": gpu_memory_mib,
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "container_image": os.environ.get("NVIDIA_BUILD_ID")
                or os.environ.get("VLLM_IMAGE_TAG"),
                "stable_uuids": args.stable_uuids,
                "vary_images": args.vary_images,
                "image_input": args.image_input,
                "processor_device": args.processor_device,
                "prompt_tokens_before_image_expansion": len(prompt_ids),
                "runs": rows,
            },
            file,
            indent=2,
        )


if __name__ == "__main__":
    main()
