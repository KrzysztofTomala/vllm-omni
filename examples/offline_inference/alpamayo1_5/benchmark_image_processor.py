#!/usr/bin/env python3
"""Benchmark Cosmos Qwen3-VL preprocessing from several image representations."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision.io import ImageReadMode, decode_jpeg
from transformers import AutoImageProcessor, AutoProcessor


def messages(images: list[Any]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    camera_names = ("Front left camera", "Front camera", "Front right camera", "Front telephoto camera")
    for index, image in enumerate(images):
        if index % 4 == 0:
            content.append({"type": "text", "text": f"{camera_names[index // 4]}: "})
        content.extend(
            ({"type": "text", "text": f"frame {index % 4} "}, {"type": "image", "image": image})
        )
    content.append({"type": "text", "text": "Describe the safe driving action."})
    return [
        {"role": "system", "content": [{"type": "text", "text": "You are a driving assistant."}]},
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": "Reasoning:"}]},
    ]


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.accelerator.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--model", default="nvidia/Cosmos-Reason2-8B")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    paths = sorted((*args.image_dir.glob("*.jpg"), *args.image_dir.glob("*.jpeg")))
    encoded = [torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint8) for path in paths]
    pil = [Image.open(path).convert("RGB") for path in paths]
    with ThreadPoolExecutor(max_workers=16) as pool:
        cpu = list(pool.map(lambda item: decode_jpeg(item, mode=ImageReadMode.RGB), encoded))
    cuda = decode_jpeg(encoded, mode=ImageReadMode.RGB, device="cuda")
    synchronize()

    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=163840,
        max_pixels=196608,
        local_files_only=True,
    )
    processor.image_processor = AutoImageProcessor.from_pretrained(
        args.model,
        min_pixels=163840,
        max_pixels=196608,
        local_files_only=True,
        backend="torchvision",
    )

    cases = {
        "pil_cpu_default": (pil, None),
        "cpu_tensor_default": (cpu, None),
        "cpu_tensor_cuda_processor": (cpu, "cuda"),
        "cuda_tensor_cuda_processor": (cuda, "cuda"),
    }
    output: dict[str, Any] = {}
    for name, (images, device) in cases.items():
        values = []
        result = None
        for iteration in range(args.repeats + 1):
            synchronize()
            start = time.perf_counter()
            kwargs = {"device": device} if device else {}
            result = processor.apply_chat_template(
                messages(images),
                tokenize=True,
                add_generation_prompt=False,
                continue_final_message=True,
                return_dict=True,
                return_tensors="pt",
                **kwargs,
            )
            synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            if iteration:
                values.append(elapsed)
        assert result is not None
        pixel_values = result["pixel_values"]
        output[name] = {
            "all_ms": [round(value, 3) for value in values],
            "median_ms": round(statistics.median(values), 3),
            "pixel_values_shape": list(pixel_values.shape),
            "pixel_values_device": str(pixel_values.device),
            "pixel_values_dtype": str(pixel_values.dtype),
        }
    output["processor_class"] = (
        f"{type(processor.image_processor).__module__}."
        f"{type(processor.image_processor).__name__}"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
