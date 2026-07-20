#!/usr/bin/env python3
"""Microbenchmark torchvision JPEG decode strategies on Alpamayo frames."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torchvision
from torchvision.io import ImageReadMode, decode_jpeg


def measure(fn, repeats: int, *, synchronize: bool = False) -> list[float]:
    values = []
    for _ in range(repeats):
        if synchronize:
            torch.accelerator.synchronize()
        start = time.perf_counter()
        output = fn()
        if synchronize:
            torch.accelerator.synchronize()
        values.append((time.perf_counter() - start) * 1000)
        del output
    return values


def summary(values: list[float]) -> dict[str, object]:
    return {
        "all_ms": [round(value, 3) for value in values],
        "median_ms": round(statistics.median(values), 3),
        "min_ms": round(min(values), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()

    paths = sorted((*args.image_dir.glob("*.jpg"), *args.image_dir.glob("*.jpeg")))
    if len(paths) != 16:
        raise ValueError(f"Expected 16 JPEGs, found {len(paths)} in {args.image_dir}")
    encoded = [torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint8) for path in paths]

    # Warm all lazy CUDA/nvJPEG initialization outside the recorded runs.
    decode_jpeg(encoded, mode=ImageReadMode.RGB, device="cuda")
    torch.accelerator.synchronize()

    cpu_pool = ThreadPoolExecutor(max_workers=len(encoded))
    cases = {
        "cuda_batch": lambda: decode_jpeg(encoded, mode=ImageReadMode.RGB, device="cuda"),
        "cuda_sequential": lambda: [
            decode_jpeg(item, mode=ImageReadMode.RGB, device="cuda") for item in encoded
        ],
        "cpu_sequential": lambda: [
            decode_jpeg(item, mode=ImageReadMode.RGB, device="cpu") for item in encoded
        ],
        "cpu_threaded": lambda: list(
            cpu_pool.map(
                lambda item: decode_jpeg(item, mode=ImageReadMode.RGB, device="cpu"),
                encoded,
            )
        ),
    }
    results = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "jpeg_count": len(encoded),
        "compressed_bytes": sum(item.numel() for item in encoded),
        "cases": {},
    }
    for name, fn in cases.items():
        results["cases"][name] = summary(
            measure(fn, args.repeats, synchronize=name.startswith("cuda"))
        )
    cpu_pool.shutdown()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
