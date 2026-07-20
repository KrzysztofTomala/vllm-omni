#!/usr/bin/env python3
"""Offline Alpamayo 1.5 trajectory inference."""

from __future__ import annotations

import argparse

import numpy as np

from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="nvidia/Alpamayo-1.5-10B")
    parser.add_argument("--deploy-config", default="vllm_omni/deploy/alpamayo1_5.yaml")
    parser.add_argument(
        "--input",
        required=True,
        help="NPZ with image_frames, camera_indices, ego_history_xyz and ego_history_rot",
    )
    parser.add_argument("--navigation", default="Drive forward.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-trajectory-samples", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.input)
    observation = {
        "image_frames": data["image_frames"],
        "camera_indices": data["camera_indices"],
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
        "navigation": args.navigation,
    }
    sampling = OmniDiffusionSamplingParams(
        extra_args={
            "robot_obs": observation,
            "num_traj_samples": args.num_trajectory_samples,
            "diffusion_steps": 10,
            "seed": args.seed,
        }
    )
    omni = Omni(model=args.model, deploy_config=args.deploy_config)
    try:
        outputs = omni.generate(args.navigation, sampling_params_list=[sampling])
    finally:
        omni.close()
    if not outputs:
        raise RuntimeError("Alpamayo produced no output")
    multimodal = outputs[0].multimodal_output or {}
    actions = np.asarray(multimodal["actions"])
    print(multimodal.get("reasoning", ""))
    print(f"trajectory samples: {actions.shape}")
    print(actions)


if __name__ == "__main__":
    main()
