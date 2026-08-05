#!/usr/bin/env python3
"""Offline Alpamayo 1.5 trajectory or VQA inference."""

from __future__ import annotations

import argparse

import numpy as np
import torch
from vllm import SamplingParams

from vllm_omni import Omni
from vllm_omni.model_executor.models.alpamayo1_5.processing import (
    create_policy_messages,
    create_vqa_messages,
    fuse_history_tokens,
)
from vllm_omni.model_executor.models.alpamayo1_5.tokenizer import (
    ensure_extended_tokenizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="nvidia/Alpamayo-1.5-10B")
    parser.add_argument("--deploy-config", default="vllm_omni/deploy/alpamayo1_5.yaml")
    parser.add_argument(
        "--input",
        required=True,
        help="NPZ with image_frames, camera_indices, ego_history_xyz and ego_history_rot",
    )
    parser.add_argument("--question", help="Run VQA instead of trajectory generation")
    parser.add_argument("--navigation")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.input)
    frames = data["image_frames"]
    if frames.ndim == 5:
        frames = frames.reshape(-1, *frames.shape[-3:])
    images = list(frames)
    camera_indices = data["camera_indices"].tolist()
    if args.question:
        messages = create_vqa_messages(images, args.question, camera_indices=camera_indices)
    else:
        messages = create_policy_messages(images, camera_indices=camera_indices, navigation=args.navigation)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ensure_extended_tokenizer())
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )
    prompt_input: dict = {"prompt": prompt, "multi_modal_data": {"image": images}}
    extra_args = {}
    if not args.question:
        extra_args["robot_obs"] = {
            # SamplingParams.extra_args is untyped across vLLM's worker RPC;
            # built-in lists avoid unresolved ndarray auxiliary buffers.
            "ego_history_xyz": data["ego_history_xyz"].tolist(),
            "ego_history_rot": data["ego_history_rot"].tolist(),
        }
        extra_args.update(
            num_traj_samples=6,
            diffusion_steps=10,
            _sampling_seed=args.seed,
            _nim_action_rng_compat=True,
        )
        prompt_ids = fuse_history_tokens(
            torch.tensor(tokenizer.encode(prompt)), torch.as_tensor(data["ego_history_xyz"])
        ).tolist()
        prompt_input = {
            "prompt_token_ids": prompt_ids,
            "multi_modal_data": {"image": images},
        }
    sampling = SamplingParams(
        temperature=0.6,
        top_p=0.98,
        max_tokens=128 if not args.question else 256,
        stop_token_ids=[155683] if not args.question else None,
        seed=args.seed,
        extra_args=extra_args,
    )
    omni = Omni(model=args.model, deploy_config=args.deploy_config)
    try:
        outputs = list(
            omni.generate(
                [prompt_input],
                [sampling],
            )
        )
    finally:
        omni.close()

    final = getattr(outputs[-1], "request_output", outputs[-1])
    print(final.outputs[0].text)
    multimodal = getattr(final, "multimodal_output", None)
    if multimodal is None and final.outputs:
        multimodal = getattr(final.outputs[0], "multimodal_output", None)
    if multimodal:
        actions = np.asarray(multimodal["actions"])
        print(f"trajectory samples: {actions.shape}")
        print(actions)


if __name__ == "__main__":
    main()
