# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Alpamayo 2 Super expert stage: flow-matching trajectory sampling over the reasoning KV.

The reasoning stage (``Alpamayo2SuperForConditionalGeneration``) hands over
each request's KV prefix, up to and including ``<|traj_future_start|>``, with
the prefix length and that token's M-RoPE position as KV-transfer metadata.
This pipeline samples one trajectory per request and runs every request of a
request batch together: prefixes are right-padded to a common length and the
padding is masked out.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from torch import nn
from transformers import DynamicCache
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.alpamayo2_super.action_expert import Alpamayo2SuperActionExpert
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.model_executor.models.alpamayo2_super import Alpamayo2SuperConfig

# Output keys: "actions" holds the (64, 3) ego positions, under the key robot
# policies use for their actions, and "rotations" the matching (64, 3, 3)
# rotation matrices.
ACTIONS_KEY = "actions"
ROTATIONS_KEY = "rotations"

_WEIGHT_PREFIX = "expert."
_DEFAULT_STEPS = 10
_MAX_STEPS = 100


def _expert_shards(model_path: str) -> list[str | None]:
    """Checkpoint shards holding ``expert.*`` weights, so the VLM-only shards are not read.

    ``None`` (no index) reads every shard.
    """
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        return [None]
    with open(index_path, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    return sorted({file for name, file in weight_map.items() if name.startswith(_WEIGHT_PREFIX)})


@dataclass
class _Row:
    """One trajectory to sample: a request, or one sample of an n-sample request."""

    index: int  # position of the request in the batch
    request: OmniDiffusionRequest
    kv: Any
    metadata: Mapping[str, Any]
    sample: int


class Alpamayo2SuperPipeline(nn.Module):
    """Flow-matching trajectory expert conditioned on a transferred VLM KV prefix.

    Each request carries the ego history in
    ``sampling_params.extra_args["robot_obs"]`` (``ego_history_xyz`` ``(16, 3)``,
    ``ego_history_rot`` ``(16, 3, 3)``); ``seed`` fixes the expert's noise and
    ``num_inference_steps`` (default 10) sets the Euler steps.
    """

    supports_request_batch: ClassVar[bool] = True

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        config = Alpamayo2SuperConfig.from_pretrained(od_config.model)
        # Built under the serving dtype, like the rest of the policy, so the
        # buffers match the reasoning stage's precision.
        with set_default_torch_dtype(torch.bfloat16):
            self.expert = Alpamayo2SuperActionExpert(config.expert_config)
        # One source per shard: the loader reads only the first pattern that matches files.
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder=None,
                revision=od_config.revision,
                prefix="",
                fall_back_to_pt=False,
                allow_patterns_overrides=None if shard is None else [shard],
            )
            for shard in _expert_shards(od_config.model)
        ]
        self.to(self.device)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, weight in weights:
            if not name.startswith(_WEIGHT_PREFIX):
                continue  # VLM weights in a shard shared with the expert
            params[name].data.copy_(weight)
            loaded.add(name)
        return loaded

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch, **kwargs: Any) -> list[DiffusionOutput]:
        del kwargs
        outputs: list[DiffusionOutput | None] = [None] * req.num_reqs
        groups: dict[int, list[_Row]] = {}
        for index, request in enumerate(req.requests):
            params = request.sampling_params
            if params.past_key_values is None:
                if request.is_dummy_run():
                    outputs[index] = DiffusionOutput(output={ACTIONS_KEY: torch.zeros(64, 3)})
                else:
                    outputs[index] = DiffusionOutput(error="Alpamayo 2 Super expert stage received no KV prefix.")
                continue
            steps = params.num_inference_steps or _DEFAULT_STEPS
            if not 1 <= steps <= _MAX_STEPS:
                outputs[index] = DiffusionOutput(error=f"num_inference_steps must be between 1 and {_MAX_STEPS}")
                continue
            # A request whose reasoning stage sampled n sequences carries one prefix per sample.
            if isinstance(params.past_key_values, list):
                rows = [
                    _Row(index, request, kv, metadata, sample)
                    for sample, (kv, metadata) in enumerate(
                        zip(params.past_key_values, params.kv_metadata, strict=True)
                    )
                ]
            else:
                rows = [_Row(index, request, params.past_key_values, params.kv_metadata, 0)]
            groups.setdefault(steps, []).extend(rows)
        for steps, rows in groups.items():
            trajectories = self._sample(rows, steps)
            by_request: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
            for row, trajectory in zip(rows, trajectories, strict=True):
                by_request.setdefault(row.index, []).append(trajectory)
            for index, samples in by_request.items():
                if isinstance(req.requests[index].sampling_params.past_key_values, list):
                    xyz, rot = torch.stack([t[0] for t in samples]), torch.stack([t[1] for t in samples])
                else:
                    xyz, rot = samples[0]
                outputs[index] = DiffusionOutput(output={ACTIONS_KEY: xyz, ROTATIONS_KEY: rot})
        # The reasoning stage shares the GPU and stages each KV handoff outside
        # its own memory budget; return the freed prefixes.
        torch.accelerator.empty_cache()
        return outputs

    def _batched_prefix(self, rows: list[_Row]) -> tuple[DynamicCache, torch.Tensor, int]:
        """Right-pad the rows' prefixes into one cache; returns it, the lengths and the padded length."""
        lengths = []
        for row in rows:
            prefix_len = int(row.metadata["prefix_len"])
            if int(row.metadata.get("seq_len", prefix_len)) != prefix_len:
                raise RuntimeError(f"KV transfer covers {row.metadata.get('seq_len')} tokens, expected {prefix_len}")
            lengths.append(prefix_len)
        padded_len = max(lengths)
        cache = DynamicCache()
        for layer in range(len(rows[0].kv.key_cache)):
            # Received per layer as [seq_len, kv_heads, head_dim]; the expert's
            # Hugging Face attention takes [batch, kv_heads, seq_len, head_dim].
            heads, head_dim = rows[0].kv.key_cache[layer].shape[1:]
            dtype = rows[0].kv.key_cache[layer].dtype
            key = torch.zeros(len(rows), heads, padded_len, head_dim, dtype=dtype, device=self.device)
            value = torch.zeros_like(key)
            for position, (row, length) in enumerate(zip(rows, lengths, strict=True)):
                key[position, :, :length] = row.kv.key_cache[layer][:length].transpose(0, 1)
                value[position, :, :length] = row.kv.value_cache[layer][:length].transpose(0, 1)
                row.kv.key_cache[layer] = row.kv.value_cache[layer] = None  # release as we copy
            cache.update(key, value, layer)
        return cache, torch.tensor(lengths, device=self.device), padded_len

    def _sample(self, rows: list[_Row], num_steps: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Flow-matching (Euler) sampling of one trajectory per row."""
        expert = self.expert
        device = self.device
        batch = len(rows)
        num_waypoints, action_dim = expert.action_space.get_action_space_dims()
        prefix, lengths, padded_len = self._batched_prefix(rows)

        trigger = torch.tensor([row.metadata["trigger_position"] for row in rows], device=device)  # [batch, 3]
        position_ids = (trigger.T[:, :, None] + 1 + torch.arange(num_waypoints, device=device)).long()
        dtype = expert.action_out_proj.weight.dtype
        # Waypoints attend to their own prefix and to each other; padding is masked.
        attention_mask = torch.zeros((batch, 1, num_waypoints, padded_len + num_waypoints), device=device, dtype=dtype)
        padding = torch.arange(padded_len, device=device)[None, :] >= lengths[:, None]
        attention_mask[:, 0, :, :padded_len].masked_fill_(padding[:, None, :], torch.finfo(dtype).min)

        # Per-row noise, so a trajectory does not depend on its batch.
        action = torch.cat(
            [torch.randn(1, num_waypoints, action_dim, device=device, generator=self._generator(row)) for row in rows]
        )
        time_steps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        for i in range(num_steps):
            hidden = expert.expert(
                inputs_embeds=expert.action_in_proj(action, time_steps[i].expand(batch, 1, 1)),
                position_ids=position_ids,
                past_key_values=prefix,
                attention_mask=attention_mask,
                use_cache=True,
                is_causal=not expert.non_causal,
            ).last_hidden_state
            prefix.crop(padded_len)
            velocity = expert.action_out_proj(hidden).view(batch, num_waypoints, action_dim)
            # A (1, 1, 1) step keeps the update in float32, as in the reference.
            action = action + (time_steps[i + 1] - time_steps[i]).view(1, 1, 1) * velocity

        history_xyz = torch.stack([self._history(row.request, "ego_history_xyz") for row in rows])
        history_rot = torch.stack([self._history(row.request, "ego_history_rot") for row in rows])
        trajectory_xyz, trajectory_rot = expert.action_space.action_to_traj(action, history_xyz, history_rot)
        return [(trajectory_xyz[position].cpu(), trajectory_rot[position].cpu()) for position in range(batch)]

    def _generator(self, row: _Row) -> torch.Generator:
        params = row.request.sampling_params
        if row.sample == 0 and isinstance(params.generator, torch.Generator):
            return params.generator
        # Engine requests always carry a seed; sample i of an n-sample request uses seed + i.
        return torch.Generator(device=self.device).manual_seed(int(params.seed) + row.sample)

    def _history(self, request: OmniDiffusionRequest, key: str) -> torch.Tensor:
        observation: Mapping[str, Any] = (request.sampling_params.extra_args or {})["robot_obs"]
        history = torch.as_tensor(observation[key], device=self.device, dtype=torch.float32)
        # (T, 3) positions or (T, 3, 3) rotations, with any leading batch dims of size 1 dropped.
        return history.reshape(-1, *history.shape[-1:]) if key == "ego_history_xyz" else history.reshape(-1, 3, 3)
