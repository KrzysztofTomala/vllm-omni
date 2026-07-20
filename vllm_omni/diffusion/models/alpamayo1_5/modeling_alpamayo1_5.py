# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transformers implementation used by the Alpamayo policy pipeline."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from transformers import (
    AutoModel,
    DynamicCache,
    GenerationConfig,
    LogitsProcessor,
    LogitsProcessorList,
    PreTrainedModel,
    Qwen3VLForConditionalGeneration,
    StoppingCriteria,
    StoppingCriteriaList,
)

from vllm_omni.model_executor.models.alpamayo1_5.action import (
    PerWaypointActionInProjV2,
    UnicycleTrajectoryDecoder,
)
from vllm_omni.model_executor.models.alpamayo1_5.configuration_alpamayo1_5 import (
    Alpamayo1_5Config,
)
from vllm_omni.model_executor.models.alpamayo1_5.processing import (
    fuse_history_tokens,
    get_observation_history,
    observation_tensor,
)


class _MaskTrajectoryTokens(LogitsProcessor):
    def __init__(self, start: int, size: int) -> None:
        self.start = start
        self.size = size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        del input_ids
        scores[:, self.start : self.start + self.size] = -torch.inf
        return scores


class _StopAfterToken(StoppingCriteria):
    """Stop one decode step after every sequence emitted ``token_id``.

    Transformers returns a generation cache which trails the returned sequence
    by one token. Waiting one extra step ensures the action expert's prefix
    cache includes ``<|traj_future_start|>``, matching the published model.
    """

    def __init__(self, token_id: int) -> None:
        self.token_id = token_id
        self._seen = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        del scores, kwargs
        present = bool(torch.all(torch.any(input_ids == self.token_id, dim=1)))
        stop = self._seen and present
        self._seen = present
        return stop


def _observation_rotation(observation: Mapping[str, Any], history: torch.Tensor) -> torch.Tensor:
    value = observation.get("ego_history_rot")
    if value is None:
        return torch.eye(3, dtype=history.dtype).expand(16, 3, 3).clone()
    rotation = observation_tensor(value)
    while rotation.ndim > 3 and rotation.shape[0] == 1:
        rotation = rotation.squeeze(0)
    if rotation.shape != (16, 3, 3):
        raise ValueError(f"ego_history_rot must have shape (16, 3, 3), got {tuple(rotation.shape)}")
    return rotation


class Alpamayo1_5TorchModel(PreTrainedModel):
    """Published Alpamayo modules with a model-local Transformers KV cache."""

    config_class = Alpamayo1_5Config
    base_model_prefix = "vlm"
    _supports_sdpa = True

    def __init__(self, config: Alpamayo1_5Config) -> None:
        super().__init__(config)
        self.vlm = Qwen3VLForConditionalGeneration(config)
        expert_config = copy.deepcopy(config.text_config)
        for key, value in config.expert_cfg.items():
            setattr(expert_config, key, value)
        expert_config._attn_implementation = config._attn_implementation
        self.expert = AutoModel.from_config(expert_config)
        if hasattr(self.expert, "embed_tokens"):
            del self.expert.embed_tokens
        input_config = config.action_in_proj_cfg
        self.action_in_proj = PerWaypointActionInProjV2(
            expert_config.hidden_size,
            num_enc_layers=int(input_config.get("num_enc_layers", 2)),
            hidden_size=int(input_config.get("hidden_size", 512)),
            max_freq=float(input_config.get("max_freq", 100.0)),
            num_fourier_feats=int(input_config.get("num_fourier_feats", 20)),
        )
        self.action_out_proj = nn.Linear(expert_config.hidden_size, 2)
        self.trajectory_decoder = UnicycleTrajectoryDecoder(config.action_space_cfg)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vlm.get_input_embeddings()

    def get_output_embeddings(self) -> nn.Module:
        return self.vlm.get_output_embeddings()

    def tie_weights(self, *args: Any, **kwargs: Any) -> None:
        self.vlm.tie_weights(*args, **kwargs)

    def _sample_actions(
        self,
        *,
        prompt_cache: DynamicCache,
        sequences: torch.Tensor,
        rope_deltas: torch.Tensor,
        prefix_mask: torch.Tensor | None,
        observation: Mapping[str, Any],
        sample_count: int,
        inference_steps: int,
        action_temperature: float,
        seed: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = sequences.device
        future_start_id = int(self.config.traj_token_ids.get("future_start", 155681))
        matches = sequences == future_start_id
        has_start = matches.any(dim=1)
        if not bool(torch.all(has_start)):
            raise RuntimeError("Alpamayo reasoning rollout did not emit <|traj_future_start|>")
        offset = matches.int().argmax(dim=1) + 1

        prompt_cache.batch_repeat_interleave(sample_count)
        cache_length = prompt_cache.get_seq_length()
        diffusion_tokens = 64
        positions = torch.arange(diffusion_tokens, device=device)
        # Transformers 5 represents Qwen3-VL positions as text + the three
        # multimodal RoPE planes. Supplying the older three-plane layout makes
        # Qwen silently drop text_position_ids in the expert forward pass.
        rope_planes = 4 if hasattr(self.expert.config, "rope_parameters") else 3
        positions = positions.view(1, 1, -1).expand(rope_planes, sample_count, -1).clone()
        action_offsets = (rope_deltas + offset[:, None]).repeat_interleave(sample_count, dim=0)
        positions += action_offsets.to(device).view(1, sample_count, 1)
        attention_mask = torch.zeros(
            sample_count,
            1,
            diffusion_tokens,
            cache_length + diffusion_tokens,
            device=device,
            dtype=torch.float32,
        )
        if prefix_mask is not None:
            repeated_mask = prefix_mask.repeat_interleave(sample_count, dim=0)
            input_mask = repeated_mask[:, None, None, :]
            attention_mask[:, :, :, : input_mask.shape[-1]].masked_fill_(
                input_mask == 0,
                torch.finfo(attention_mask.dtype).min,
            )
        repeated_offsets = offset.repeat_interleave(sample_count)
        for index, action_offset in enumerate(repeated_offsets.tolist()):
            attention_mask[index, :, :, action_offset:cache_length] = torch.finfo(attention_mask.dtype).min

        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        action = (
            torch.randn(
                sample_count,
                diffusion_tokens,
                2,
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
            * action_temperature
        )
        for step in range(inference_steps):
            timestep = action.new_full((sample_count, 1, 1), step / inference_steps)
            embeddings = self.action_in_proj(action, timestep).to(self.action_out_proj.weight.dtype)
            output = self.expert(
                inputs_embeds=embeddings,
                position_ids=positions,
                past_key_values=prompt_cache,
                attention_mask=attention_mask,
                use_cache=True,
                is_causal=not bool(self.config.expert_non_causal_attention),
            )
            prompt_cache.crop(cache_length)
            velocity = self.action_out_proj(output.last_hidden_state[:, -diffusion_tokens:])
            action = action + velocity.float() / inference_steps

        history = get_observation_history(observation).to(device=device, dtype=torch.float32)
        rotation = _observation_rotation(observation, history).to(device=device, dtype=torch.float32)
        xyz, rotations = self.trajectory_decoder(
            action,
            history.unsqueeze(0).expand(sample_count, -1, -1),
            rotation.unsqueeze(0).expand(sample_count, -1, -1, -1),
        )
        return xyz, rotations, action

    @torch.inference_mode()
    def sample_trajectories(
        self,
        tokenized: Mapping[str, torch.Tensor],
        observation: Mapping[str, Any],
        *,
        tokenizer: Any,
        temperature: float = 0.6,
        top_p: float = 0.98,
        top_k: int | None = None,
        max_tokens: int = 128,
        sample_count: int = 6,
        inference_steps: int = 10,
        action_temperature: float = 1.0,
        seed: int | None = None,
    ) -> dict[str, Any]:
        model_inputs = dict(tokenized)
        input_ids = fuse_history_tokens(
            model_inputs.pop("input_ids"),
            get_observation_history(observation).unsqueeze(0),
        )
        generation_config = GenerationConfig(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_tokens,
            return_dict_in_generate=True,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        future_start_id = int(self.config.traj_token_ids.get("future_start", 155681))
        generated = self.vlm.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            stopping_criteria=StoppingCriteriaList([_StopAfterToken(future_start_id)]),
            logits_processor=LogitsProcessorList(
                [
                    _MaskTrajectoryTokens(
                        int(self.config.traj_token_start_idx),
                        int(self.config.traj_vocab_size),
                    )
                ]
            ),
            **model_inputs,
        )
        rope_deltas = self.vlm.model.rope_deltas
        xyz, rotations, controls = self._sample_actions(
            prompt_cache=generated.past_key_values,
            sequences=generated.sequences,
            rope_deltas=rope_deltas,
            prefix_mask=model_inputs.get("attention_mask"),
            observation=observation,
            sample_count=sample_count,
            inference_steps=inference_steps,
            action_temperature=action_temperature,
            seed=seed,
        )
        generated_tokens = generated.sequences[0, input_ids.shape[1] :]
        reasoning = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return {
            "actions": xyz,
            "rotations": rotations,
            "normalized_controls": controls,
            "reasoning": reasoning,
        }
