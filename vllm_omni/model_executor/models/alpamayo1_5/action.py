# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Alpamayo action-expert layers and unicycle trajectory decoder."""

from __future__ import annotations

import math

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value.float() * torch.rsqrt(value.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normalized.to(value.dtype) * self.weight


class FourierEncoder(nn.Module):
    def __init__(self, dim: int = 20, max_freq: float = 100.0) -> None:
        super().__init__()
        frequencies = torch.logspace(0, math.log10(max_freq), steps=dim // 2)
        self.out_dim = dim
        self.register_buffer("freqs", frequencies[None, :], persistent=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        phase = value[..., None] * self.freqs * (2 * torch.pi)
        return torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1) * math.sqrt(2)


class PerWaypointActionInProjV2(nn.Module):
    """Projection whose parameter names match the published checkpoint."""

    def __init__(
        self,
        out_dim: int,
        *,
        action_dim: int = 2,
        num_enc_layers: int = 2,
        hidden_size: int = 512,
        max_freq: float = 100.0,
        num_fourier_feats: int = 20,
    ) -> None:
        super().__init__()
        self.sinus = nn.ModuleList(FourierEncoder(num_fourier_feats, max_freq) for _ in range(action_dim))
        self.timestep_fourier_encoder = FourierEncoder(num_fourier_feats, max_freq)
        input_size = (action_dim + 1) * num_fourier_feats
        layers: list[nn.Module] = [nn.Linear(input_size, hidden_size), nn.SiLU()]
        for layer_index in range(num_enc_layers):
            output_size = hidden_size if layer_index < num_enc_layers - 1 else out_dim
            layers.extend((RMSNorm(hidden_size), nn.Linear(hidden_size, output_size)))
            if layer_index < num_enc_layers - 1:
                layers.append(nn.SiLU())
        self.encoder = nn.Module()
        self.encoder.trunk = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, action: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        batch, waypoints, _ = action.shape
        action_features = torch.cat([encoder(action[:, :, index]) for index, encoder in enumerate(self.sinus)], dim=-1)
        time_features = self.timestep_fourier_encoder(timestep[..., -1]).repeat(1, waypoints, 1)
        features = torch.cat((action_features, time_features), dim=-1)
        encoded = self.encoder.trunk(features.flatten(0, 1)).reshape(batch, waypoints, -1)
        return self.norm(encoded)


def _estimate_initial_velocity(history_xyz: torch.Tensor, history_rot: torch.Tensor, dt: float) -> torch.Tensor:
    """Use Alpamayo's regularized trapezoidal velocity estimate."""

    xyz = history_xyz.float()
    rot = history_rot.float()
    dxy = torch.diff(xyz[..., :2], dim=-2)
    theta = torch.atan2(rot[..., 1, 0], rot[..., 0, 0])
    delta_theta = torch.atan2(torch.sin(torch.diff(theta, dim=-1)), torch.cos(torch.diff(theta, dim=-1)))
    theta = torch.cat((theta[..., :1], theta[..., :1] + torch.cumsum(delta_theta, dim=-1)), dim=-1)
    *leading, intervals, _ = dxy.shape
    matrix = xyz.new_zeros(*leading, 2 * intervals, intervals + 1)
    rows = torch.arange(intervals, device=xyz.device)
    matrix[..., 2 * rows, rows] = torch.cos(theta[..., :-1])
    matrix[..., 2 * rows, rows + 1] = torch.cos(theta[..., 1:])
    matrix[..., 2 * rows + 1, rows] = torch.sin(theta[..., :-1])
    matrix[..., 2 * rows + 1, rows + 1] = torch.sin(theta[..., 1:])
    target = (2.0 / dt * dxy).flatten(start_dim=-2)
    lhs = matrix.transpose(-1, -2) @ matrix
    rhs = (matrix.transpose(-1, -2) @ target.unsqueeze(-1)).squeeze(-1)
    # Third-order Tikhonov smoothing, matching the reference action space.
    count = intervals + 1
    d3 = xyz.new_zeros(count - 3, count)
    if count > 3:
        d3_rows = torch.arange(count - 3, device=xyz.device)
        d3[d3_rows, d3_rows] = -1
        d3[d3_rows, d3_rows + 1] = 3
        d3[d3_rows, d3_rows + 2] = -3
        d3[d3_rows, d3_rows + 3] = 1
        lhs = lhs + (1e-6 / dt**6) * (d3.T @ d3)
    lhs = lhs + 1e-4 * torch.eye(count, device=xyz.device, dtype=xyz.dtype)
    # AMP may cast the matrix multiplications above to BF16 independently.
    # Keep the small regularized solve in FP32 for matching dtypes and better
    # numerical stability.
    velocity = torch.linalg.solve(lhs.float(), rhs.float())
    return velocity[..., -1]


class UnicycleTrajectoryDecoder(nn.Module):
    """Decode normalized acceleration/curvature controls to 64 poses."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.dt = float(config.get("dt", 0.1))
        self.n_waypoints = int(config.get("n_waypoints", 64))
        # These are immutable action-space configuration values, not model
        # state. Keeping them as Python scalars also prevents Transformers'
        # low-memory checkpoint loader from materializing non-persistent
        # buffers as zero-filled tensors.
        self.accel_mean = float(config.get("accel_mean", 0.0))
        self.accel_std = float(config.get("accel_std", 1.0))
        self.curvature_mean = float(config.get("curvature_mean", 0.0))
        self.curvature_std = float(config.get("curvature_std", 1.0))

    @torch.no_grad()
    def forward(
        self, action: torch.Tensor, history_xyz: torch.Tensor, history_rot: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if action.shape[-2:] != (self.n_waypoints, 2):
            raise ValueError(f"actions must end in ({self.n_waypoints}, 2)")
        action = action.float()
        acceleration = action[..., 0] * self.accel_std + self.accel_mean
        curvature = action[..., 1] * self.curvature_std + self.curvature_mean
        initial_velocity = _estimate_initial_velocity(history_xyz, history_rot, self.dt)
        velocity = torch.cat(
            (
                initial_velocity.unsqueeze(-1),
                initial_velocity.unsqueeze(-1) + torch.cumsum(acceleration * self.dt, dim=-1),
            ),
            dim=-1,
        )
        theta = torch.cat(
            (
                torch.zeros_like(initial_velocity).unsqueeze(-1),
                torch.cumsum(
                    curvature * (velocity[..., :-1] * self.dt + 0.5 * acceleration * self.dt**2),
                    dim=-1,
                ),
            ),
            dim=-1,
        )
        x = torch.cumsum(
            0.5
            * self.dt
            * (velocity[..., :-1] * torch.cos(theta[..., :-1]) + velocity[..., 1:] * torch.cos(theta[..., 1:])),
            dim=-1,
        )
        y = torch.cumsum(
            0.5
            * self.dt
            * (velocity[..., :-1] * torch.sin(theta[..., :-1]) + velocity[..., 1:] * torch.sin(theta[..., 1:])),
            dim=-1,
        )
        xyz = torch.zeros(*x.shape, 3, device=x.device, dtype=x.dtype)
        xyz[..., 0], xyz[..., 1] = x, y
        xyz[..., 2] = history_xyz[..., -1:, 2]
        yaw = theta[..., 1:]
        rotation = torch.zeros(*yaw.shape, 3, 3, device=yaw.device, dtype=yaw.dtype)
        rotation[..., 0, 0] = torch.cos(yaw)
        rotation[..., 0, 1] = -torch.sin(yaw)
        rotation[..., 1, 0] = torch.sin(yaw)
        rotation[..., 1, 1] = torch.cos(yaw)
        rotation[..., 2, 2] = 1
        return xyz, rotation
