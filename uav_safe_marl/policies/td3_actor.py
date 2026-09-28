"""Shared deterministic actor for the TD3/CAL-inspired mode."""

from __future__ import annotations

from .base import require_torch

torch = require_torch()
nn = torch.nn


class TD3Actor(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int, action_dim: int = 3) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim), nn.Tanh())

    def forward(self, observation):
        return self.network(observation)

