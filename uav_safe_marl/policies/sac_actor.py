"""Shared squashed-Gaussian actor for SAC/CAL."""

from __future__ import annotations

from .base import require_torch

torch = require_torch()
nn = torch.nn


class SACActor(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int, action_dim: int = 3) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, observation):
        features = self.body(observation)
        return self.mean(features), self.log_std(features).clamp(-20.0, 2.0)

    def sample(self, observation, deterministic: bool = False):
        mean, log_std = self(observation)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        raw = mean if deterministic else distribution.rsample()
        action = torch.tanh(raw)
        log_probability = distribution.log_prob(raw) - torch.log(1.0 - action.square() + 1e-6)
        return action, log_probability.sum(dim=-1, keepdim=True)

