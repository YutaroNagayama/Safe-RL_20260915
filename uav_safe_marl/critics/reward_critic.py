"""Centralized scalar reward critic."""

from uav_safe_marl.policies.base import require_torch

torch = require_torch()
nn = torch.nn


class RewardCritic(nn.Module):
    def __init__(self, state_dim: int, joint_action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(state_dim + joint_action_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, state, joint_nominal_action):
        return self.network(torch.cat((state, joint_nominal_action), dim=-1))

