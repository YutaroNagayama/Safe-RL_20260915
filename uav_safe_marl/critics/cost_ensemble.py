"""Conservative cost critic ensemble."""

from uav_safe_marl.policies.base import require_torch
from .cost_critic import CostCritic

torch = require_torch()
nn = torch.nn


class CostCriticEnsemble(nn.Module):
    def __init__(self, size: int, state_dim: int, joint_action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if size < 2:
            raise ValueError("cost ensemble size must be at least two")
        self.members = nn.ModuleList([CostCritic(state_dim, joint_action_dim, hidden_dim) for _ in range(size)])

    def forward(self, state, action):
        return torch.stack([member(state, action) for member in self.members], dim=0)

    def ucb(self, state, action, beta: float):
        estimates = self(state, action)
        return estimates.mean(dim=0) + beta * estimates.std(dim=0, unbiased=False)

