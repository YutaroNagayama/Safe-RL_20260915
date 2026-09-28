"""Centralized scalar cost critic."""

from .reward_critic import RewardCritic


class CostCritic(RewardCritic):
    """Architecturally identical critic trained on CMDP cost targets."""

