"""Conservative Augmented Lagrangian (CAL) utilities."""

from __future__ import annotations

import numpy as np

from uav_safe_marl.policies.base import require_torch


def cost_ucb(estimates: np.ndarray, beta: float) -> np.ndarray:
    """PDF slide 10: ensemble mean plus beta times population std."""
    values = np.asarray(estimates, dtype=np.float64)
    if values.shape[0] < 1:
        raise ValueError("cost ensemble cannot be empty")
    return values.mean(axis=0) + beta * values.std(axis=0, ddof=0)


class CALObjective:
    """Piecewise augmented objective from PDF slide 8."""

    def __init__(self, cost_limit: float, coefficient: float) -> None:
        if coefficient <= 0:
            raise ValueError("CAL coefficient must be positive")
        self.cost_limit = cost_limit
        self.coefficient = coefficient

    def evaluate(self, reward_return: np.ndarray | float, conservative_cost: np.ndarray | float, dual: float) -> np.ndarray:
        reward = np.asarray(reward_return, dtype=np.float64)
        violation = np.asarray(conservative_cost, dtype=np.float64) - self.cost_limit
        active = dual / self.coefficient > -violation
        penalized = reward - dual * violation - 0.5 * self.coefficient * violation * violation
        return np.where(active, penalized, reward)

    def effective_multiplier(self, conservative_cost: np.ndarray | float, dual: float) -> np.ndarray:
        """Return ``max(0, lambda + c_AL * g)`` for actor optimization."""
        violation = np.asarray(conservative_cost, dtype=np.float64) - self.cost_limit
        return np.maximum(0.0, dual + self.coefficient * violation)


class TorchCAL:
    """Representation-independent CAL variants operating only on scalar Q values."""

    def __init__(self, cost_limit: float, coefficient: float, beta: float, mode: str = "pdf_variant") -> None:
        if mode not in {"pdf_variant", "paper"}:
            raise ValueError("CAL mode must be pdf_variant or paper")
        self.cost_limit, self.coefficient, self.beta, self.mode = cost_limit, coefficient, beta, mode

    def ucb(self, member_values):
        return member_values.mean(dim=0) + self.beta * member_values.std(dim=0, unbiased=False)

    def actor_penalty(self, actor_ucb, replay_ucb, dual: float, *, constraint_violation: float | None = None):
        """Return the actor Cost term.

        ``constraint_violation`` selects the episodic-CMDP variant: one scalar
        start-state violation determines the CAL multiplier, while gradients
        are still taken from Cost-UCB at replay states.  Omitting it preserves
        the legacy Replay-CAL behavior exactly.
        """
        torch = require_torch()
        if constraint_violation is not None:
            multiplier = torch.clamp(
                torch.as_tensor(dual + self.coefficient * constraint_violation, device=actor_ucb.device),
                min=0.0,
            ).detach()
            return multiplier * actor_ucb
        if self.mode == "paper":
            multiplier = torch.clamp(torch.as_tensor(dual, device=actor_ucb.device) + self.coefficient * (replay_ucb.mean() - self.cost_limit), min=0.0).detach()
            return multiplier * actor_ucb
        violation = actor_ucb - self.cost_limit
        active = dual / self.coefficient > -violation
        return torch.where(active, dual * violation + 0.5 * self.coefficient * violation.square(), torch.zeros_like(violation))

    def dual_violation(self, replay_ucb) -> float:
        return float(replay_ucb.mean().detach()) - self.cost_limit

    def effective_multiplier(self, dual: float, constraint_violation: float) -> float:
        """Scalar CAL weight associated with the same global constraint signal."""
        return max(0.0, float(dual) + self.coefficient * float(constraint_violation))
