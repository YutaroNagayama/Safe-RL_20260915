"""Equal and capability-aware pair responsibility allocation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_safe_marl.core.types import AgentState


def effective_bounds(state: AgentState, dt: float, speed_constraint_mode: str = "box") -> tuple[np.ndarray, np.ndarray]:
    """Intersect acceleration and one-tick velocity limits (PDF slide 24)."""
    if speed_constraint_mode == "norm":
        return -state.acceleration_limit.copy(), state.acceleration_limit.copy()
    lower = np.maximum(-state.acceleration_limit, (-state.velocity_limit - state.velocity) / dt)
    upper = np.minimum(state.acceleration_limit, (state.velocity_limit - state.velocity) / dt)
    if np.any(lower > upper):
        raise ValueError("empty effective input set")
    return lower, upper


def _max_linear(coefficient: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    point = np.where(coefficient >= 0.0, upper, lower)
    return max(0.0, float(coefficient @ point))


@dataclass(slots=True, frozen=True)
class ResponsibilityAllocation:
    """Pair allocation plus its feasibility diagnostics."""

    rho_first: float
    rho_second: float
    kappa_first: float
    kappa_second: float
    required: float
    feasible: bool


class CapabilityResponsibilityAllocator:
    """Allocate complementary pair responsibility using available control margin."""

    def __init__(self, mode: str = "capability", dt: float = 0.1, epsilon: float = 1e-10, speed_constraint_mode: str = "box") -> None:
        if mode not in {"equal", "capability"}:
            raise ValueError("mode must be equal or capability")
        self.mode = mode
        self.dt = dt
        self.epsilon = epsilon
        self.speed_constraint_mode = speed_constraint_mode

    def _capability(self, coefficient: np.ndarray, state: AgentState) -> float:
        lower, upper = effective_bounds(state, self.dt, self.speed_constraint_mode)
        if self.speed_constraint_mode == "box":
            return _max_linear(coefficient, lower, upper)
        from .qp_backend import ActiveSetQPSolver

        point, feasible = ActiveSetQPSolver().maximize_linear(
            coefficient,
            [],
            lower,
            upper,
            velocity=state.velocity,
            speed_limit=float(np.min(state.velocity_limit)),
            dt=self.dt,
        )
        return max(0.0, float(coefficient @ point)) if feasible else 0.0

    def allocate_pair(self, first: AgentState, second: AgentState, q_ij: float) -> tuple[float, float]:
        """Return a numerically safe pair ``(rho_ij, rho_ji)`` summing to one."""
        allocation = self.allocate_pair_details(first, second, q_ij)
        return allocation.rho_first, allocation.rho_second

    def allocate_pair_details(self, first: AgentState, second: AgentState, q_ij: float) -> ResponsibilityAllocation:
        """Return responsibility and detect insufficient combined capability."""
        required = max(0.0, -float(q_ij))
        if self.mode == "equal":
            return ResponsibilityAllocation(0.5, 0.5, float("nan"), float("nan"), required, True)
        relative = first.position - second.position
        norm = float(np.linalg.norm(relative))
        if norm <= self.epsilon:
            return ResponsibilityAllocation(0.5, 0.5, 0.0, 0.0, required, required <= self.epsilon)
        # kappa_i^j is agent i's currently available acceleration away from j.
        kappa_first = self._capability(2.0 * relative, first)
        kappa_second = self._capability(-2.0 * relative, second)
        total = kappa_first + kappa_second
        rho_pref = 0.5 if total <= self.epsilon else kappa_first / total
        feasible = total + self.epsilon >= required
        if required <= self.epsilon:
            rho_first = rho_pref
        elif total <= self.epsilon:
            rho_first = 0.5
        else:
            rho_min = max(0.0, 1.0 - kappa_second / required)
            rho_max = min(1.0, kappa_first / required)
            if rho_min <= rho_max:
                rho_first = float(np.clip(rho_pref, rho_min, rho_max))
            else:
                # Preserve the capability ratio under unavoidable overload.  This
                # equalizes relative utilization and exposes infeasibility separately.
                rho_first = rho_pref
        rho_first = float(np.clip(rho_first, 0.0, 1.0))
        return ResponsibilityAllocation(rho_first, 1.0 - rho_first, kappa_first, kappa_second, required, feasible)
