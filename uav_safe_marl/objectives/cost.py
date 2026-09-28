"""Distance and composite CMDP costs."""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np

from uav_safe_marl.core.types import AgentState
from .intervention import intervention_cost


def distance_cost(states: Sequence[AgentState], d_safe: float, d_warn: float, active_mask: np.ndarray | None = None) -> float:
    """Mean each active agent's cost to its closest (highest-risk) peer.

    The distance penalty curve is unchanged.  Only its aggregation is
    nearest-peer based so that adding simultaneous neighbours cannot increase
    one agent's distance cost merely by increasing its degree.
    """
    if d_warn <= d_safe:
        raise ValueError("d_warn must be greater than d_safe")
    active_states = list(states) if active_mask is None else [state for state, active in zip(states, np.asarray(active_mask, dtype=bool), strict=True) if active]
    if len(active_states) < 2:
        return 0.0
    worst_peer_cost = np.zeros(len(active_states), dtype=np.float64)
    for (first_index, first), (second_index, second) in combinations(enumerate(active_states), 2):
        distance = float(np.linalg.norm(first.position - second.position))
        phi = max(0.0, (d_warn - distance) / (d_warn - d_safe))
        pair_cost = phi * phi
        worst_peer_cost[first_index] = max(worst_peer_cost[first_index], pair_cost)
        worst_peer_cost[second_index] = max(worst_peer_cost[second_index], pair_cost)
    return float(np.mean(worst_peer_cost))


class CompositeCost:
    """Combine independently switchable distance and intervention terms."""

    def __init__(self, d_safe: float, d_warn: float, distance_enabled: bool = True, intervention_enabled: bool = True, w_intervention: float = 1.0) -> None:
        self.d_safe = d_safe
        self.d_warn = d_warn
        self.distance_enabled = distance_enabled
        self.intervention_enabled = intervention_enabled
        self.w_intervention = w_intervention

    def compute(self, states: Sequence[AgentState], corrections: np.ndarray, active_mask: np.ndarray | None = None) -> tuple[float, dict[str, float]]:
        distance = distance_cost(states, self.d_safe, self.d_warn, active_mask) if self.distance_enabled else 0.0
        intervention = intervention_cost(corrections, active_mask) if self.intervention_enabled else 0.0
        return distance + self.w_intervention * intervention, {"distance_cost": distance, "intervention_cost": intervention}
