"""Shared immutable data structures used across module boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class AgentState:
    position: FloatArray
    velocity: FloatArray
    goal: FloatArray
    acceleration_limit: FloatArray
    velocity_limit: FloatArray

    def __post_init__(self) -> None:
        for name in ("position", "velocity", "goal", "acceleration_limit", "velocity_limit"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (3,):
                raise ValueError(f"{name} must have shape (3,)")
            setattr(self, name, value)


@dataclass(slots=True, frozen=True)
class HOCBFConstraint:
    neighbour_id: int
    a: FloatArray
    b: float
    urgency: float
    responsibility_feasible: bool = True


@dataclass(slots=True)
class SafetyResult:
    safe_action: FloatArray
    intervention_action: FloatArray
    active_neighbours: tuple[int, ...] = ()
    skipped_neighbours: tuple[int, ...] = ()
    infeasible: bool = False
    resolver_used: bool = False
    emergency: bool = False
    max_urgency: float = 0.0
    responsibility_infeasible_count: int = 0


@dataclass(slots=True)
class Transition:
    state: Any
    nominal_action: FloatArray
    safe_action: FloatArray
    reward: float
    cost: float
    next_state: Any
    terminated: bool
    truncated: bool = False
    extras: dict[str, Any] = field(default_factory=dict)
    reference_action: FloatArray | None = None
    normalized_residual_action: FloatArray | None = None
    residual_action: FloatArray | None = None
    command_action: FloatArray | None = None
    actor_phase: int = 0
    is_actor_decision_tick: bool = True
    held_residual_action: FloatArray | None = None
    next_actor_phase: int = 0
    next_is_actor_decision_tick: bool = True
    next_held_residual_action: FloatArray | None = None

    @property
    def done(self) -> bool:
        """Compatibility alias: only true MDP termination stops bootstrap."""
        return self.terminated
