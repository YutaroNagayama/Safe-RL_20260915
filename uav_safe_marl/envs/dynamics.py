"""Three-dimensional point-mass UAV dynamics."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from uav_safe_marl.core.types import AgentState


class DoubleIntegratorDynamics:
    """Apply constant acceleration over one base tick."""

    def __init__(self, speed_constraint_mode: str = "norm") -> None:
        if speed_constraint_mode not in {"norm", "box"}:
            raise ValueError("speed_constraint_mode must be norm or box")
        self.speed_constraint_mode = speed_constraint_mode

    def step(self, states: Sequence[AgentState], accelerations: np.ndarray, dt: float, active_mask: np.ndarray | None = None) -> list[AgentState]:
        if accelerations.shape != (len(states), 3):
            raise ValueError("accelerations must have shape (agent_count, 3)")
        result: list[AgentState] = []
        active = np.ones(len(states), dtype=bool) if active_mask is None else np.asarray(active_mask, dtype=bool)
        for index, (state, acceleration) in enumerate(zip(states, accelerations, strict=True)):
            if not active[index]:
                result.append(replace(state, velocity=np.zeros(3)))
                continue
            acceleration = np.clip(acceleration, -state.acceleration_limit, state.acceleration_limit)
            velocity = state.velocity + dt * acceleration
            if self.speed_constraint_mode == "box":
                velocity = np.clip(velocity, -state.velocity_limit, state.velocity_limit)
            else:
                speed_limit = float(np.min(state.velocity_limit))
                speed = float(np.linalg.norm(velocity))
                if speed > speed_limit:
                    velocity = velocity * (speed_limit / speed)
            position = state.position + dt * state.velocity + 0.5 * dt * dt * acceleration
            result.append(replace(state, position=position, velocity=velocity))
        return result
