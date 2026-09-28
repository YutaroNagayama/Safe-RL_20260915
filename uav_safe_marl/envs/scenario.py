"""Deterministic scenario generation."""

from __future__ import annotations

import numpy as np

from uav_safe_marl.core.types import AgentState


def opposing_circle_scenario(agent_count: int, radius: float, acceleration_limit: float, velocity_limit: float) -> list[AgentState]:
    """Place agents on a circle with antipodal goals."""
    angles = np.linspace(0.0, 2.0 * np.pi, agent_count, endpoint=False)
    states: list[AgentState] = []
    for angle in angles:
        position = radius * np.array([np.cos(angle), np.sin(angle), 0.0])
        states.append(AgentState(position, np.zeros(3), -position, np.full(3, acceleration_limit), np.full(3, velocity_limit)))
    return states


def predicted_los_candidates(states: list[AgentState], threshold: float, prediction_horizon: float = 50.0) -> set[tuple[int, int]]:
    """Preflight predictor using initial motion and straight reference routes."""
    candidates: set[tuple[int, int]] = set()
    for first_id, first in enumerate(states):
        for second_id in range(first_id + 1, len(states)):
            second = states[second_id]
            relative_start = first.position - second.position
            relative_change = (first.goal - first.position) - (second.goal - second.position)
            denominator = float(relative_change @ relative_change)
            fraction = 0.0 if denominator <= 1e-12 else float(np.clip(-(relative_start @ relative_change) / denominator, 0.0, 1.0))
            closest = relative_start + fraction * relative_change
            relative_velocity = first.velocity - second.velocity
            velocity_denominator = float(relative_velocity @ relative_velocity)
            closest_time = 0.0 if velocity_denominator <= 1e-12 else float(np.clip(-(relative_start @ relative_velocity) / velocity_denominator, 0.0, prediction_horizon))
            moving_closest = relative_start + closest_time * relative_velocity
            if min(np.linalg.norm(closest), np.linalg.norm(moving_closest)) <= threshold:
                candidates.add((first_id, second_id))
    return candidates
