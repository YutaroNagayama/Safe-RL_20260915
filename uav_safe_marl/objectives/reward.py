"""Navigation reward terms, excluding all collision and safety costs."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from uav_safe_marl.core.types import AgentState


class GoalProgressReward:
    """Navigation efficiency plus a weak penalty on unnecessary residual control."""

    def __init__(
        self,
        goal_reward: float,
        w_progress: float,
        w_time: float,
        progress_mode: str = "distance_delta",
        dt: float = 0.1,
        absolute_distance_enabled: bool = True,
        absolute_distance_scale: float = 280.0,
        w_absolute_distance: float = 0.002,
        reference_speed_fraction: float = 0.75,
        residual_penalty_enabled: bool = False,
        w_residual: float = 0.0,
    ) -> None:
        if progress_mode not in {"distance_delta", "directional", "normalized_goal_delta"}:
            raise ValueError("invalid progress mode")
        if absolute_distance_scale <= 0.0 or w_absolute_distance < 0.0:
            raise ValueError("invalid absolute-distance reward parameters")
        self.goal_reward = goal_reward
        self.w_progress = w_progress
        self.w_time = w_time
        self.progress_mode = progress_mode
        self.dt = dt
        self.absolute_distance_enabled = absolute_distance_enabled
        self.absolute_distance_scale = absolute_distance_scale
        self.w_absolute_distance = w_absolute_distance
        self.reference_speed_fraction = reference_speed_fraction
        self.residual_penalty_enabled = residual_penalty_enabled
        self.w_residual = w_residual

    def compute(self, previous: Sequence[AgentState], current: Sequence[AgentState], reached: np.ndarray, active_mask: np.ndarray | None = None, agent_mask: np.ndarray | None = None, normalized_residual: np.ndarray | None = None) -> tuple[float, dict[str, float]]:
        active = np.ones(len(previous), dtype=bool) if active_mask is None else np.asarray(active_mask, dtype=bool)
        present = np.ones(len(previous), dtype=bool) if agent_mask is None else np.asarray(agent_mask, dtype=bool)
        before = np.array([np.linalg.norm(s.goal - s.position) for s in previous])
        after = np.array([np.linalg.norm(s.goal - s.position) for s in current])
        if self.progress_mode == "distance_delta":
            progress = float(np.mean((before - after)[active])) if np.any(active) else 0.0
        elif self.progress_mode == "normalized_goal_delta":
            values = []
            for index, old in enumerate(previous):
                if not active[index]:
                    continue
                scale = self.reference_speed_fraction * float(np.min(old.velocity_limit)) * self.dt
                values.append(float(np.clip((before[index] - after[index]) / max(scale, 1e-12), -1.0, 1.0)))
            progress = float(np.mean(values)) if values else 0.0
        else:
            values = []
            for index, (old, new) in enumerate(zip(previous, current, strict=True)):
                if not active[index]:
                    continue
                goal_vector = old.goal - old.position
                goal_distance = float(np.linalg.norm(goal_vector))
                speed_limit = float(np.min(old.velocity_limit))
                if goal_distance <= 1e-12 or speed_limit <= 1e-12 or self.dt <= 0.0:
                    values.append(0.0)
                    continue
                displacement = new.position - old.position
                values.append(float(displacement @ (goal_vector / goal_distance)) / (speed_limit * self.dt))
            progress = float(np.mean(values)) if values else 0.0
        goal = self.goal_reward * float(np.mean(reached[present].astype(float)))
        time = 1.0
        mean_normalized_distance = float(np.mean(after[active] / self.absolute_distance_scale)) if np.any(active) else 0.0
        absolute_distance = self.w_absolute_distance * mean_normalized_distance if self.absolute_distance_enabled else 0.0
        residual_values = np.zeros((len(previous), 3), dtype=np.float64) if normalized_residual is None else np.asarray(normalized_residual, dtype=np.float64)
        residual_magnitude = float(np.mean(np.sum(residual_values[active] ** 2, axis=1) / 3.0)) if np.any(active) else 0.0
        residual_penalty = self.w_residual * residual_magnitude if self.residual_penalty_enabled else 0.0
        total = goal + self.w_progress * progress - self.w_time * time - absolute_distance - residual_penalty
        return total, {
            "goal_reward": goal,
            "progress": progress,
            "time_penalty": self.w_time * time,
            "absolute_distance_penalty": absolute_distance,
            "mean_normalized_goal_distance": mean_normalized_distance,
            "residual_penalty": residual_penalty,
            "mean_normalized_residual_squared": residual_magnitude,
        }
