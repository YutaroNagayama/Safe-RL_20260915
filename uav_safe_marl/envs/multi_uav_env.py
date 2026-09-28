"""Gymnasium-compatible multi-UAV CMDP environment."""

from __future__ import annotations

from typing import Any
from time import perf_counter

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from uav_safe_marl.config.schemas import ExperimentConfig
from uav_safe_marl.control import compose_action_numpy, reference_action_numpy
from uav_safe_marl.core.types import AgentState, SafetyResult
from uav_safe_marl.graph.observations import direct_node_features
from uav_safe_marl.objectives.cost import CompositeCost
from uav_safe_marl.objectives.reward import GoalProgressReward
from uav_safe_marl.safety.qp_filter import HOCBFSafetyFilter
from uav_safe_marl.safety.responsibility import CapabilityResponsibilityAllocator
from uav_safe_marl.safety.execution import SafetyExecutor
from .dynamics import DoubleIntegratorDynamics
from .scenario import opposing_circle_scenario, predicted_los_candidates


class MultiUAVEnv(gym.Env[np.ndarray, np.ndarray]):
    """Central training environment with decentralized per-agent safety filters."""

    metadata = {"render_modes": []}

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config
        count = config.environment.capacity
        self.action_space = spaces.Box(-1.0, 1.0, shape=(count, 3), dtype=np.float32)
        self.observation_space = spaces.Box(-1e12, 1e12, shape=(count, 14), dtype=np.float64)
        self.dynamics = DoubleIntegratorDynamics(config.safety.speed_constraint_mode)
        self.reward_model = GoalProgressReward(
            config.objective.goal_reward,
            config.objective.w_progress,
            config.objective.w_time,
            config.objective.progress_mode,
            config.environment.dt_base,
            config.objective.absolute_distance_enabled,
            config.objective.absolute_distance_scale,
            config.objective.w_absolute_distance,
            config.control.reference_speed_fraction,
            config.objective.residual_penalty_enabled,
            config.objective.w_residual,
        )
        self.cost_model = CompositeCost(config.safety.d_safe, config.safety.d_warn, config.objective.distance_cost_enabled, config.objective.intervention_cost_enabled, config.objective.w_intervention)
        allocator = CapabilityResponsibilityAllocator(config.safety.responsibility_mode, config.environment.dt_base, speed_constraint_mode=config.safety.speed_constraint_mode)
        self.safety_filter = HOCBFSafetyFilter(config.safety.d_safe, config.safety.d_eng, config.safety.k1, config.safety.k2, config.environment.dt_base, allocator, config.safety.conflict_resolver_enabled, config.safety.speed_constraint_mode)
        self.safety_executor = SafetyExecutor(
            self.safety_filter,
            config.performance.safety_backend,
            config.performance.safety_workers,
            config.performance.safety_parallel_min_agents,
        )
        self.states: list[AgentState] = []
        self.step_count = 0
        self._reached = np.zeros(count, dtype=bool)
        self._active = np.ones(count, dtype=bool)
        self._present = np.ones(count, dtype=bool)
        self.actual_agent_count = config.environment.agent_count
        self._reached_step = np.full(count, -1, dtype=np.int64)
        self.reference_route_lengths = np.zeros(count, dtype=np.float64)
        self.last_safety_solve_time = 0.0
        self.last_safety_worker_time_sum = 0.0
        self.last_safety_max_agent_time = 0.0
        self.last_safety_backend = "serial"
        self.last_cbf_engaged_agent_count = 0
        self.last_cbf_engaged_pair_count = 0
        self.predicted_los_edges: set[tuple[int, int]] = set()

    def _observation(self) -> np.ndarray:
        return direct_node_features(self.states, self._active, self.reference_actions())

    def reference_actions(self) -> np.ndarray:
        """Reference command at the current base tick; inactive slots are zero."""
        return reference_action_numpy(
            self.states,
            self.config.control.reference_speed_fraction,
            self.config.control.reference_velocity_gain,
            self._active,
        )

    @property
    def active_mask(self) -> np.ndarray:
        """Agents still inside the controlled airspace."""
        return self._active.copy()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset a seeded opposing-circle scenario using Gymnasium semantics."""
        super().reset(seed=self.config.seed if seed is None else seed)
        radius = float((options or {}).get("radius", max(2.0 * self.config.safety.d_eng, 50.0)))
        scenario = (options or {}).get("scenario")
        if scenario is not None and "distance_thresholds" in scenario:
            expected = {"d_safe": self.config.safety.d_safe, "d_warn": self.config.safety.d_warn, "d_eng": self.config.safety.d_eng}
            supplied = {key: float(scenario["distance_thresholds"][key]) for key in expected}
            if supplied != expected:
                raise ValueError("scenario distance_thresholds must match the experiment configuration")
        actual_count = self.config.environment.agent_count
        if scenario is not None:
            positions = np.asarray(scenario["positions"], dtype=np.float64)
            if positions.ndim != 2 or positions.shape[1:] != (3,):
                raise ValueError("scenario positions must have shape (actual_agent_count, 3)")
            declared = int(scenario.get("actual_agent_count", scenario.get("agent_count", len(positions))))
            if declared != len(positions):
                raise ValueError("scenario agent count must match positions")
            actual_count = declared
        if not 1 <= actual_count <= self.config.environment.capacity:
            raise ValueError("actual_agent_count must be in [1, max_agent_count]")
        real_states = opposing_circle_scenario(actual_count, radius, self.config.environment.acceleration_limit, self.config.environment.velocity_limit)
        if scenario is not None:
            velocities = np.asarray(scenario.get("velocities", np.zeros_like(positions)), dtype=np.float64)
            goals = np.asarray(scenario["goals"], dtype=np.float64)
            if velocities.shape != positions.shape or goals.shape != positions.shape:
                raise ValueError("scenario positions, velocities, and goals must have matching (actual_agent_count, 3) shape")

            def capability(names: tuple[str, ...], default: float) -> np.ndarray:
                raw = next((scenario[name] for name in names if name in scenario), default)
                values = np.asarray(raw, dtype=np.float64)
                if values.ndim == 0:
                    values = np.full(actual_count, float(values))
                if values.shape != (actual_count,) or np.any(values <= 0.0):
                    raise ValueError(f"{names[0]} must be positive with shape (actual_agent_count,)")
                return values

            accelerations = capability(("maximum_accelerations", "acceleration_limits"), self.config.environment.acceleration_limit)
            velocities_max = capability(("maximum_velocities", "velocity_limits"), self.config.environment.velocity_limit)
            real_states = [
                AgentState(positions[i], velocities[i], goals[i], np.full(3, accelerations[i]), np.full(3, velocities_max[i]))
                for i in range(actual_count)
            ]
        dummy = AgentState(np.zeros(3), np.zeros(3), np.zeros(3), np.ones(3), np.ones(3))
        self.states = real_states + [dummy for _ in range(self.config.environment.capacity - actual_count)]
        self.actual_agent_count = actual_count
        self._present = np.arange(self.config.environment.capacity) < actual_count
        supplied_edges = (options or {}).get("predicted_los_edges")
        if supplied_edges is None and scenario is not None:
            supplied_edges = scenario.get("predicted_los_edges")
        if supplied_edges is None:
            self.predicted_los_edges = predicted_los_candidates(
                real_states,
                self.config.scenario.interaction_threshold,
                self.config.scenario.prediction_horizon,
            )
        else:
            self.predicted_los_edges = set()
            for edge in supplied_edges:
                first, second = map(int, edge)
                if first == second or not (0 <= first < actual_count and 0 <= second < actual_count):
                    raise ValueError(f"invalid predicted LoS edge: {(first, second)}")
                self.predicted_los_edges.add(tuple(sorted((first, second))))
        self.step_count = 0
        self._reached.fill(False)
        self._active = self._present.copy()
        self._reached_step.fill(-1)
        supplied_reference = None if scenario is None else scenario.get("reference_route_lengths")
        self.reference_route_lengths = (
            np.linalg.norm(np.vstack([state.goal - state.position for state in real_states]), axis=1)
            if supplied_reference is None
            else np.asarray(supplied_reference, dtype=np.float64)
        )
        if self.reference_route_lengths.shape != (actual_count,) or np.any(self.reference_route_lengths <= 0.0):
            raise ValueError("reference_route_lengths must be positive with shape (actual_agent_count,)")
        self.reference_route_lengths = np.pad(self.reference_route_lengths, (0, self.config.environment.capacity - actual_count))
        return self._observation(), {
            "agent_count": actual_count,
            "actual_agent_count": actual_count,
            "max_agent_count": self.config.environment.capacity,
            "predicted_los_edges": tuple(sorted(self.predicted_los_edges)),
            "predicted_interaction_edges": tuple(sorted(self.predicted_los_edges)),
            "predicted_pair_data": [] if scenario is None else scenario.get("predicted_pair_data", []),
            "scenario_descriptor": {} if scenario is None else scenario.get("scenario_descriptor", {}),
            "reference_route_lengths": self.reference_route_lengths.copy(),
            "initial_positions": np.vstack([state.position for state in self.states]),
            "initial_velocities": np.vstack([state.velocity for state in self.states]),
            "goals": np.vstack([state.goal for state in self.states]),
            "maximum_accelerations": np.asarray([np.min(state.acceleration_limit) for state in self.states]),
            "maximum_velocities": np.asarray([np.min(state.velocity_limit) for state in self.states]),
            "actual_agent_mask": self._present.copy(),
            "distance_thresholds": {"d_safe": self.config.safety.d_safe, "d_warn": self.config.safety.d_warn, "d_eng": self.config.safety.d_eng},
            "goal_completion_mode": self.config.environment.goal_completion_mode,
            "scenario_id": None if scenario is None else scenario.get("scenario_id"),
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Compose reference/residual commands, filter them, and advance dynamics."""
        normalized = np.asarray(action, dtype=np.float64)
        if normalized.shape != self.action_space.shape:
            raise ValueError(f"action must have shape {self.action_space.shape}")
        active_before = self._active.copy()
        normalized = np.clip(normalized, -1.0, 1.0)
        normalized[~active_before] = 0.0
        reference = self.reference_actions()
        limits = np.vstack([state.acceleration_limit for state in self.states])
        residual, command, nominal = compose_action_numpy(
            normalized,
            reference,
            limits,
            self.config.control.actor_mode,
            self.config.control.residual_acceleration_fraction,
        )
        effective_normalized_residual = normalized if self.config.control.actor_mode == "residual" else np.zeros_like(normalized)
        reference[~active_before] = residual[~active_before] = command[~active_before] = 0.0
        nominal[~active_before] = 0.0
        safety_started = perf_counter()
        results, engaged_agents, engaged_pairs = self.safety_executor.solve(
            self.states, nominal, active_before, self.config.safety.enabled
        )
        # When pairwise HOCBF is disabled, SafetyExecutor returns a physical-only
        # acceleration/next-speed projection while still diagnosing the
        # counterfactual HOCBF intervention for learning.
        safe = np.vstack([result.safe_action for result in results])
        safe[~active_before] = 0.0
        pure_corrections = np.vstack([result.intervention_action - nominal[i] for i, result in enumerate(results)])
        runtime_corrections = safe - nominal
        self.last_safety_solve_time = perf_counter() - safety_started
        self.last_safety_worker_time_sum = self.safety_executor.last_worker_time_sum
        self.last_safety_max_agent_time = self.safety_executor.last_max_agent_time
        self.last_safety_backend = self.safety_executor.last_backend
        self.last_cbf_engaged_agent_count = engaged_agents
        self.last_cbf_engaged_pair_count = engaged_pairs
        previous = self.states
        self.states = self.dynamics.step(previous, safe, self.config.environment.dt_base, active_before)
        self.step_count += 1
        distances = np.array([np.linalg.norm(state.goal - state.position) for state in self.states])
        newly_reached = active_before & (distances <= self.config.environment.goal_tolerance)
        self._reached |= newly_reached
        self._reached_step[newly_reached] = self.step_count
        self._active = active_before & ~newly_reached
        reward, reward_terms = self.reward_model.compute(
            previous, self.states, newly_reached, active_before, self._present, effective_normalized_residual
        )
        cost, cost_terms = self.cost_model.compute(self.states, pure_corrections, active_before)
        terminated = bool(np.all(self._reached[self._present]))
        truncated = bool(self.step_count >= self.config.environment.max_steps and not terminated)
        active_ids = np.flatnonzero(active_before)
        pair_distances = [
            np.linalg.norm(self.states[first].position - self.states[second].position)
            for offset, first in enumerate(active_ids) for second in active_ids[offset + 1:]
        ]
        pair_tick_count = len(active_ids) * (len(active_ids) - 1) // 2
        separation_violations = sum(distance < self.config.safety.d_safe for distance in pair_distances)
        pure_squared = np.sum(pure_corrections[active_before] ** 2, axis=1)
        runtime_squared = np.sum(runtime_corrections[active_before] ** 2, axis=1)
        info = {
            "cost": cost,
            "reward_terms": reward_terms,
            "cost_terms": cost_terms,
            "reference_action": reference,
            "normalized_residual_action": effective_normalized_residual,
            "residual_action": residual,
            "command_action": command,
            "nominal_action": nominal,
            "safe_action": safe,
            "intervention_action": np.vstack([result.intervention_action for result in results]),
            "pure_safety_correction": pure_corrections,
            "runtime_filter_correction": runtime_corrections,
            "command_to_nominal_correction": nominal - command,
            "minimum_pair_distance": min(pair_distances, default=float("inf")),
            "d_safe_violation_count": separation_violations,
            "pair_tick_count": pair_tick_count,
            "active_agent_count": int(active_before.sum()),
            "active_mask_before": active_before,
            "active_mask_after": self._active.copy(),
            "newly_reached": newly_reached,
            "goal_reached_count": int(self._reached[self._present].sum()),
            "reached_step": self._reached_step.copy(),
            "pure_intervention_squared_sum": float(pure_squared.sum()),
            "runtime_correction_squared_sum": float(runtime_squared.sum()),
            "safety": [
                {
                    "active_neighbours": result.active_neighbours,
                    "skipped_neighbours": result.skipped_neighbours,
                    "infeasible": result.infeasible,
                    "resolver_used": result.resolver_used,
                    "emergency": result.emergency,
                    "max_urgency": result.max_urgency,
                    "responsibility_infeasible_count": result.responsibility_infeasible_count,
                }
                for result in results
            ],
        }
        return self._observation(), reward, terminated, truncated, info

    def close(self) -> None:
        self.safety_executor.close()
        super().close()
