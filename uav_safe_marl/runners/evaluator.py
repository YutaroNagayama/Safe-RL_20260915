"""Deterministic paired-scenario policy evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from uav_safe_marl.graph.observations import build_actor_graph, copy_graph
from uav_safe_marl.graph.feature_layout import EDGE_INFORMATION_AGE_INDEX


def evaluate(
    trainer: Any,
    episodes: int = 1,
    reset_options: dict[str, Any] | None = None,
    *,
    policy_execution_mode: str | None = None,
    pair_initial_action_with_cost: bool = False,
    capture_critic_inputs: bool = False,
) -> dict[str, Any]:
    """Evaluate and retain enough raw rollout data to recompute all metrics."""
    execution_mode = policy_execution_mode or trainer.config.policy.execution_mode
    if execution_mode not in {"mean", "stochastic"}:
        raise ValueError("policy_execution_mode must be mean or stochastic")
    if pair_initial_action_with_cost and trainer.config.control.actor_mode == "reference_only":
        raise ValueError("paired Cost calibration requires an Actor action")
    returns: list[float] = []
    costs: list[float] = []
    trajectories: list[np.ndarray] = []
    rollouts: list[dict[str, Any]] = []
    episode_metrics: list[dict[str, Any]] = []
    for episode in range(episodes):
        _, reset_info = trainer.env.reset(seed=trainer.config.seed + 10_000 + episode, options=reset_options)
        trainer.communication.reset(reset_info["predicted_los_edges"])
        cache = {}
        policy_observation = trainer._policy_observation(cache)
        paired_initial_action = None
        if pair_initial_action_with_cost:
            paired_initial_action, start_cost_statistics = trainer.backend.sample_start_action_cost_statistics(
                policy_observation,
                deterministic=execution_mode == "mean",
            )
        else:
            start_cost_statistics = trainer.backend.estimate_start_cost_statistics(
                policy_observation,
                deterministic=execution_mode == "mean",
            )
        outgoing_embedding = np.zeros((trainer.config.environment.capacity, trainer.config.graph.actor_embedding_dim)) if trainer.graph_mode else policy_observation
        held = np.zeros(trainer.env.action_space.shape)
        total_reward = total_cost = discounted_cost = 0.0
        cost_discount = 1.0
        capacity = trainer.config.environment.capacity
        candidate_adjacency = np.zeros((capacity, capacity), dtype=bool)
        for first, second in reset_info["predicted_interaction_edges"]:
            candidate_adjacency[first, second] = candidate_adjacency[second, first] = True
        actual_minimum_distances = np.full((capacity, capacity), np.inf, dtype=np.float64)
        cbf_ever_adjacency = np.zeros((capacity, capacity), dtype=bool)
        candidate_edge_lifetime_ticks = np.zeros((capacity, capacity), dtype=np.int64)
        positions = [np.vstack([state.position for state in trainer.env.states])]
        velocities = [np.vstack([state.velocity for state in trainer.env.states])]
        active_masks = [trainer.env.active_mask]
        nominal_actions: list[np.ndarray] = []
        policy_actions: list[np.ndarray] = []
        reference_actions: list[np.ndarray] = []
        normalized_residual_actions: list[np.ndarray] = []
        residual_actions: list[np.ndarray] = []
        command_actions: list[np.ndarray] = []
        safe_actions: list[np.ndarray] = []
        intervention_actions: list[np.ndarray] = []
        newly_reached_masks: list[np.ndarray] = []
        pairwise_distances: list[np.ndarray] = []
        actor_adjacencies: list[np.ndarray] = []
        cbf_adjacencies: list[np.ndarray] = []
        information_ages: list[np.ndarray] = []
        transmitted_byte_deltas: list[int] = []
        reward_components: list[dict[str, float]] = []
        cost_components: list[dict[str, float]] = []
        step_rewards: list[float] = []
        step_costs: list[float] = []
        policy_observations: list[Any] = []
        actor_phases: list[int] = []
        per_agent_path = np.zeros(trainer.config.environment.capacity, dtype=np.float64)
        violation_count = pair_tick_count = active_agent_ticks = 0
        pure_squared_sum = runtime_squared_sum = 0.0
        terminated = truncated = False
        for step in range(trainer.config.environment.max_steps):
            delivered = trainer.communication.receive(step, trainer.env.active_mask)
            trainer._merge_messages(cache, delivered)
            trainer._purge_invalid_cache(cache)
            policy_observation = trainer._policy_observation(cache)
            if capture_critic_inputs:
                policy_observations.append(
                    copy_graph(policy_observation) if trainer.graph_mode else policy_observation.copy()
                )
                actor_phases.append(step % trainer.config.communication.actor_period_k)
            communication_graph = policy_observation if trainer.graph_mode else build_actor_graph(
                trainer.env.states, cache, trainer.config.graph.actor_embedding_dim,
                trainer.env.active_mask, trainer.env.reference_actions(),
            )
            actor_adjacencies.append(communication_graph["adjacency"].copy())
            ages = np.full(communication_graph["adjacency"].shape, -1.0, dtype=np.float64)
            ages[communication_graph["adjacency"]] = communication_graph["edge_features"][..., EDGE_INFORMATION_AGE_INDEX][communication_graph["adjacency"]]
            information_ages.append(ages)
            pre_positions = np.vstack([state.position for state in trainer.env.states])
            pre_distances = np.linalg.norm(pre_positions[:, None, :] - pre_positions[None, :, :], axis=-1)
            active_pair = trainer.env.active_mask[:, None] & trainer.env.active_mask[None, :]
            cbf_now = active_pair & (pre_distances <= trainer.config.safety.d_eng) & ~np.eye(len(pre_positions), dtype=bool)
            cbf_adjacencies.append(cbf_now)
            cbf_ever_adjacency |= cbf_now
            candidate_edge_lifetime_ticks += (candidate_adjacency & active_pair).astype(np.int64)
            if step % trainer.config.communication.actor_period_k == 0 and trainer.config.control.actor_mode != "reference_only":
                if step == 0 and paired_initial_action is not None:
                    held = paired_initial_action.copy()
                else:
                    held = trainer.backend.act(
                        policy_observation,
                        deterministic=execution_mode == "mean",
                    )
                outgoing_embedding = trainer.backend.last_actor_embedding if trainer.graph_mode else policy_observation
            held[~trainer.env.active_mask] = 0.0
            embeddings = outgoing_embedding if step % trainer.config.communication.actor_period_k == 0 else None
            bytes_before = trainer.communication.transmitted_bytes
            trainer.communication.send(trainer.env.states, embeddings, step, trainer.env.active_mask)
            transmitted_byte_deltas.append(trainer.communication.transmitted_bytes - bytes_before)
            previous_positions = positions[-1]
            _, reward, terminated, truncated, info = trainer.env.step(held)
            trainer._purge_invalid_cache(cache)
            policy_observation = trainer._policy_observation(cache)
            if not trainer.graph_mode:
                outgoing_embedding = policy_observation
            current_positions = np.vstack([state.position for state in trainer.env.states])
            current_velocities = np.vstack([state.velocity for state in trainer.env.states])
            current_pair_distances = np.linalg.norm(current_positions[:, None, :] - current_positions[None, :, :], axis=-1)
            actual_minimum_distances[active_pair] = np.minimum(actual_minimum_distances[active_pair], current_pair_distances[active_pair])
            increments = np.linalg.norm(current_positions - previous_positions, axis=1)
            per_agent_path += increments * info["active_mask_before"]
            violation_count += info["d_safe_violation_count"]
            pair_tick_count += info["pair_tick_count"]
            active_agent_ticks += info["active_agent_count"]
            pure_squared_sum += info["pure_intervention_squared_sum"]
            runtime_squared_sum += info["runtime_correction_squared_sum"]
            total_reward += reward
            total_cost += info["cost"]
            discounted_cost += cost_discount * info["cost"]
            cost_discount *= trainer.config.learning.gamma_cost
            positions.append(current_positions)
            velocities.append(current_velocities)
            active_masks.append(info["active_mask_after"].copy())
            nominal_actions.append(info["nominal_action"].copy())
            policy_actions.append(held.copy())
            reference_actions.append(info["reference_action"].copy())
            normalized_residual_actions.append(info["normalized_residual_action"].copy())
            residual_actions.append(info["residual_action"].copy())
            command_actions.append(info["command_action"].copy())
            safe_actions.append(info["safe_action"].copy())
            intervention_actions.append(info["intervention_action"].copy())
            newly_reached_masks.append(info["newly_reached"].copy())
            pairwise_distances.append(current_pair_distances)
            reward_components.append(dict(info["reward_terms"]))
            cost_components.append(dict(info["cost_terms"]))
            step_rewards.append(float(reward))
            step_costs.append(float(info["cost"]))
            if terminated or truncated:
                break
        actual_count = int(reset_info["actual_agent_count"])
        position_history = np.stack(positions)
        goals = np.asarray(reset_info["goals"], dtype=np.float64)
        goal_distances = np.linalg.norm(position_history - goals[None, ...], axis=2)
        final_goal_distances = goal_distances[-1, :actual_count]
        minimum_goal_distances = np.min(goal_distances[:, :actual_count], axis=0)
        path_stretch = float(np.mean(per_agent_path[:actual_count] / reset_info["reference_route_lengths"][:actual_count])) if terminated else None
        duration = (step + 1) * trainer.config.environment.dt_base
        upper = np.triu(np.ones((capacity, capacity), dtype=bool), 1)
        candidate_miss_adjacency = cbf_ever_adjacency & ~candidate_adjacency
        pair_results = [
            {
                "first": first,
                "second": second,
                "candidate": bool(candidate_adjacency[first, second]),
                "actual_minimum_distance": float(actual_minimum_distances[first, second]),
                "cbf_engaged": bool(cbf_ever_adjacency[first, second]),
                "candidate_miss": bool(candidate_miss_adjacency[first, second]),
                "candidate_edge_lifetime_ticks": int(candidate_edge_lifetime_ticks[first, second]),
            }
            for first in range(actual_count) for second in range(first + 1, actual_count)
        ]
        scenario_descriptor = dict(reset_info.get("scenario_descriptor", {}))
        metrics = {
            "scenario_id": reset_info.get("scenario_id"),
            "actual_agent_count": actual_count,
            "max_agent_count": int(reset_info["max_agent_count"]),
            "episode_success": float(terminated),
            "goal_reached_count": int(trainer.env._reached.sum()),
            "separation_violation_pair_ticks": int(violation_count),
            "active_pair_ticks": int(pair_tick_count),
            "separation_violation_rate": violation_count / max(1, pair_tick_count),
            "path_stretch": path_stretch,
            "pure_intervention_squared_sum": pure_squared_sum,
            "runtime_correction_squared_sum": runtime_squared_sum,
            "active_agent_ticks": int(active_agent_ticks),
            "pure_safety_intervention_magnitude": pure_squared_sum / max(1, active_agent_ticks),
            "runtime_filter_correction": runtime_squared_sum / max(1, active_agent_ticks),
            "transmitted_bytes": int(trainer.communication.transmitted_bytes),
            "active_agent_seconds": active_agent_ticks * trainer.config.environment.dt_base,
            "communication_load": trainer.communication.transmitted_bytes / max(trainer.config.environment.dt_base, active_agent_ticks * trainer.config.environment.dt_base),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "episode_duration": duration,
            "undiscounted_episode_cost": float(total_cost),
            "discounted_episode_cost": float(discounted_cost),
            "start_state_cost_members": start_cost_statistics["members"],
            "start_state_cost_mean": start_cost_statistics["mean"],
            "start_state_cost_std": start_cost_statistics["std"],
            "start_state_cost_ucb": start_cost_statistics["ucb"],
            "policy_execution_mode": execution_mode,
            "initial_action_paired_with_cost_estimate": bool(pair_initial_action_with_cost),
            "cost_mean_error": float(start_cost_statistics["mean"] - discounted_cost),
            "cost_ucb_error": float(start_cost_statistics["ucb"] - discounted_cost),
            "empirical_ucb_covers_return": bool(discounted_cost <= start_cost_statistics["ucb"]),
            "distance_cost": float(sum(item["distance_cost"] for item in cost_components)),
            "intervention_cost": float(sum(item["intervention_cost"] for item in cost_components)),
            "scenario_descriptor": scenario_descriptor,
            "rollout_pair_results": pair_results,
            "cbf_engaged_pair_count": int(np.sum(cbf_ever_adjacency & upper)),
            "candidate_miss_count": int(np.sum(candidate_miss_adjacency & upper)),
            "goal_reward": float(sum(item["goal_reward"] for item in reward_components)),
            "progress": float(sum(item["progress"] for item in reward_components)),
            "time_penalty": float(sum(item["time_penalty"] for item in reward_components)),
            "absolute_distance_penalty": float(sum(item["absolute_distance_penalty"] for item in reward_components)),
            "residual_penalty": float(sum(item["residual_penalty"] for item in reward_components)),
            "mean_normalized_goal_distance": float(np.mean([item["mean_normalized_goal_distance"] for item in reward_components])),
            "final_goal_distance_mean": float(np.mean(final_goal_distances)),
            "final_goal_distance_max": float(np.max(final_goal_distances)),
            "minimum_goal_distance_mean": float(np.mean(minimum_goal_distances)),
            "minimum_goal_distance_per_agent": minimum_goal_distances.tolist(),
            **{
                f"scenario_{key}": value
                for key, value in scenario_descriptor.items()
                if isinstance(value, (bool, int, float)) and value is not None
            },
        }
        if paired_initial_action is not None:
            actual_initial_action = policy_actions[0]
            conditioned_nominal = np.asarray(start_cost_statistics["conditioned_nominal_action"])
            actual_initial_nominal = nominal_actions[0]
            execution_error = float(np.max(np.abs(np.asarray(paired_initial_action) - actual_initial_action)))
            nominal_error = float(np.max(np.abs(conditioned_nominal - actual_initial_nominal)))
            metrics.update({
                "conditioned_initial_execution_action": np.asarray(paired_initial_action).tolist(),
                "conditioned_initial_nominal_action": conditioned_nominal.tolist(),
                "rollout_initial_execution_action": np.asarray(actual_initial_action).tolist(),
                "rollout_initial_nominal_action": np.asarray(actual_initial_nominal).tolist(),
                "paired_initial_execution_action_max_abs_error": execution_error,
                "paired_initial_nominal_action_max_abs_error": nominal_error,
                # Execution actions are the same sampled array.  Nominal
                # acceleration is reconstructed once by Torch/float32 for the
                # Critic and once by NumPy/float64 in the environment, so use a
                # tolerance that accepts only the expected conversion noise.
                "paired_initial_action_matches_rollout": bool(np.allclose(paired_initial_action, actual_initial_action, rtol=1e-6, atol=1e-6)),
                "paired_initial_nominal_matches_rollout": bool(np.allclose(conditioned_nominal, actual_initial_nominal, rtol=1e-5, atol=1e-5)),
            })
        rollout = {
            "reset_info": reset_info,
            "positions": position_history,
            "velocities": np.stack(velocities),
            "active_masks": np.stack(active_masks),
            "nominal_actions": np.stack(nominal_actions),
            "policy_actions": np.stack(policy_actions),
            "reference_actions": np.stack(reference_actions),
            "normalized_residual_actions": np.stack(normalized_residual_actions),
            "residual_actions": np.stack(residual_actions),
            "command_actions": np.stack(command_actions),
            "safe_actions": np.stack(safe_actions),
            "intervention_actions": np.stack(intervention_actions),
            "pure_cbf_actions": np.stack(intervention_actions),
            "newly_reached_masks": np.stack(newly_reached_masks),
            "pairwise_distances": np.stack(pairwise_distances),
            "actor_adjacencies": np.stack(actor_adjacencies),
            "cbf_adjacencies": np.stack(cbf_adjacencies),
            "information_ages": np.stack(information_ages),
            "transmitted_byte_deltas": np.asarray(transmitted_byte_deltas),
            "rewards": np.asarray(step_rewards),
            "costs": np.asarray(step_costs),
            "goal_distances": goal_distances,
            "reward_components": reward_components,
            "cost_components": cost_components,
            "reached_step": trainer.env._reached_step.copy(),
            "maximum_accelerations": reset_info["maximum_accelerations"].copy(),
            "maximum_velocities": reset_info["maximum_velocities"].copy(),
            "distance_thresholds": dict(reset_info["distance_thresholds"]),
            "candidate_adjacency": candidate_adjacency,
            "actual_minimum_pair_distances": actual_minimum_distances,
            "cbf_ever_adjacency": cbf_ever_adjacency,
            "candidate_miss_adjacency": candidate_miss_adjacency,
            "candidate_edge_lifetime_ticks": candidate_edge_lifetime_ticks,
            "rollout_pair_results": pair_results,
        }
        if capture_critic_inputs:
            rollout["policy_observations"] = policy_observations
            rollout["actor_phases"] = np.asarray(actor_phases, dtype=np.int64)
        returns.append(total_reward)
        costs.append(total_cost)
        trajectories.append(rollout["positions"])
        rollouts.append(rollout)
        episode_metrics.append(metrics)

    total_violations = sum(item["separation_violation_pair_ticks"] for item in episode_metrics)
    total_pair_ticks = sum(item["active_pair_ticks"] for item in episode_metrics)
    total_pure = sum(item["pure_intervention_squared_sum"] for item in episode_metrics)
    total_active_ticks = sum(item["active_agent_ticks"] for item in episode_metrics)
    total_bytes = sum(item["transmitted_bytes"] for item in episode_metrics)
    successful_stretches = [item["path_stretch"] for item in episode_metrics if item["path_stretch"] is not None]
    primary = {
        "episode_success_rate": float(np.mean([item["episode_success"] for item in episode_metrics])),
        "separation_violation_rate": total_violations / max(1, total_pair_ticks),
        "path_stretch": float(np.mean(successful_stretches)) if successful_stretches else None,
        "pure_safety_intervention_magnitude": total_pure / max(1, total_active_ticks),
        "communication_load": total_bytes / max(trainer.config.environment.dt_base, total_active_ticks * trainer.config.environment.dt_base),
    }
    return {
        "mean_reward": float(np.mean(returns)),
        "mean_cost": float(np.mean(costs)),
        "primary_metrics": primary,
        "episode_metrics": episode_metrics,
        "trajectories": trajectories,
        "rollouts": rollouts,
    }
