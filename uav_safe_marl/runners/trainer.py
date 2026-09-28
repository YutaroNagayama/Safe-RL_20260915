"""High-level Gymnasium training runner."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
from time import perf_counter
from typing import Any, Callable, Sequence

import numpy as np

from uav_safe_marl.communication.message import P2PMessage
from uav_safe_marl.communication.p2p import P2PCommunication
from uav_safe_marl.config.schemas import ExperimentConfig
from uav_safe_marl.core.types import Transition
from uav_safe_marl.graph.unimp import UniMPEncoder
from uav_safe_marl.graph.observations import build_actor_graph, copy_graph
from uav_safe_marl.metrics.logger import ExperimentLogger
from uav_safe_marl.monitoring import LiveTrainingMonitor
from uav_safe_marl.replay import InitialStateBuffer, NumpyReplayBuffer
from uav_safe_marl.runtime import configure_runtime


class SafeRLTrainer:
    """Coordinate P2P/UniMP, shared actor, safety filtering, and CTDE updates."""

    CHECKPOINT_SCHEMA_VERSION = 3

    @staticmethod
    def _source_hash() -> str:
        package_root = Path(__file__).resolve().parents[1]
        digest = hashlib.sha256()
        for source in sorted(package_root.rglob("*.py")):
            digest.update(str(source.relative_to(package_root)).encode())
            digest.update(source.read_bytes())
        return digest.hexdigest()

    def __init__(self, config: ExperimentConfig, env: Any, output_dir: str | Path = "runs/latest") -> None:
        configure_runtime(config)
        self.config, self.env = config, env
        self.encoder = UniMPEncoder(config.communication.unimp_enabled)
        self.graph_mode = config.graph.actor_aggregation_backend == "graph"
        self.communication = P2PCommunication(config.communication.information_mode, config.communication.delay_steps, config.safety.d_eng, config.communication.actor_period_k)
        self.buffer = NumpyReplayBuffer(config.learning.buffer_size, config.seed)
        self.initial_state_buffer = InitialStateBuffer(config.learning.initial_state_buffer_size, config.seed + 31)
        scale = np.full((config.environment.capacity, 3), config.environment.acceleration_limit)
        if config.policy.backend.upper() == "SAC":
            from uav_safe_marl.algorithms.sac_backend import SACBackend

            self.backend = SACBackend(config, self.encoder.output_dim, config.environment.capacity, scale)
        else:
            from uav_safe_marl.algorithms.td3_backend import TD3Backend

            self.backend = TD3Backend(config, self.encoder.output_dim, config.environment.capacity, scale)
        self.logger = ExperimentLogger(output_dir, config)
        self.total_environment_steps = 0
        self.total_gradient_updates = 0
        self.completed_episodes = 0
        self.transitions_collected_by_n: dict[int, int] = {}
        self.scenario_bank_cursor = 0
        self.scenario_bank_hash: str | None = None
        self.source_hash = self._source_hash()

    @staticmethod
    def _merge_messages(cache: dict[int, dict[int, P2PMessage]], delivered: dict[int, list[P2PMessage]]) -> None:
        for receiver, messages in delivered.items():
            receiver_cache = cache.setdefault(receiver, {})
            for message in messages:
                previous = receiver_cache.get(message.sender_id)
                if message.embedding.size == 0 and previous is not None:
                    message = replace(message, embedding=previous.embedding, age_steps=previous.age_steps + 1)
                receiver_cache[message.sender_id] = message

    def _encoded(self, cache: dict[int, dict[int, P2PMessage]]) -> np.ndarray:
        active = self.env.active_mask
        references = self.env.reference_actions()
        encoded = np.vstack([self.encoder.encode(state, list(cache.get(i, {}).values()), bool(active[i]), references[i]) for i, state in enumerate(self.env.states)])
        encoded[~self.env.active_mask] = 0.0
        return encoded

    def _policy_observation(self, cache: dict[int, dict[int, P2PMessage]]):
        if self.graph_mode:
            return build_actor_graph(
                self.env.states, cache, self.config.graph.actor_embedding_dim,
                self.env.active_mask, self.env.reference_actions(),
            )
        return self._encoded(cache)

    @staticmethod
    def _purge_inactive_cache(cache: dict[int, dict[int, P2PMessage]], active_mask: np.ndarray) -> None:
        """Remove exited senders and receivers from the execution message state."""
        for receiver in list(cache):
            if not active_mask[receiver]:
                del cache[receiver]
                continue
            cache[receiver] = {sender: message for sender, message in cache[receiver].items() if active_mask[sender]}

    def _purge_invalid_cache(self, cache: dict[int, dict[int, P2PMessage]]) -> None:
        """Apply method-specific edge lifetime without changing message delay."""
        self._purge_inactive_cache(cache, self.env.active_mask)
        if self.config.communication.information_mode != "reactive_share":
            return
        for receiver, messages in list(cache.items()):
            cache[receiver] = {
                sender: message for sender, message in messages.items()
                if self.communication._linked(sender, receiver, self.env.states[sender], self.env.states[receiver])
            }

    def train(
        self,
        episodes: int | None = 1,
        reset_options: dict[str, Any] | None = None,
        scenarios: Sequence[dict[str, Any]] | Callable[[int], dict[str, Any]] | None = None,
        live_display: bool | None = None,
        checkpoint_path: str | Path | None = None,
        update_mode: str = "full",
    ) -> list[dict[str, Any]]:
        """Train for complete Gymnasium episodes and return episode summaries."""
        if update_mode not in {"full", "collect_only"}:
            raise ValueError("update_mode must be 'full' or 'collect_only'")
        history: list[dict[str, Any]] = []
        monitor = LiveTrainingMonitor(
            self.logger.output_dir,
            self.config,
            enabled=self.config.monitoring.enabled if live_display is None else live_display,
        )
        budget = self.config.training.total_environment_steps_budget
        if episodes is None and budget is None:
            raise ValueError("episodes may be null only when total_environment_steps_budget is configured")
        if episodes is not None and episodes < 0:
            raise ValueError("episodes must be non-negative")
        monitor.start(episodes, self.completed_episodes)
        if scenarios is not None and not callable(scenarios):
            serialized = json.dumps(list(scenarios), sort_keys=True, separators=(",", ":"), default=str).encode()
            supplied_hash = hashlib.sha256(serialized).hexdigest()
            if self.scenario_bank_hash is not None and self.scenario_bank_hash != supplied_hash:
                raise ValueError("scenario bank differs from the resumed checkpoint")
            self.scenario_bank_hash = supplied_hash
        episode = 0
        while (budget is not None and self.total_environment_steps < budget) or (
            budget is None and episodes is not None and episode < episodes
        ):
            episode_index = self.completed_episodes
            episode_options = reset_options
            if scenarios is not None:
                if callable(scenarios):
                    scenario = scenarios(episode_index)
                else:
                    if not scenarios:
                        raise ValueError("scenarios cannot be empty")
                    scenario = scenarios[self.scenario_bank_cursor % len(scenarios)]
                episode_options = {**(reset_options or {}), "scenario": scenario}
            _, reset_info = self.env.reset(seed=self.config.seed + episode_index, options=episode_options)
            self.communication.reset(reset_info["predicted_los_edges"])
            cache: dict[int, dict[int, P2PMessage]] = {}
            policy_observation = self._policy_observation(cache)
            start_observation = copy_graph(policy_observation) if self.graph_mode else policy_observation.copy()
            self.initial_state_buffer.add(
                start_observation,
                int(reset_info["actual_agent_count"]),
                reset_info.get("scenario_id"),
            )
            outgoing_embedding = np.zeros((self.config.environment.capacity, self.config.graph.actor_embedding_dim)) if self.graph_mode else policy_observation
            held_action = np.zeros(self.env.action_space.shape, dtype=np.float64)
            episode_reward = episode_cost = discounted_cost = 0.0
            discount = 1.0
            last_losses: dict[str, float] = {}
            totals = {key: 0.0 for key in ("goal_reward", "progress", "time_penalty", "absolute_distance_penalty", "residual_penalty", "distance_cost", "intervention_cost", "d_safe_violation_count", "cbf_intervention_count", "intervention_magnitude", "qp_infeasible_count", "conflict_resolution_count", "skipped_constraint_count", "emergency_fallback_count", "responsibility_infeasible_count", "actor_inference_time", "unimp_inference_time", "cbf_qp_solve_time", "cbf_qp_wall_time", "cbf_qp_worker_time_sum", "cbf_qp_max_agent_time", "cbf_engaged_agent_count", "cbf_engaged_pair_count", "neighbour_count")}
            minimum_distance, maximum_urgency, path_length = float("inf"), 0.0, 0.0
            minimum_goal_distances = np.asarray([
                np.linalg.norm(state.goal - state.position) for state in self.env.states
            ], dtype=np.float64)
            normalized_goal_distance_sum = 0.0
            acceleration_norms: list[float] = []
            per_agent_path = np.zeros(self.config.environment.capacity, dtype=np.float64)
            separation_violation_count = pair_tick_count = 0
            pure_intervention_squared_sum = runtime_correction_squared_sum = 0.0
            command_to_nominal_squared_sum = 0.0
            active_agent_ticks = 0
            action_norm_sums = {name: 0.0 for name in ("reference", "normalized_residual", "residual", "command", "nominal", "safe")}
            action_norm_max = {name: 0.0 for name in action_norm_sums}
            residual_saturated_components = 0
            episode_terminated = False
            last_replay_observation = start_observation
            last_nominal_action = np.zeros(self.env.action_space.shape, dtype=np.float64)
            pending: Transition | None = None

            def commit_pending(next_value, next_phase: int, next_held: np.ndarray) -> None:
                nonlocal pending, last_replay_observation, last_nominal_action, last_losses
                if pending is None:
                    return
                pending.next_state = copy_graph(next_value) if self.graph_mode else next_value.copy()
                pending.next_actor_phase = int(next_phase)
                pending.next_is_actor_decision_tick = bool(next_phase == 0)
                pending.next_held_residual_action = np.asarray(next_held).copy()
                self.buffer.add(pending)
                last_replay_observation = pending.state
                last_nominal_action = pending.nominal_action
                ready = (
                    update_mode == "full"
                    and
                    self.total_environment_steps >= self.config.training.learning_starts
                    and len(self.buffer) >= self.config.learning.batch_size
                )
                if ready and self.config.control.actor_mode != "reference_only":
                    for _ in range(self.config.training.updates_per_environment_step):
                        start_distribution = self.config.scenario.training_agent_count_distribution
                        initial_batch = (
                            self.initial_state_buffer.sample(
                                self.config.training.dual_batch_size,
                                start_distribution,
                            )
                            if self.config.learning.cal_variant == "episodic_dual_replay_gradient"
                            and self.initial_state_buffer.supports(start_distribution)
                            else None
                        )
                        cost_batch = (
                            self.buffer.sample_n_step(
                                self.config.learning.batch_size,
                                self.config.learning.cost_target_n_step,
                                self.config.learning.gamma_cost,
                            )
                            if self.config.learning.cost_target_n_step > 1
                            else None
                        )
                        last_losses.update(self.backend.update(
                            self.buffer.sample(self.config.learning.batch_size),
                            cost_batch=cost_batch,
                            initial_state_batch=initial_batch,
                            allow_dual_update=(
                                self.config.training.dual_update_mode == "gradient_step"
                                and self.total_environment_steps >= self.config.training.dual_learning_starts
                            ),
                        ))
                        self.total_gradient_updates += 1
                pending = None

            for step in range(self.config.environment.max_steps):
                # Causal order: receive previously sent messages, build the local
                # graph, produce z_i,t/action, then transmit z_i,t.  A message sent
                # here cannot be observed by a peer during this same tick.
                delivered = self.communication.receive(step, self.env.active_mask)
                self._merge_messages(cache, delivered)
                self._purge_invalid_cache(cache)
                encode_started = perf_counter(); policy_observation = self._policy_observation(cache); totals["unimp_inference_time"] += perf_counter() - encode_started
                # Finalize t-1 only after the exact communication snapshot used
                # for control at t has become observable.
                commit_pending(policy_observation, step % self.config.communication.actor_period_k, held_action)
                if step % self.config.communication.actor_period_k == 0 and self.config.control.actor_mode != "reference_only":
                    actor_started = perf_counter()
                    # Off-policy data collection explores with the stochastic SAC
                    # policy (or TD3 exploration noise).  execution_mode controls
                    # evaluation/deployment only; it must not silently disable
                    # exploration during training.
                    held_action = self.backend.act(policy_observation, deterministic=False)
                    if self.graph_mode and self.backend.last_actor_embedding is not None:
                        outgoing_embedding = self.backend.last_actor_embedding
                    elif not self.graph_mode:
                        outgoing_embedding = policy_observation
                    held_action[~self.env.active_mask] = 0.0
                    totals["actor_inference_time"] += perf_counter() - actor_started
                embeddings = outgoing_embedding if step % self.config.communication.actor_period_k == 0 else None
                self.communication.send(self.env.states, embeddings, step, self.env.active_mask)
                previous_positions = np.vstack([state.position for state in self.env.states])
                _, reward, terminated, truncated, info = self.env.step(held_action)
                episode_done = terminated or truncated
                # Gymnasium separates MDP termination from time-limit truncation;
                # critics must bootstrap through a truncation.
                state_to_store = copy_graph(policy_observation) if self.graph_mode else policy_observation.copy()
                actual_n = int(reset_info["actual_agent_count"])
                pending = Transition(
                    state_to_store, info["nominal_action"], info["safe_action"], reward,
                    info["cost"], None, terminated, truncated,
                    extras={"actual_agent_count": actual_n},
                    reference_action=info["reference_action"],
                    normalized_residual_action=info["normalized_residual_action"],
                    residual_action=info["residual_action"],
                    command_action=info["command_action"],
                    actor_phase=step % self.config.communication.actor_period_k,
                    is_actor_decision_tick=step % self.config.communication.actor_period_k == 0,
                    held_residual_action=info["normalized_residual_action"],
                )
                self.total_environment_steps += 1
                self.transitions_collected_by_n[actual_n] = self.transitions_collected_by_n.get(actual_n, 0) + 1
                monitor.update_step(
                    episode,
                    step,
                    self.config.environment.max_steps,
                    self.total_environment_steps,
                    self.total_gradient_updates,
                    len(self.buffer),
                    int(reset_info["actual_agent_count"]),
                )
                episode_reward += reward
                episode_cost += info["cost"]
                discounted_cost += discount * info["cost"]
                discount *= self.config.learning.gamma_cost
                current_positions = np.vstack([state.position for state in self.env.states])
                increments = np.linalg.norm(current_positions - previous_positions, axis=1)
                per_agent_path += increments * info["active_mask_before"]
                path_length += float((increments * info["active_mask_before"]).sum())
                minimum_distance = min(minimum_distance, info["minimum_pair_distance"])
                totals["distance_cost"] += info["cost_terms"]["distance_cost"]
                totals["intervention_cost"] += info["cost_terms"]["intervention_cost"]
                totals["goal_reward"] += info["reward_terms"]["goal_reward"]
                totals["progress"] += info["reward_terms"]["progress"]
                totals["time_penalty"] += info["reward_terms"]["time_penalty"]
                totals["absolute_distance_penalty"] += info["reward_terms"]["absolute_distance_penalty"]
                totals["residual_penalty"] += info["reward_terms"]["residual_penalty"]
                normalized_goal_distance_sum += info["reward_terms"]["mean_normalized_goal_distance"]
                goal_distances = np.asarray([
                    np.linalg.norm(state.goal - state.position) for state in self.env.states
                ], dtype=np.float64)
                active_before = info["active_mask_before"]
                minimum_goal_distances[active_before] = np.minimum(
                    minimum_goal_distances[active_before], goal_distances[active_before]
                )
                totals["d_safe_violation_count"] += info["d_safe_violation_count"]
                separation_violation_count += info["d_safe_violation_count"]
                pair_tick_count += info["pair_tick_count"]
                pure_intervention_squared_sum += info["pure_intervention_squared_sum"]
                runtime_correction_squared_sum += info["runtime_correction_squared_sum"]
                command_to_nominal_squared_sum += float(np.sum(info["command_to_nominal_correction"][active_before] ** 2))
                active_agent_ticks += info["active_agent_count"]
                diagnostic_actions = {
                    "reference": info["reference_action"],
                    "normalized_residual": info["normalized_residual_action"],
                    "residual": info["residual_action"],
                    "command": info["command_action"],
                    "nominal": info["nominal_action"],
                    "safe": info["safe_action"],
                }
                for name, values in diagnostic_actions.items():
                    norms = np.linalg.norm(values[active_before], axis=1)
                    action_norm_sums[name] += float(norms.sum())
                    action_norm_max[name] = max(action_norm_max[name], float(np.max(norms, initial=0.0)))
                residual_saturated_components += int(np.sum(np.abs(info["normalized_residual_action"][active_before]) >= 1.0 - 1e-6))
                totals["cbf_qp_solve_time"] += self.env.last_safety_solve_time
                totals["cbf_qp_wall_time"] += self.env.last_safety_solve_time
                totals["cbf_qp_worker_time_sum"] += self.env.last_safety_worker_time_sum
                totals["cbf_qp_max_agent_time"] = max(totals["cbf_qp_max_agent_time"], self.env.last_safety_max_agent_time)
                totals["cbf_engaged_agent_count"] += self.env.last_cbf_engaged_agent_count
                totals["cbf_engaged_pair_count"] += self.env.last_cbf_engaged_pair_count
                differences = np.linalg.norm(info["safe_action"] - info["nominal_action"], axis=1)
                totals["cbf_intervention_count"] += float(np.sum(differences > 1e-8))
                totals["intervention_magnitude"] += float(differences.sum())
                acceleration_norms.extend(np.linalg.norm(info["safe_action"][info["active_mask_before"]], axis=1).tolist())
                for safety in info["safety"]:
                    totals["qp_infeasible_count"] += float(safety["infeasible"])
                    totals["conflict_resolution_count"] += float(safety["resolver_used"])
                    totals["skipped_constraint_count"] += len(safety["skipped_neighbours"])
                    totals["emergency_fallback_count"] += float(safety["emergency"])
                    totals["responsibility_infeasible_count"] += safety["responsibility_infeasible_count"]
                    totals["neighbour_count"] += len(safety["active_neighbours"])
                    maximum_urgency = max(maximum_urgency, safety["max_urgency"])
                if episode_done:
                    # There is no next receive phase. Preserve Gymnasium's final
                    # observation and flush both termination and truncation.
                    self._purge_invalid_cache(cache)
                    terminal_observation = self._policy_observation(cache)
                    commit_pending(
                        terminal_observation,
                        (step + 1) % self.config.communication.actor_period_k,
                        held_action,
                    )
                    episode_terminated = terminated
                    break
            if (
                update_mode == "full"
                and self.config.training.dual_update_mode == "episode"
                and self.total_environment_steps >= self.config.training.dual_learning_starts
                and (episode_index + 1) % self.config.training.dual_update_interval_episodes == 0
            ):
                start_distribution = self.config.scenario.training_agent_count_distribution
                if self.initial_state_buffer.supports(start_distribution):
                    initial_batch = self.initial_state_buffer.sample(
                        self.config.training.dual_batch_size,
                        start_distribution,
                    )
                    last_losses.update(self.backend.update_dual_from_initial_states(initial_batch))
            dual = self.backend.dual.value
            start_state_ucb = self.backend.estimate_start_cost_ucb(start_observation)
            replay_state_ucb = float(last_losses.get("ucb_cost_estimate", self.backend.estimate_action_cost_ucb(last_replay_observation, last_nominal_action)))
            episode_duration = (step + 1) * self.config.environment.dt_base
            successful_path_stretch = (
                float(np.mean(per_agent_path[:reset_info["actual_agent_count"]] / reset_info["reference_route_lengths"][:reset_info["actual_agent_count"]]))
                if episode_terminated else None
            )
            scenario_descriptor = dict(reset_info.get("scenario_descriptor", {}))
            actual_count = int(reset_info["actual_agent_count"])
            final_goal_distances = np.asarray([
                np.linalg.norm(state.goal - state.position) for state in self.env.states[:actual_count]
            ], dtype=np.float64)
            flat_descriptor = {
                f"scenario_{key}": value
                for key, value in scenario_descriptor.items()
                if isinstance(value, (bool, int, float)) and value is not None
            }
            tracked_counts = sorted({2, 4, 8, 16, *self.transitions_collected_by_n})
            replay_occupancy = self.buffer.occupancy_by_agent_count()
            initial_occupancy = self.initial_state_buffer.occupancy_by_agent_count()
            n_statistics = {
                **{f"transitions_collected_N{n}": int(self.transitions_collected_by_n.get(n, 0)) for n in tracked_counts},
                **{f"replay_occupancy_N{n}": int(replay_occupancy.get(n, 0)) for n in tracked_counts},
                **{f"initial_state_occupancy_N{n}": int(initial_occupancy.get(n, 0)) for n in tracked_counts},
            }
            budget_overshoot = 0 if budget is None else max(0, self.total_environment_steps - budget)
            summary = {
                "episode": float(episode_index), "episode_reward": episode_reward, "episode_total_cost": episode_cost,
                "undiscounted_episode_cost": episode_cost, "discounted_cost": discounted_cost,
                "discounted_episode_cost": discounted_cost, "start_state_cost_ucb": start_state_ucb,
                "replay_state_cost_ucb": replay_state_ucb,
                "configured_d_cost": self.config.learning.d_cost, "cal_mode": self.config.learning.cal_mode,
                "cal_variant": self.config.learning.cal_variant,
                "constraint_semantics": self.config.learning.constraint_semantics,
                "cbf_parallel_backend": self.env.last_safety_backend,
                "cbf_worker_count": self.env.safety_executor.workers if self.env.last_safety_backend == "process" else 1,
                "lambda": dual, "message_count": float(self.communication.message_count),
                "transmitted_bytes": float(self.communication.transmitted_bytes), **last_losses,
                "minimum_pair_distance": minimum_distance,
                "episode_success": float(episode_terminated),
                "agent_count": float(reset_info["actual_agent_count"]),
                "actual_agent_count": int(reset_info["actual_agent_count"]),
                "max_agent_count": int(reset_info["max_agent_count"]),
                "scenario_id": reset_info.get("scenario_id"),
                "scenario_descriptor": scenario_descriptor,
                "goal_reached_count": float(np.sum(self.env._reached[self.env._present])),
                "separation_violation_rate": separation_violation_count / max(1, pair_tick_count),
                "separation_violation_pair_ticks": float(separation_violation_count),
                "active_pair_ticks": float(pair_tick_count),
                "path_stretch": successful_path_stretch,
                "pure_safety_intervention_magnitude": pure_intervention_squared_sum / max(1, active_agent_ticks),
                "runtime_filter_correction": runtime_correction_squared_sum / max(1, active_agent_ticks),
                "command_to_nominal_correction": command_to_nominal_squared_sum / max(1, active_agent_ticks),
                "active_agent_ticks": float(active_agent_ticks),
                "communication_load": float(self.communication.transmitted_bytes) / max(self.config.environment.dt_base, active_agent_ticks * self.config.environment.dt_base),
                "cbf_intervention_rate": totals["cbf_intervention_count"] / max(1, active_agent_ticks),
                "max_urgency": maximum_urgency,
                "goal_success_rate": float(np.mean(self.env._reached[self.env._present])),
                "mean_normalized_goal_distance": normalized_goal_distance_sum / max(1, step + 1),
                "final_goal_distance_mean": float(np.mean(final_goal_distances)),
                "final_goal_distance_max": float(np.max(final_goal_distances)),
                "minimum_goal_distance_mean": float(np.mean(minimum_goal_distances[:actual_count])),
                "minimum_goal_distance_per_agent": minimum_goal_distances[:actual_count].tolist(),
                "flight_time": episode_duration,
                "path_length": path_length,
                "mean_acceleration": float(np.mean(acceleration_norms)) if acceleration_norms else 0.0,
                "max_acceleration": float(np.max(acceleration_norms)) if acceleration_norms else 0.0,
                "std_acceleration": float(np.std(acceleration_norms)) if acceleration_norms else 0.0,
                "cumulative_environment_steps": int(self.total_environment_steps),
                "gradient_updates": int(self.total_gradient_updates),
                "replay_size": int(len(self.buffer)),
                "total_environment_steps_budget": budget,
                "environment_steps_budget_overshoot": int(budget_overshoot),
                "completed_episodes": int(self.completed_episodes + 1),
                **{f"{name}_action_norm_mean": total / max(1, active_agent_ticks) for name, total in action_norm_sums.items()},
                **{f"{name}_action_norm_max": value for name, value in action_norm_max.items()},
                "residual_saturation_rate": residual_saturated_components / max(1, 3 * active_agent_ticks),
                **n_statistics,
                **flat_descriptor,
                **totals,
            }
            self.logger.log(summary)
            history.append(summary)
            self.completed_episodes += 1
            self.scenario_bank_cursor += 1
            if checkpoint_path is not None and self.completed_episodes % self.config.training.checkpoint_every_episodes == 0:
                self.save_checkpoint(checkpoint_path)
            monitor.end_episode(summary, self.total_environment_steps, self.total_gradient_updates, len(self.buffer))
            episode += 1
        monitor.close()
        return history

    def save_checkpoint(self, path: str | Path) -> None:
        """Atomically save an episode-boundary, exact training-resume snapshot."""
        import torch

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        config_payload = self.config.to_dict()
        config_hash = hashlib.sha256(json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        payload = {
            "schema_version": self.CHECKPOINT_SCHEMA_VERSION,
            "backend": self.backend.state_dict(),
            "config": config_payload,
            "experiment_identity": {
                "config_sha256": config_hash,
                "scenario_bank_sha256": self.scenario_bank_hash,
                "training_seed": self.config.seed,
                "source_sha256": self.source_hash,
                "method": {
                    "actor_mode": self.config.control.actor_mode,
                    "communication_mode": self.config.communication.information_mode,
                    "actor_backend": self.config.graph.actor_aggregation_backend,
                    "critic_backend": self.config.graph.critic_aggregation_backend,
                    "runtime_hocbf": self.config.safety.enabled,
                    "intervention_cost": self.config.objective.intervention_cost_enabled,
                    "cal_variant": self.config.learning.cal_variant,
                },
            },
            "trainer_progress": {
                "total_environment_steps": self.total_environment_steps,
                "total_gradient_updates": self.total_gradient_updates,
                "completed_episodes": self.completed_episodes,
                "transitions_collected_by_n": self.transitions_collected_by_n,
                "scenario_bank_cursor": self.scenario_bank_cursor,
            },
            "replay_buffer": self.buffer.state_dict(),
            "initial_state_buffer": self.initial_state_buffer.state_dict(),
            "rng_state": {
                "python": random.getstate(),
                "numpy_global": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        temporary = destination.with_name(destination.name + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, destination)

    def load_checkpoint(self, path: str | Path, *, resume_training: bool = True) -> None:
        """Load either an exact training resume or weights for evaluation."""
        import torch

        payload = torch.load(Path(path), map_location=self.backend.device, weights_only=False)
        version = payload.get("schema_version")
        if version not in {2, self.CHECKPOINT_SCHEMA_VERSION}:
            raise ValueError(
                f"incompatible checkpoint schema {version!r}; expected {self.CHECKPOINT_SCHEMA_VERSION}. "
                "Legacy full-action checkpoints are intentionally not converted."
            )
        if resume_training:
            current_config = self.config.to_dict()
            current_hash = hashlib.sha256(json.dumps(current_config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            expected_hash = payload.get("experiment_identity", {}).get("config_sha256")
            if expected_hash != current_hash:
                raise ValueError("checkpoint config differs from the current resolved experiment config")
            expected_source = payload.get("experiment_identity", {}).get("source_sha256")
            if expected_source != self.source_hash:
                raise ValueError("checkpoint source hash differs from the current implementation")
        self.backend.load_state_dict(payload["backend"])
        if not resume_training:
            return
        self.buffer.load_state_dict(payload["replay_buffer"])
        if "initial_state_buffer" in payload:
            self.initial_state_buffer.load_state_dict(payload["initial_state_buffer"])
        progress = payload.get("trainer_progress", {})
        self.total_environment_steps = int(progress.get("total_environment_steps", 0))
        self.total_gradient_updates = int(progress.get("total_gradient_updates", 0))
        self.completed_episodes = int(progress.get("completed_episodes", 0))
        self.transitions_collected_by_n = {
            int(key): int(value) for key, value in progress.get("transitions_collected_by_n", {}).items()
        }
        self.scenario_bank_cursor = int(progress.get("scenario_bank_cursor", self.completed_episodes))
        self.scenario_bank_hash = payload.get("experiment_identity", {}).get("scenario_bank_sha256")
        rng = payload["rng_state"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy_global"])
        torch.set_rng_state(rng["torch_cpu"].cpu())
        if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in rng["torch_cuda"]])
