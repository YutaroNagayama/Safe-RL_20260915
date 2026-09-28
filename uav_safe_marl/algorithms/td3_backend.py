"""Deterministic TD3/CAL-inspired backend."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from uav_safe_marl.config.schemas import ExperimentConfig
from uav_safe_marl.control import compose_action_torch
from uav_safe_marl.critics.cost_ensemble import CostCriticEnsemble
from uav_safe_marl.critics.graph_critic import GraphCostCriticEnsemble, GraphScalarCritic
from uav_safe_marl.critics.reward_critic import RewardCritic
from uav_safe_marl.graph.neural import centralized_adjacency
from uav_safe_marl.graph.feature_layout import NODE_FEATURES
from uav_safe_marl.policies.base import graph_to_torch, resolve_device, require_torch
from uav_safe_marl.policies.graph_actor import GraphTD3Actor
from uav_safe_marl.policies.td3_actor import TD3Actor
from .cal import TorchCAL
from .primal_dual import ProjectedDual
from .multirate import augment_critic_state, frozen_parameters, held_nominal_action, select_decision_action

torch = require_torch()
F = torch.nn.functional


class TD3Backend:
    """TD3 backbone retaining CAL Cost-UCB and augmented penalty."""

    def __init__(self, config: ExperimentConfig, observation_dim: int, agent_count: int, acceleration_scale: np.ndarray) -> None:
        self.config, self.agent_count = config, agent_count
        torch.manual_seed(config.seed)
        self.device = resolve_device(config.device)
        hidden = config.policy.hidden_dim
        self.graph_mode = config.graph.actor_aggregation_backend == "graph"
        if self.graph_mode:
            graph = config.graph
            self.actor = GraphTD3Actor(7 + graph.actor_embedding_dim, graph.actor_hidden_dim, graph.actor_embedding_dim, graph.actor_layers, graph.actor_attention_heads, hidden).to(self.device)
        else:
            self.actor = TD3Actor(observation_dim, hidden).to(self.device)
        self.actor_target = deepcopy(self.actor)
        if self.graph_mode:
            critic_args = (config.graph.critic_hidden_dim, config.graph.critic_embedding_dim, config.graph.critic_layers, config.graph.critic_attention_heads)
            self.reward_critics = torch.nn.ModuleList([GraphScalarCritic(*critic_args), GraphScalarCritic(*critic_args)]).to(self.device)
            self.cost_critics = GraphCostCriticEnsemble(config.learning.cost_ensemble_size, *critic_args).to(self.device)
        else:
            global_dim = (observation_dim + NODE_FEATURES.critic_timing_width) * agent_count
            action_dim = NODE_FEATURES.action_width * agent_count
            self.reward_critics = torch.nn.ModuleList([RewardCritic(global_dim, action_dim, hidden), RewardCritic(global_dim, action_dim, hidden)]).to(self.device)
            self.cost_critics = CostCriticEnsemble(config.learning.cost_ensemble_size, global_dim, action_dim, hidden).to(self.device)
        self.reward_targets, self.cost_targets = deepcopy(self.reward_critics), deepcopy(self.cost_critics)
        self.critic_optimizer = torch.optim.Adam(list(self.reward_critics.parameters()) + list(self.cost_critics.parameters()), lr=config.policy.learning_rate)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.policy.learning_rate)
        self.dual = ProjectedDual(config.learning.cal_lambda_initial, config.learning.cal_lambda_lr)
        self.last_actor_embedding: np.ndarray | None = None
        self.cal = TorchCAL(config.learning.d_cost, config.learning.cal_coefficient, config.learning.beta_ucb, config.learning.cal_mode)
        self.tau, self.update_count = 0.005, 0
        self.cost_ucb_sample_count = 0
        self.cost_ucb_above_limit_count = 0
        self.dual_update_count = 0
        self.lambda_positive_update_count = 0

    def act(self, observations: np.ndarray, deterministic: bool = True) -> np.ndarray:
        with torch.no_grad():
            if self.graph_mode:
                graph = graph_to_torch(observations, self.device)
                action, embedding = self.actor(graph)
                if not deterministic:
                    action = (action + 0.1 * torch.randn_like(action)).clamp(-1.0, 1.0) * graph["node_mask"].unsqueeze(-1)
                self.last_actor_embedding = embedding[0].cpu().numpy()
                return action[0].cpu().numpy()
            tensor = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
            action = self.actor(tensor)
            if not deterministic:
                action = (action + 0.1 * torch.randn_like(action)).clamp(-1.0, 1.0)
            action = action * tensor[..., NODE_FEATURES.active_mask].clamp(0.0, 1.0)
        return action.cpu().numpy()

    def _joint(self, actor, observations):
        if self.graph_mode:
            normalized, embedding = actor(observations)
            action = compose_action_torch(
                normalized, observations["node_features"], self.config.control.actor_mode,
                self.config.control.residual_acceleration_fraction,
                self.config.control.reference_speed_fraction,
                self.config.control.reference_velocity_gain,
            ) * observations["node_mask"].unsqueeze(-1)
            return action, embedding
        batch, agents, width = observations.shape
        mask = observations[..., NODE_FEATURES.active_mask].clamp(0.0, 1.0)
        normalized = actor(observations.reshape(batch * agents, width)).reshape(batch, agents, 3)
        action = compose_action_torch(
            normalized, observations, self.config.control.actor_mode,
            self.config.control.residual_acceleration_fraction,
            self.config.control.reference_speed_fraction,
            self.config.control.reference_velocity_gain,
        ) * mask
        return action.flatten(1), None

    def _critic_adjacency(self, graph):
        return centralized_adjacency(graph, self.config.graph.critic_edge_mode, self.config.safety.d_eng)

    def _reward_values(self, critics, state, action):
        if self.graph_mode:
            adjacency = self._critic_adjacency(state)
            return [critic(state, action, adjacency) for critic in critics]
        return [critic(state.flatten(1), action) for critic in critics]

    def _cost_values(self, critics, state, action):
        if self.graph_mode:
            return critics(state, action, self._critic_adjacency(state))
        return critics(state.flatten(1), action)

    def _initial_constraint_values(self, batch):
        state = graph_to_torch(batch["state"], self.device) if self.graph_mode else torch.as_tensor(batch["state"], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action, _ = self._joint(self.actor, state)
            agents = state["node_features"].shape[1] if self.graph_mode else state.shape[1]
            batch_size = state["node_features"].shape[0] if self.graph_mode else state.shape[0]
            held = torch.zeros((batch_size, agents, 3), device=self.device)
            phase = torch.zeros((batch_size, 1), device=self.device)
            critic_state = augment_critic_state(state, held, phase, self.config.communication.actor_period_k, self.graph_mode)
            members = self._cost_values(self.cost_critics, critic_state, action)
            return members, self.cal.ucb(members)

    def update(
        self,
        batch: dict[str, np.ndarray],
        *,
        initial_state_batch: dict | None = None,
        allow_dual_update: bool = True,
    ) -> dict[str, float]:
        self.update_count += 1
        data = {key: torch.as_tensor(value, dtype=torch.float32, device=self.device) for key, value in batch.items() if key not in {"state", "next_state"}}
        data["state"] = graph_to_torch(batch["state"], self.device) if self.graph_mode else torch.as_tensor(batch["state"], dtype=torch.float32, device=self.device)
        data["next_state"] = graph_to_torch(batch["next_state"], self.device) if self.graph_mode else torch.as_tensor(batch["next_state"], dtype=torch.float32, device=self.device)
        state, next_state = data["state"], data["next_state"]
        nominal = data["nominal_action"] if self.graph_mode else data["nominal_action"].flatten(1)
        period = self.config.communication.actor_period_k
        critic_state = augment_critic_state(state, data["held_residual_action"], data["actor_phase"], period, self.graph_mode)
        next_critic_state = augment_critic_state(next_state, data["next_held_residual_action"], data["next_actor_phase"], period, self.graph_mode)
        with torch.no_grad():
            next_policy_action, _ = self._joint(self.actor_target, next_state)
            if self.graph_mode:
                scales = next_state["node_features"][..., NODE_FEATURES.acceleration_limit]
            else:
                scales = next_state[..., NODE_FEATURES.acceleration_limit]
            scale_flat = scales if self.graph_mode else scales.expand(-1, -1, 3).reshape(scales.shape[0], -1)
            noise = torch.maximum(torch.minimum(0.2 * torch.randn_like(next_policy_action) * scale_flat, 0.5 * scale_flat), -0.5 * scale_flat)
            next_policy_action = torch.maximum(torch.minimum(next_policy_action + noise, scale_flat), -scale_flat)
            if self.graph_mode:
                next_policy_action = next_policy_action * next_state["node_mask"].unsqueeze(-1)
            next_hold_action = held_nominal_action(next_state, data["next_held_residual_action"], self.config, self.graph_mode)
            next_action = select_decision_action(next_policy_action, next_hold_action, data["next_is_actor_decision_tick"])
            next_reward = torch.minimum(*self._reward_values(self.reward_targets, next_critic_state, next_action))
            reward_target = data["reward"] + self.config.learning.gamma_reward * data["bootstrap_mask"] * next_reward
            next_cost_members = self._cost_values(self.cost_targets, next_critic_state, next_action)
            cost_target = data["cost"].unsqueeze(0) + self.config.learning.gamma_cost * data["bootstrap_mask"].unsqueeze(0) * next_cost_members
        reward_loss = sum(F.mse_loss(value, reward_target) for value in self._reward_values(self.reward_critics, critic_state, nominal))
        cost_loss = F.mse_loss(self._cost_values(self.cost_critics, critic_state, nominal), cost_target)
        critic_loss = reward_loss + cost_loss
        self.critic_optimizer.zero_grad(); critic_loss.backward(); self.critic_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)
        metrics = {"reward_critic_loss": float(reward_loss.detach()), "cost_critic_loss": float(cost_loss.detach())}
        with torch.no_grad():
            replay_members = self._cost_values(self.cost_critics, critic_state, nominal)
            replay_ucb = self.cal.ucb(replay_members)
            cost_mean = replay_members.mean(dim=0)
            cost_std = replay_members.std(dim=0, unbiased=False)
            batch_above = replay_ucb > self.config.learning.d_cost
            self.cost_ucb_sample_count += int(batch_above.numel())
            self.cost_ucb_above_limit_count += int(batch_above.sum())
        episodic_members = episodic_ucb = None
        episodic_variant = self.config.learning.cal_variant == "episodic_dual_replay_gradient"
        constraint_available = not episodic_variant or initial_state_batch is not None
        if episodic_variant and initial_state_batch is not None:
            episodic_members, episodic_ucb = self._initial_constraint_values(initial_state_batch)
            constraint_estimate = float(episodic_ucb.mean())
            constraint_violation = constraint_estimate - self.config.learning.d_cost
            dual_signal_type = "start_state_ucb"
        elif episodic_variant:
            constraint_estimate = float(self.config.learning.d_cost)
            constraint_violation = 0.0
            dual_signal_type = "start_state_ucb"
        else:
            constraint_estimate = float(replay_ucb.mean())
            constraint_violation = constraint_estimate - self.config.learning.d_cost
            dual_signal_type = "replay_state_ucb"
        effective_multiplier = self.cal.effective_multiplier(self.dual.value, constraint_violation)
        decision = data["is_actor_decision_tick"]
        if self.update_count % 2 == 0 and int(decision.sum().item()):
            with frozen_parameters(self.reward_critics, self.cost_critics):
                policy_action, _ = self._joint(self.actor, state)
                reward_value = self._reward_values(self.reward_critics[:1], critic_state, policy_action)[0]
                conservative_cost = self.cal.ucb(self._cost_values(self.cost_critics, critic_state, policy_action))
                if self.config.learning.cal_variant == "episodic_dual_replay_gradient":
                    penalty = (
                        self.cal.actor_penalty(
                            conservative_cost,
                            replay_ucb,
                            self.dual.value,
                            constraint_violation=constraint_violation,
                        )
                        if episodic_ucb is not None
                        else torch.zeros_like(conservative_cost)
                    )
                else:
                    penalty = self.cal.actor_penalty(conservative_cost, replay_ucb, self.dual.value)
                actor_loss = ((-reward_value + penalty) * decision).sum() / decision.sum()
                self.actor_optimizer.zero_grad(); actor_loss.backward(); self.actor_optimizer.step()
            self._soft_update()
            metrics.update(
                actor_loss=float(actor_loss.detach()),
                policy_ucb_cost_estimate=float(conservative_cost.mean().detach()),
                policy_action_cost_ucb=float(conservative_cost.mean().detach()),
                actor_decision_fraction_batch=float(decision.mean()),
            )
        lambda_before = self.dual.value
        dual_update_applied = bool(
            allow_dual_update
            and self.update_count % self.config.training.dual_update_period == 0
            and (self.config.learning.cal_variant == "replay_cal_legacy" or episodic_ucb is not None)
        )
        if dual_update_applied:
            self.dual.update(constraint_violation)
            self.dual_update_count += 1
            self.lambda_positive_update_count += int(self.dual.value > 0.0)
        lambda_after = self.dual.value
        metrics.update({
            "ucb_cost_estimate": float(replay_ucb.mean()),
            "cost_critic_mean": float(cost_mean.mean()),
            "cost_critic_std": float(cost_std.mean()),
            "cost_ucb": float(replay_ucb.mean()),
            "cost_ucb_batch_mean": float(replay_ucb.mean()),
            "fraction_cost_ucb_above_d_cost_batch": float(batch_above.float().mean()),
            "fraction_cost_ucb_above_d_cost": self.cost_ucb_above_limit_count / max(1, self.cost_ucb_sample_count),
            "lambda": lambda_after,
            "lambda_before_update": lambda_before,
            "lambda_after_update": lambda_after,
            "lambda_update_magnitude": abs(lambda_after - lambda_before),
            "fraction_lambda_positive": self.lambda_positive_update_count / max(1, self.dual_update_count),
            "dual_constraint_estimate": constraint_estimate,
            "dual_constraint_violation": constraint_violation,
            "effective_cal_multiplier": effective_multiplier,
            "dual_update_applied": float(dual_update_applied),
            "dual_update_count": self.dual_update_count,
            "dual_signal_start_state": float(dual_signal_type == "start_state_ucb"),
            "dual_constraint_available": float(constraint_available),
            "replay_cost_ucb_p50": float(torch.quantile(replay_ucb, 0.50)),
            "replay_cost_ucb_p75": float(torch.quantile(replay_ucb, 0.75)),
            "replay_cost_ucb_p90": float(torch.quantile(replay_ucb, 0.90)),
            "replay_cost_ucb_p95": float(torch.quantile(replay_ucb, 0.95)),
            "replay_cost_ucb_max": float(replay_ucb.max()),
        })
        if episodic_ucb is not None and episodic_members is not None:
            metrics.update({
                "episodic_constraint_estimate": float(episodic_ucb.mean()),
                "start_cost_critic_mean": float(episodic_members.mean(dim=0).mean()),
                "start_cost_critic_std": float(episodic_members.std(dim=0, unbiased=False).mean()),
                "start_cost_ucb_p50": float(torch.quantile(episodic_ucb, 0.50)),
                "start_cost_ucb_p75": float(torch.quantile(episodic_ucb, 0.75)),
                "start_cost_ucb_p90": float(torch.quantile(episodic_ucb, 0.90)),
                "start_cost_ucb_p95": float(torch.quantile(episodic_ucb, 0.95)),
                "start_cost_ucb_max": float(episodic_ucb.max()),
            })
            metrics.update({
                f"dual_initial_batch_N{int(count)}": int(value)
                for count, value in initial_state_batch.get("sampled_count_by_n", {}).items()
            })
        return metrics

    def estimate_start_cost_ucb(self, observation) -> float:
        return self.estimate_start_cost_statistics(observation)["ucb"]

    def sample_start_action_cost_statistics(self, observation, *, deterministic: bool = True) -> tuple[np.ndarray, dict[str, float | list[float]]]:
        execution_action = self.act(observation, deterministic=deterministic)
        return execution_action, self.start_cost_statistics_for_action(observation, execution_action)

    def start_cost_statistics_for_action(self, observation, execution_action: np.ndarray) -> dict[str, float | list[float]]:
        with torch.no_grad():
            state = graph_to_torch(observation, self.device) if self.graph_mode else torch.as_tensor(observation[None], dtype=torch.float32, device=self.device)
            normalized = torch.as_tensor(execution_action[None], dtype=torch.float32, device=self.device)
            if self.graph_mode:
                nominal = compose_action_torch(
                    normalized, state["node_features"], self.config.control.actor_mode,
                    self.config.control.residual_acceleration_fraction,
                    self.config.control.reference_speed_fraction,
                    self.config.control.reference_velocity_gain,
                ) * state["node_mask"].unsqueeze(-1)
            else:
                mask = state[..., NODE_FEATURES.active_mask].clamp(0.0, 1.0)
                nominal = compose_action_torch(
                    normalized, state, self.config.control.actor_mode,
                    self.config.control.residual_acceleration_fraction,
                    self.config.control.reference_speed_fraction,
                    self.config.control.reference_velocity_gain,
                ) * mask
                nominal = nominal.flatten(1)
            agents = state["node_features"].shape[1] if self.graph_mode else state.shape[1]
            critic_state = augment_critic_state(state, torch.zeros((1, agents, 3), device=self.device), torch.zeros((1, 1), device=self.device), self.config.communication.actor_period_k, self.graph_mode)
            members = self._cost_values(self.cost_critics, critic_state, nominal).reshape(-1)
            return {
                "members": [float(value) for value in members],
                "mean": float(members.mean()),
                "std": float(members.std(unbiased=False)),
                "ucb": float(members.mean() + self.config.learning.beta_ucb * members.std(unbiased=False)),
                "conditioned_execution_action": np.asarray(execution_action).tolist(),
                "conditioned_nominal_action": nominal.reshape(-1, 3).cpu().numpy().tolist(),
            }

    def estimate_start_cost_statistics(self, observation, *, deterministic: bool = True) -> dict[str, float | list[float]]:
        with torch.no_grad():
            state = graph_to_torch(observation, self.device) if self.graph_mode else torch.as_tensor(observation[None], dtype=torch.float32, device=self.device)
            action, _ = self._joint(self.actor, state)
            agents = state["node_features"].shape[1] if self.graph_mode else state.shape[1]
            critic_state = augment_critic_state(state, torch.zeros((1, agents, 3), device=self.device), torch.zeros((1, 1), device=self.device), self.config.communication.actor_period_k, self.graph_mode)
            members = self._cost_values(self.cost_critics, critic_state, action).reshape(-1)
            return {
                "members": [float(value) for value in members],
                "mean": float(members.mean()),
                "std": float(members.std(unbiased=False)),
                "ucb": float(members.mean() + self.config.learning.beta_ucb * members.std(unbiased=False)),
            }

    def estimate_action_cost_ucb(self, observation, nominal_action: np.ndarray) -> float:
        with torch.no_grad():
            state = graph_to_torch(observation, self.device) if self.graph_mode else torch.as_tensor(observation[None], dtype=torch.float32, device=self.device)
            action = torch.as_tensor(nominal_action[None], dtype=torch.float32, device=self.device)
            if not self.graph_mode:
                action = action.flatten(1)
            agents = state["node_features"].shape[1] if self.graph_mode else state.shape[1]
            critic_state = augment_critic_state(state, torch.zeros((1, agents, 3), device=self.device), torch.zeros((1, 1), device=self.device), self.config.communication.actor_period_k, self.graph_mode)
            return float(self.cal.ucb(self._cost_values(self.cost_critics, critic_state, action)).mean())

    def _soft_update(self) -> None:
        with torch.no_grad():
            for target_module, source_module in ((self.actor_target, self.actor), (self.reward_targets, self.reward_critics), (self.cost_targets, self.cost_critics)):
                for target, source in zip(target_module.parameters(), source_module.parameters(), strict=True):
                    target.lerp_(source, self.tau)

    def state_dict(self) -> dict:
        return {"actor": self.actor.state_dict(), "actor_target": self.actor_target.state_dict(), "reward_critics": self.reward_critics.state_dict(), "cost_critics": self.cost_critics.state_dict(), "reward_targets": self.reward_targets.state_dict(), "cost_targets": self.cost_targets.state_dict(), "actor_optimizer": self.actor_optimizer.state_dict(), "critic_optimizer": self.critic_optimizer.state_dict(), "dual": self.dual.value, "update_count": self.update_count, "diagnostic_counts": {"cost_ucb_sample_count": self.cost_ucb_sample_count, "cost_ucb_above_limit_count": self.cost_ucb_above_limit_count, "dual_update_count": self.dual_update_count, "lambda_positive_update_count": self.lambda_positive_update_count}}

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"]); self.actor_target.load_state_dict(state["actor_target"])
        self.reward_critics.load_state_dict(state["reward_critics"]); self.cost_critics.load_state_dict(state["cost_critics"])
        self.reward_targets.load_state_dict(state["reward_targets"]); self.cost_targets.load_state_dict(state["cost_targets"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"]); self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.dual.value = float(state["dual"]); self.update_count = int(state["update_count"])
        counts = state.get("diagnostic_counts", {})
        self.cost_ucb_sample_count = int(counts.get("cost_ucb_sample_count", 0))
        self.cost_ucb_above_limit_count = int(counts.get("cost_ucb_above_limit_count", 0))
        self.dual_update_count = int(counts.get("dual_update_count", 0))
        self.lambda_positive_update_count = int(counts.get("lambda_positive_update_count", 0))
