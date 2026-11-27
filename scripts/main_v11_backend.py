"""v1.1-only SAC update controls; the v1.0 engine and control law are untouched.

The Bellman targets, CAL expression, optimizer schedules and multi-rate handling
are copied from the frozen parent SAC update. Only explicit update masks and a
permanently frozen dual variable differ. Parity/invariance tests cover this copy.
"""
import numpy as np

from uav_safe_marl.algorithms.sac_backend import (
    SACBackend, torch, F, graph_to_torch, augment_critic_state,
    held_nominal_action, select_decision_action, frozen_parameters, NODE_FEATURES,
)


class V11Backend(SACBackend):
    def update_partition(
        self,
        batch: dict[str, np.ndarray],
        *,
        cost_batch: dict[str, np.ndarray] | None = None,
        initial_state_batch: dict | None = None,
        train_actor: bool = True,
        train_cost: bool = True,
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
            next_policy_action, next_logp, _ = self._policy_joint(next_state)
            next_held_action = held_nominal_action(next_state, data["next_held_residual_action"], self.config, self.graph_mode)
            next_action = select_decision_action(next_policy_action, next_held_action, data["next_is_actor_decision_tick"])
            next_logp = next_logp * data["next_is_actor_decision_tick"]
            next_reward = torch.minimum(*self._reward_values(self.reward_targets, next_critic_state, next_action)) - self.alpha.detach() * next_logp
            reward_target = data["reward"] + self.config.learning.gamma_reward * data["bootstrap_mask"] * next_reward
            # With n-step Cost data this target is unused. Cost encoders have
            # no stochastic/stateful layers, so omitting it preserves RNG.
            if train_cost and cost_batch is None:
                next_cost_members = self._cost_values(self.cost_targets, next_critic_state, next_action)
                cost_target = data["cost"].unsqueeze(0) + self.config.learning.gamma_cost * data["bootstrap_mask"].unsqueeze(0) * next_cost_members
        reward_loss = sum(F.mse_loss(value, reward_target) for value in self._reward_values(self.reward_critics, critic_state, nominal))
        cost_horizon = 1.0
        cost_loss = torch.zeros((), device=self.device)
        if train_cost:
            if cost_batch is None:
                cost_loss = F.mse_loss(self._cost_values(self.cost_critics, critic_state, nominal), cost_target)
            else:
                excluded = {"state", "next_state", "n_step_next_state"}
                cost_data = {
                    key: torch.as_tensor(value, dtype=torch.float32, device=self.device)
                    for key, value in cost_batch.items()
                    if key not in excluded
                }
                cost_state = graph_to_torch(cost_batch["state"], self.device) if self.graph_mode else torch.as_tensor(cost_batch["state"], dtype=torch.float32, device=self.device)
                cost_next_state = graph_to_torch(cost_batch["n_step_next_state"], self.device) if self.graph_mode else torch.as_tensor(cost_batch["n_step_next_state"], dtype=torch.float32, device=self.device)
                cost_nominal = cost_data["nominal_action"] if self.graph_mode else cost_data["nominal_action"].flatten(1)
                cost_critic_state = augment_critic_state(cost_state, cost_data["held_residual_action"], cost_data["actor_phase"], period, self.graph_mode)
                cost_next_critic_state = augment_critic_state(
                    cost_next_state,
                    cost_data["n_step_next_held_residual_action"],
                    cost_data["n_step_next_actor_phase"],
                    period,
                    self.graph_mode,
                )
                with torch.no_grad():
                    cost_next_policy_action, _, _ = self._policy_joint(cost_next_state)
                    cost_next_held_action = held_nominal_action(
                        cost_next_state,
                        cost_data["n_step_next_held_residual_action"],
                        self.config,
                        self.graph_mode,
                    )
                    cost_next_action = select_decision_action(
                        cost_next_policy_action,
                        cost_next_held_action,
                        cost_data["n_step_next_is_actor_decision_tick"],
                    )
                    cost_next_members = self._cost_values(self.cost_targets, cost_next_critic_state, cost_next_action)
                    cost_target = (
                        cost_data["n_step_cost"].unsqueeze(0)
                        + cost_data["n_step_discount"].unsqueeze(0)
                        * cost_data["n_step_bootstrap_mask"].unsqueeze(0)
                        * cost_next_members
                    )
                cost_loss = F.mse_loss(
                    self._cost_values(self.cost_critics, cost_critic_state, cost_nominal),
                    cost_target,
                )
                cost_horizon = float(cost_data["n_step_horizon"].float().mean())
        critic_loss = reward_loss + cost_loss
        self.critic_optimizer.zero_grad(); critic_loss.backward(); self.critic_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)

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
            # Until the initial-state buffer represents the configured mu, keep
            # the episodic constraint path neutral instead of silently falling
            # back to a replay-state dual signal.
            constraint_estimate = float(self.config.learning.d_cost)
            constraint_violation = 0.0
            dual_signal_type = "start_state_ucb"
        else:
            constraint_estimate = float(replay_ucb.mean())
            constraint_violation = constraint_estimate - self.config.learning.d_cost
            dual_signal_type = "replay_state_ucb"
        effective_multiplier = self.cal.effective_multiplier(self.dual.value, constraint_violation)
        decision = data["is_actor_decision_tick"]
        decision_count = int(decision.sum().item())
        actor_update_applied = bool(
            train_actor and decision_count
            and self.update_count % self.config.training.actor_update_period == 0
        )
        actor_loss = torch.zeros((), device=self.device)
        conservative_cost = replay_ucb.detach()
        logp = torch.zeros_like(data["reward"])
        if actor_update_applied:
            with frozen_parameters(self.reward_critics, self.cost_critics):
                policy_action, logp, _ = self._policy_joint(state)
                reward_value = torch.minimum(*self._reward_values(self.reward_critics, critic_state, policy_action))
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
                if not train_cost:
                    penalty = torch.zeros_like(conservative_cost)
                per_row = self.alpha.detach() * logp - reward_value + penalty
                actor_loss = (per_row * decision).sum() / decision.sum()
                self.actor_optimizer.zero_grad(); actor_loss.backward(); self.actor_optimizer.step()
        alpha_loss_value = 0.0
        alpha_before = float(self.alpha.detach())
        target_entropy = -3.0 * (
            state["node_mask"].sum(dim=1, keepdim=True)
            if self.graph_mode else state[..., NODE_FEATURES.active_mask.start].sum(dim=1, keepdim=True)
        )
        if self.alpha_optimizer is not None and actor_update_applied:
            alpha_loss = -((self.log_alpha * (logp + target_entropy).detach()) * decision).sum() / decision.sum()
            self.alpha_optimizer.zero_grad(); alpha_loss.backward(); self.alpha_optimizer.step()
            alpha_loss_value = float(alpha_loss.detach())
        alpha_after = float(self.alpha.detach())
        lambda_before = self.dual.value
        dual_update_applied = bool(
            False  # v1.1 always freezes the parent lambda
            and self.config.training.dual_update_mode == "gradient_step"
            and self.update_count % self.config.training.dual_update_period == 0
            and (self.config.learning.cal_variant == "replay_cal_legacy" or episodic_ucb is not None)
        )
        if dual_update_applied:
            self.dual.update(constraint_violation)
            self.dual_update_count += 1
            self.lambda_positive_update_count += int(self.dual.value > 0.0)
        lambda_after = self.dual.value
        with torch.no_grad():
            for target, source in zip(self.reward_targets.parameters(), self.reward_critics.parameters(), strict=True):
                target.lerp_(source, self.tau)
            if train_cost:
                for target, source in zip(self.cost_targets.parameters(), self.cost_critics.parameters(), strict=True):
                    target.lerp_(source, self.tau)
        metrics = {
            "actor_loss": float(actor_loss.detach()),
            "reward_critic_loss": float(reward_loss.detach()),
            "cost_critic_loss": float(cost_loss.detach()),
            "cost_target_mean_horizon": cost_horizon,
            "ucb_cost_estimate": float(replay_ucb.mean()),
            "policy_ucb_cost_estimate": float(conservative_cost.mean().detach()),
            "cost_critic_mean": float(cost_mean.mean()),
            "cost_critic_std": float(cost_std.mean()),
            "cost_ucb": float(replay_ucb.mean()),
            "cost_ucb_batch_mean": float(replay_ucb.mean()),
            "fraction_cost_ucb_above_d_cost_batch": float(batch_above.float().mean()),
            "fraction_cost_ucb_above_d_cost": self.cost_ucb_above_limit_count / max(1, self.cost_ucb_sample_count),
            "policy_action_cost_ucb": float(conservative_cost.mean().detach()),
            "entropy_alpha": alpha_after,
            "log_alpha": float(self.log_alpha.detach()),
            "alpha_loss": alpha_loss_value,
            "alpha_update_magnitude": abs(alpha_after - alpha_before),
            "effective_target_entropy": float(target_entropy.float().mean()),
            "mean_policy_log_prob": float(logp.detach().mean()),
            "actor_decision_fraction_batch": float(decision.mean()),
            "actor_update_applied": float(actor_update_applied),
            "actor_update_period": float(self.config.training.actor_update_period),
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
        }
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
        metrics.update(v11_actor_enabled=float(train_actor), v11_cost_enabled=float(train_cost),
                       v11_applied_cal_multiplier=effective_multiplier if train_cost and train_actor else 0.)
        return metrics
