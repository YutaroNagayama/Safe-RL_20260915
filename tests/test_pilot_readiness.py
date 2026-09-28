from __future__ import annotations

import numpy as np

from uav_safe_marl import build_env, load_config
from uav_safe_marl.core.types import AgentState, Transition
from uav_safe_marl.replay.buffer import NumpyReplayBuffer
from uav_safe_marl.runners.trainer import SafeRLTrainer


def _state(speed: float = 0.0) -> AgentState:
    return AgentState(
        position=np.zeros(3),
        velocity=np.array([speed, 0.0, 0.0]),
        goal=np.array([100.0, 0.0, 0.0]),
        acceleration_limit=np.full(3, 3.0),
        velocity_limit=np.full(3, 10.0),
    )


def test_environment_step_budget_finishes_the_current_episode(tmp_path):
    config = load_config(overrides={
        "environment": {"agent_count": 2, "max_steps": 2},
        "learning": {"batch_size": 100},
        "training": {"total_environment_steps_budget": 3},
        "monitoring": {"enabled": False},
    })
    trainer = SafeRLTrainer(config, build_env(config), tmp_path)
    history = trainer.train(episodes=None)
    assert trainer.total_environment_steps == 4
    assert trainer.completed_episodes == 2
    assert history[-1]["total_environment_steps_budget"] == 3
    assert history[-1]["environment_steps_budget_overshoot"] == 1


def test_learning_starts_and_update_ratio_are_independent_of_batch_size(tmp_path):
    config = load_config(overrides={
        "environment": {"agent_count": 2, "max_steps": 3},
        "learning": {"batch_size": 2, "buffer_size": 20},
        "training": {"learning_starts": 3, "updates_per_environment_step": 2},
        "monitoring": {"enabled": False},
    })
    trainer = SafeRLTrainer(config, build_env(config), tmp_path)
    calls = []
    trainer.backend.update = lambda batch, **kwargs: calls.append((batch, kwargs)) or {}
    trainer.train(episodes=1)
    assert trainer.total_environment_steps == 3
    assert trainer.total_gradient_updates == 2
    assert len(calls) == 2


def test_online_sac_uses_configured_n_step_only_for_cost_target(tmp_path):
    config = load_config(overrides={
        "environment": {"agent_count": 2, "max_steps": 12},
        "graph": {"actor_aggregation_backend": "pooled", "critic_aggregation_backend": "concat"},
        "learning": {"batch_size": 2, "buffer_size": 50, "cost_target_n_step": 3},
        "training": {"learning_starts": 10, "dual_learning_starts": 10},
        "monitoring": {"enabled": False},
    })
    trainer = SafeRLTrainer(config, build_env(config), tmp_path)
    calls = []

    def record(batch, **kwargs):
        calls.append((batch, kwargs))
        return {}

    trainer.backend.update = record
    trainer.train(episodes=1)
    assert calls
    ordinary_batch, kwargs = calls[0]
    assert "n_step_cost" not in ordinary_batch
    assert kwargs["cost_batch"]["n_step_horizon"].max() <= 3
    assert kwargs["cost_batch"]["n_step_horizon"].min() >= 1


def test_episode_dual_updates_once_per_completed_episode(tmp_path):
    config = load_config(overrides={
        "environment": {"agent_count": 2, "max_steps": 3},
        "graph": {"actor_aggregation_backend": "pooled", "critic_aggregation_backend": "concat"},
        "learning": {"batch_size": 1, "buffer_size": 20},
        "training": {
            "learning_starts": 1,
            "dual_learning_starts": 1,
            "dual_update_mode": "episode",
            "dual_update_interval_episodes": 1,
        },
        "monitoring": {"enabled": False},
    })
    trainer = SafeRLTrainer(config, build_env(config), tmp_path)
    gradient_allow_flags = []
    episode_batches = []
    trainer.backend.update = lambda batch, **kwargs: gradient_allow_flags.append(kwargs["allow_dual_update"]) or {}
    trainer.backend.update_dual_from_initial_states = lambda batch: episode_batches.append(batch) or {"dual_update_applied": 1.0}
    trainer.train(episodes=2)
    assert gradient_allow_flags and not any(gradient_allow_flags)
    assert len(episode_batches) == 2


def test_collect_only_populates_fresh_replay_without_any_learning_update(tmp_path):
    config = load_config(overrides={
        "environment": {"agent_count": 2, "max_steps": 2},
        "learning": {"batch_size": 1, "buffer_size": 20},
        "training": {"learning_starts": 1},
        "monitoring": {"enabled": False},
    })
    trainer = SafeRLTrainer(config, build_env(config), tmp_path)
    calls = []
    trainer.backend.update = lambda batch, **kwargs: calls.append((batch, kwargs)) or {}
    trainer.train(episodes=1, update_mode="collect_only")
    assert len(trainer.buffer) == 2
    assert trainer.total_gradient_updates == 0
    assert calls == []


def test_sac_cost_only_update_freezes_every_non_cost_parameter(tmp_path):
    import torch

    config = load_config(overrides={
        "environment": {"agent_count": 2, "max_steps": 2},
        "graph": {"actor_aggregation_backend": "pooled", "critic_aggregation_backend": "concat"},
        "learning": {"batch_size": 2, "buffer_size": 20},
        "monitoring": {"enabled": False},
    })
    trainer = SafeRLTrainer(config, build_env(config), tmp_path)
    trainer.train(episodes=1, update_mode="collect_only")

    def snapshot(module):
        return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}

    actor_before = snapshot(trainer.backend.actor)
    reward_before = snapshot(trainer.backend.reward_critics)
    reward_target_before = snapshot(trainer.backend.reward_targets)
    cost_before = snapshot(trainer.backend.cost_critics)
    alpha_before = trainer.backend.log_alpha.detach().cpu().clone()
    lambda_before = trainer.backend.dual.value
    metrics = trainer.backend.update_cost_critic_only(trainer.buffer.sample(2))

    assert all(torch.equal(value, trainer.backend.actor.state_dict()[name].cpu()) for name, value in actor_before.items())
    assert all(torch.equal(value, trainer.backend.reward_critics.state_dict()[name].cpu()) for name, value in reward_before.items())
    assert all(torch.equal(value, trainer.backend.reward_targets.state_dict()[name].cpu()) for name, value in reward_target_before.items())
    assert any(not torch.equal(value, trainer.backend.cost_critics.state_dict()[name].cpu()) for name, value in cost_before.items())
    assert torch.equal(alpha_before, trainer.backend.log_alpha.detach().cpu())
    assert trainer.backend.dual.value == lambda_before
    assert "cost_critic_loss" in metrics

    cost_before_n_step = snapshot(trainer.backend.cost_critics)
    metrics_n_step = trainer.backend.update_cost_critic_only_n_step(
        trainer.buffer.sample_n_step(2, n_step=2, gamma=config.learning.gamma_cost)
    )
    assert all(torch.equal(value, trainer.backend.actor.state_dict()[name].cpu()) for name, value in actor_before.items())
    assert all(torch.equal(value, trainer.backend.reward_critics.state_dict()[name].cpu()) for name, value in reward_before.items())
    assert any(not torch.equal(value, trainer.backend.cost_critics.state_dict()[name].cpu()) for name, value in cost_before_n_step.items())
    assert torch.equal(alpha_before, trainer.backend.log_alpha.detach().cpu())
    assert trainer.backend.dual.value == lambda_before
    assert 1.0 <= metrics_n_step["n_step_mean_horizon"] <= 2.0


def test_sac_actor_update_period_delays_actor_and_alpha_only(tmp_path):
    import torch

    config = load_config(overrides={
        "environment": {"agent_count": 2, "max_steps": 3},
        "communication": {"actor_period_k": 1},
        "graph": {"actor_aggregation_backend": "pooled", "critic_aggregation_backend": "concat"},
        "learning": {"batch_size": 2, "buffer_size": 20},
        "training": {"learning_starts": 10, "actor_update_period": 2},
        "monitoring": {"enabled": False},
    })
    trainer = SafeRLTrainer(config, build_env(config), tmp_path)
    trainer.train(episodes=1, update_mode="collect_only")

    def snapshot(module):
        return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}

    batch = trainer.buffer.sample(2)
    actor_initial = snapshot(trainer.backend.actor)
    alpha_initial = trainer.backend.log_alpha.detach().cpu().clone()
    first = trainer.backend.update(batch, allow_dual_update=False)
    assert first["actor_update_applied"] == 0.0
    assert all(torch.equal(value, trainer.backend.actor.state_dict()[name].cpu()) for name, value in actor_initial.items())
    assert torch.equal(alpha_initial, trainer.backend.log_alpha.detach().cpu())

    second = trainer.backend.update(batch, allow_dual_update=False)
    assert second["actor_update_applied"] == 1.0
    assert any(not torch.equal(value, trainer.backend.actor.state_dict()[name].cpu()) for name, value in actor_initial.items())
    assert not torch.equal(alpha_initial, trainer.backend.log_alpha.detach().cpu())


def test_hocbf_off_applies_physical_projection_but_keeps_counterfactual_action(tmp_path):
    config = load_config(overrides={
        "environment": {"agent_count": 1, "max_steps": 1},
        "safety": {"enabled": False},
        "control": {"actor_mode": "full"},
        "learning": {"batch_size": 8},
        "monitoring": {"enabled": False},
    })
    env = build_env(config)
    env.reset()
    env.states[0] = _state(speed=9.9)
    _, _, _, _, info = env.step(np.array([[1.0, 1.0, 0.0]]))
    assert not np.allclose(info["safe_action"], info["nominal_action"])
    assert np.linalg.norm(env.states[0].velocity) <= 10.0 + 1e-8
    # There is no pairwise constraint for one agent, so pure HOCBF demand is zero.
    np.testing.assert_allclose(info["intervention_action"], info["nominal_action"])
    assert info["pure_intervention_squared_sum"] == 0.0
    assert info["runtime_correction_squared_sum"] > 0.0


def test_hocbf_off_does_not_apply_pair_constraint_but_retains_its_cost():
    config = load_config(overrides={
        "environment": {"agent_count": 2, "max_steps": 1},
        "safety": {"enabled": False},
        "monitoring": {"enabled": False},
    })
    env = build_env(config)
    env.reset()
    env.states = [
        AgentState(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([100.0, 0.0, 0.0]), np.full(3, 3.0), np.full(3, 10.0)),
        AgentState(np.array([11.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), np.array([-100.0, 0.0, 0.0]), np.full(3, 3.0), np.full(3, 10.0)),
    ]
    _, _, _, _, info = env.step(np.zeros((2, 3)))
    np.testing.assert_allclose(info["safe_action"], info["nominal_action"])
    assert not np.allclose(info["intervention_action"], info["nominal_action"])
    assert info["pure_intervention_squared_sum"] > 0.0
    assert info["runtime_correction_squared_sum"] == 0.0


def test_replay_reports_current_n_occupancy_after_ring_overwrite():
    buffer = NumpyReplayBuffer(capacity=3, seed=0)
    for count in (2, 4, 8, 16):
        state = np.zeros((16, 2))
        buffer.add(Transition(
            state, np.zeros((16, 3)), np.zeros((16, 3)), 0.0, 0.0,
            state, False, extras={"actual_agent_count": count},
        ))
    assert buffer.occupancy_by_agent_count() == {4: 1, 8: 1, 16: 1}


def test_sac_update_emits_cal_and_entropy_diagnostics(tmp_path):
    config = load_config(overrides={
        "environment": {"agent_count": 2, "max_steps": 2},
        "graph": {"actor_aggregation_backend": "pooled", "critic_aggregation_backend": "concat"},
        "learning": {"batch_size": 2, "buffer_size": 20},
        "training": {"learning_starts": 2},
        "monitoring": {"enabled": False},
    })
    trainer = SafeRLTrainer(config, build_env(config), tmp_path)
    history = trainer.train(episodes=1)
    required = {
        "cost_critic_mean", "cost_critic_std", "cost_ucb", "cost_ucb_batch_mean",
        "fraction_cost_ucb_above_d_cost", "lambda_before_update", "lambda_after_update",
        "lambda_update_magnitude", "fraction_lambda_positive", "entropy_alpha", "log_alpha",
        "alpha_loss", "alpha_update_magnitude", "effective_target_entropy", "mean_policy_log_prob",
        "policy_action_cost_ucb",
        "dual_constraint_estimate", "dual_constraint_violation", "effective_cal_multiplier",
        "episodic_constraint_estimate", "dual_signal_start_state",
    }
    assert required <= history[-1].keys()
    assert history[-1]["configured_d_cost"] == 3.0
    assert history[-1]["effective_target_entropy"] == -6.0
    assert history[-1]["transitions_collected_N2"] == 2
    assert history[-1]["replay_occupancy_N2"] == 2
    assert history[-1]["initial_state_occupancy_N2"] == 1
    assert history[-1]["dual_signal_start_state"] == 1.0
    assert all(parameter.grad is None for parameter in trainer.backend.reward_critics.parameters())
    assert all(parameter.grad is None for parameter in trainer.backend.cost_critics.parameters())
