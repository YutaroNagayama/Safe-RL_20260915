import numpy as np
import pytest

from uav_safe_marl import build_env, load_config
from uav_safe_marl.runners.trainer import SafeRLTrainer


def mixed_scenario(identifier, count):
    positions = [[30.0 * i, 0.0, 0.0] for i in range(count)]
    return {
        "scenario_id": identifier,
        "actual_agent_count": count,
        "positions": positions,
        "velocities": [[1.0, 0.0, 0.0] for _ in range(count)],
        "goals": [[point[0] + 100.0, 0.0, 0.0] for point in positions],
        "reference_route_lengths": [100.0] * count,
        "maximum_accelerations": [1.0 + i for i in range(count)],
        "maximum_velocities": [8.0 + i for i in range(count)],
        "predicted_los_edges": [],
    }


def test_three_agent_environment_runs_with_variable_neighbour_count():
    config = load_config(overrides={"environment": {"agent_count": 3, "max_steps": 3}})
    env = build_env(config)
    observation, _ = env.reset(seed=3, options={"radius": 12.0})
    counts = []
    for _ in range(3):
        observation, _, terminated, truncated, info = env.step(np.zeros((3, 3), dtype=np.float32))
        counts.append(sum(len(item["active_neighbours"]) for item in info["safety"]))
        if terminated or truncated:
            break
    assert observation.shape == (3, 14)
    assert max(counts) > 0


def test_yaml_ablation_changes_only_requested_section():
    config = load_config("configs/cbf_off.yaml")
    assert not config.safety.enabled
    assert config.communication.unimp_enabled


def test_training_collection_keeps_exploration_when_execution_uses_mean(tmp_path):
    config = load_config(overrides={
        "policy": {"execution_mode": "mean"},
        "environment": {"max_steps": 1},
        "learning": {"batch_size": 8},
    })
    trainer = SafeRLTrainer(config, build_env(config), output_dir=tmp_path)
    deterministic_flags = []

    def record_action(observation, deterministic):
        deterministic_flags.append(deterministic)
        return np.zeros((config.environment.agent_count, 3), dtype=np.float32)

    trainer.backend.act = record_action
    trainer.train(episodes=1)
    assert deterministic_flags == [False]


def test_training_summary_exposes_cmdp_cost_scales(tmp_path):
    config = load_config(overrides={
        "environment": {"max_steps": 1},
        "learning": {"batch_size": 8, "d_cost": 7.5},
    })
    summary = SafeRLTrainer(config, build_env(config), output_dir=tmp_path).train(episodes=1)[0]
    required = {
        "undiscounted_episode_cost",
        "discounted_episode_cost",
        "start_state_cost_ucb",
        "replay_state_cost_ucb",
        "configured_d_cost",
    }
    assert required <= summary.keys()
    assert summary["configured_d_cost"] == 7.5
    assert summary["cal_mode"] == "pdf_variant"
    assert summary["cal_variant"] == "episodic_dual_replay_gradient"
    assert summary["cumulative_environment_steps"] == 1
    assert summary["gradient_updates"] == 0
    assert summary["replay_size"] == 1


def test_training_progress_is_cumulative_across_calls(tmp_path):
    config = load_config(overrides={"environment": {"max_steps": 1}, "learning": {"batch_size": 8}})
    trainer = SafeRLTrainer(config, build_env(config), output_dir=tmp_path)
    first = trainer.train(episodes=1, live_display=False)[0]
    second = trainer.train(episodes=1, live_display=False)[0]
    assert [first["episode"], second["episode"]] == [0.0, 1.0]
    assert second["cumulative_environment_steps"] == 2
    assert second["replay_size"] == 2


def test_trainer_accepts_external_preflight_los_candidates(tmp_path):
    config = load_config(overrides={
        "environment": {"agent_count": 3, "max_steps": 1},
        "communication": {"delay_steps": 0},
        "learning": {"batch_size": 8},
    })
    trainer = SafeRLTrainer(config, build_env(config), output_dir=tmp_path)
    trainer.train(episodes=1, reset_options={"predicted_los_edges": [(0, 1)]})
    assert trainer.communication.predicted_los_edges == {(0, 1)}
    assert trainer.communication.message_count == 2


def test_environment_pads_actual_agents_and_applies_heterogeneous_action_scales():
    config = load_config(overrides={"environment": {"agent_count": 2, "max_agent_count": 4, "max_steps": 1}, "control": {"actor_mode": "full"}})
    env = build_env(config)
    observation, info = env.reset(options={"scenario": mixed_scenario("heterogeneous", 3)})
    assert observation.shape == (4, 14)
    np.testing.assert_array_equal(info["actual_agent_mask"], [True, True, True, False])
    np.testing.assert_allclose(observation[:3, 9], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(observation[:3, 10], [8.0, 9.0, 10.0])
    np.testing.assert_array_equal(observation[3], np.zeros(14))
    _, _, _, _, step = env.step(np.ones((4, 3), dtype=np.float32))
    np.testing.assert_allclose(step["nominal_action"][:3], [[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    np.testing.assert_array_equal(step["nominal_action"][3], np.zeros(3))


@pytest.mark.parametrize("aggregation", [("graph", "graph"), ("pooled", "concat")])
def test_one_trainer_collects_mixed_n_for_graph_and_baseline(tmp_path, aggregation):
    actor_mode, critic_mode = aggregation
    config = load_config(overrides={
        "graph": {"actor_aggregation_backend": actor_mode, "critic_aggregation_backend": critic_mode},
        "environment": {"agent_count": 2, "max_agent_count": 4, "max_steps": 1},
        "learning": {"batch_size": 2},
    })
    trainer = SafeRLTrainer(config, build_env(config), output_dir=tmp_path / actor_mode)
    history = trainer.train(episodes=2, scenarios=[mixed_scenario("n2", 2), mixed_scenario("n3", 3)])
    assert [row["actual_agent_count"] for row in history] == [2, 3]
    assert len(trainer.buffer) == 2
    sample = trainer.buffer.sample(2)
    if actor_mode == "graph":
        assert sorted(sample["state"]["node_mask"].sum(axis=1).tolist()) == [2, 3]
    else:
        assert sorted(sample["state"][:, :, 14].sum(axis=1).tolist()) == [2, 3]
