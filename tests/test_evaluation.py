import json

import numpy as np

from uav_safe_marl import build_env, load_config
from uav_safe_marl.evaluation import (
    aggregate_primary_by_method,
    aggregate_primary_by_method_and_n,
    calibrate_cost_critic,
    evaluate_mean_policy_cost_diagnostic,
    evaluate_scenario_bank,
    generate_random_traffic_bank,
    load_scenario_bank,
    predicted_interaction_data,
    save_evaluation,
    save_scenario_bank,
)
from uav_safe_marl.runners.trainer import SafeRLTrainer
from uav_safe_marl.runners.evaluator import evaluate


def scenario(identifier, offset=0.0):
    return {
        "scenario_id": identifier,
        "positions": [[offset, 0, 0], [20 + offset, 0, 0]],
        "velocities": [[0, 0, 0], [0, 0, 0]],
        "goals": [[1 + offset, 0, 0], [21 + offset, 0, 0]],
        "reference_route_lengths": [1, 1],
        "predicted_los_edges": [[0, 1]],
    }


def test_scenario_bank_roundtrip_preserves_exact_common_conditions(tmp_path):
    path = save_scenario_bank(tmp_path / "bank.json", [scenario("a"), scenario("b", 3.0)])
    loaded = load_scenario_bank(path)
    assert [item["scenario_id"] for item in loaded] == ["a", "b"]
    np.testing.assert_array_equal(loaded[1]["positions"], scenario("b", 3.0)["positions"])


def test_evaluation_saves_recomputable_primary_metrics_and_rollouts(tmp_path):
    config = load_config(overrides={"environment": {"agent_count": 2, "max_steps": 1}, "learning": {"batch_size": 8}, "control": {"actor_mode": "full"}})
    trainer = SafeRLTrainer(config, build_env(config), output_dir=tmp_path / "train")
    trainer.backend.act = lambda observation, deterministic: np.zeros((2, 3), dtype=np.float32)
    result = evaluate_scenario_bank(trainer, [scenario("paired")])
    assert result["primary_metrics"]["episode_success_rate"] == 1.0
    assert result["primary_metrics"]["path_stretch"] == 0.0
    root = save_evaluation(result, tmp_path / "evaluation")
    summary = json.loads((root / "summary.json").read_text())
    assert summary["primary_metrics"] == result["primary_metrics"]
    rollout = np.load(root / "rollouts" / "rollout_0000.npz")
    assert {"positions", "velocities", "active_masks", "reference_actions", "normalized_residual_actions", "residual_actions", "command_actions", "nominal_actions", "safe_actions", "intervention_actions"} <= set(rollout.files)
    assert {"candidate_adjacency", "actual_minimum_pair_distances", "cbf_ever_adjacency", "candidate_miss_adjacency", "candidate_edge_lifetime_ticks"} <= set(rollout.files)
    assert {"reward_absolute_distance_penalty", "mean_normalized_goal_distance", "goal_distances"} <= set(rollout.files)
    episode = result["episode_metrics"][0]
    assert "rollout_pair_results" in episode and "scenario_descriptor" in episode
    assert "absolute_distance_penalty" in episode and "final_goal_distance_mean" in episode


def test_cost_critic_calibration_is_frozen_and_saves_empirical_coverage(tmp_path):
    config = load_config(overrides={
        "environment": {"agent_count": 2, "max_steps": 2},
        "learning": {"batch_size": 8},
        "control": {"actor_mode": "full"},
        "communication": {"actor_period_k": 1},
    })
    trainer = SafeRLTrainer(config, build_env(config), output_dir=tmp_path / "train")
    stochastic_flags = []
    original_act = trainer.backend.act

    def tracked_act(observation, deterministic):
        stochastic_flags.append(not deterministic)
        return original_act(observation, deterministic)

    trainer.backend.act = tracked_act
    before = {
        name: value.detach().cpu().clone()
        for name, value in trainer.backend.state_dict()["cost_critics"].items()
    }
    calibration_scenario = {
        **scenario("calibration"),
        "goals": [[100, 0, 0], [-100, 0, 0]],
        "reference_route_lengths": [100, 120],
    }
    result = calibrate_cost_critic(
        trainer,
        [calibration_scenario],
        repeats=2,
        output_dir=tmp_path / "calibration",
    )
    after = trainer.backend.state_dict()["cost_critics"]
    assert all(np.array_equal(value.numpy(), after[name].detach().cpu().numpy()) for name, value in before.items())
    assert result["overall"]["count"] == 2
    assert 0.0 <= result["overall"]["empirical_ucb_coverage"] <= 1.0
    assert result["rows"][0]["coverage_is_empirical_diagnostic"] is True
    assert result["critic_continuation_policy"] == "stochastic_policy"
    assert result["initial_action_conditioning"] == "same_sample_for_q_and_rollout"
    assert result["rows"][0]["paired_initial_action_matches_rollout"] is True
    assert result["rows"][0]["paired_initial_nominal_matches_rollout"] is True
    assert all(stochastic_flags)
    assert result["rows"][0]["runtime_hocbf"] == config.safety.enabled
    assert (tmp_path / "calibration" / "cost_calibration_summary.json").exists()
    assert len((tmp_path / "calibration" / "cost_calibration.jsonl").read_text().splitlines()) == 2

    diagnostic = evaluate_mean_policy_cost_diagnostic(
        trainer,
        [calibration_scenario],
        output_dir=tmp_path / "mean_diagnostic",
    )
    assert diagnostic["diagnostic_kind"] == "mean_policy_deployment_cost"
    assert diagnostic["critic_rollout_semantics_match"] is False
    assert (tmp_path / "mean_diagnostic" / "mean_policy_diagnostic_summary.json").exists()


def test_evaluator_can_capture_collection_time_critic_inputs(tmp_path):
    config = load_config(overrides={
        "environment": {"agent_count": 2, "max_steps": 2},
        "learning": {"batch_size": 8},
        "control": {"actor_mode": "full"},
        "monitoring": {"enabled": False},
    })
    trainer = SafeRLTrainer(config, build_env(config), output_dir=tmp_path / "train")
    result = evaluate(
        trainer,
        episodes=1,
        reset_options={"scenario": scenario("capture")},
        policy_execution_mode="stochastic",
        pair_initial_action_with_cost=True,
        capture_critic_inputs=True,
    )
    rollout = result["rollouts"][0]
    assert len(rollout["policy_observations"]) == len(rollout["costs"])
    assert len(rollout["actor_phases"]) == len(rollout["costs"])


def test_primary_statistics_cluster_by_training_seed():
    records = []
    for training_seed, success in ((0, 1.0), (1, 0.0)):
        records.append({
            "method": "full", "training_seed": training_seed,
            "episode_success": success,
            "separation_violation_pair_ticks": 0,
            "active_pair_ticks": 10,
            "path_stretch": 1.2 if success else None,
            "pure_intervention_squared_sum": 2.0,
            "active_agent_ticks": 10,
            "transmitted_bytes": 100,
            "active_agent_seconds": 1.0,
        })
    summary = aggregate_primary_by_method(records, bootstrap_samples=100, seed=0)["full"]
    assert summary["episode_success_rate"]["mean"] == 0.5
    assert summary["episode_success_rate"]["training_seed_count"] == 2
    assert summary["path_stretch"]["mean"] == 1.2


def test_random_scenario_bank_is_seeded_and_fixed_by_default():
    first = generate_random_traffic_bank(4, 2, seed=9)
    second = generate_random_traffic_bank(4, 2, seed=9)
    assert first == second
    positions = np.asarray(first[0]["positions"])
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    assert np.min(distances[np.triu_indices(4, 1)]) >= 20.0
    assert first[0]["spatial_side_length"] == 280.0
    assert first[0]["spatial_dimensions"] == [280.0, 280.0, 56.0]
    assert first[0]["altitude_range"] == [-28.0, 28.0]
    assert first[0]["spatial_scaling_mode"] == "fixed"
    assert first[0]["geometry"] == "boundary_crossing_corridor"
    assert first[0]["predicted_los_edges"]
    assert first[0]["los_prediction_threshold"] == 75.0
    assert first[0]["scenario_descriptor"]["predicted_edge_count"] >= 1
    assert first[0]["scenario_descriptor"]["candidate_empty"] is False
    assert first[0]["spatial_volume_factor"] == 1.0
    assert first[0]["spatial_volume"] == 280.0 * 280.0 * 56.0
    assert first[0]["scenario_descriptor"]["initial_cbf_edge_count"] >= 0
    assert first[0]["scenario_descriptor"]["generation_attempts"] == first[0]["generation_attempts"]
    assert first[0]["scenario_descriptor"]["candidate_zero_rejections"] == first[0]["candidate_zero_rejections"]
    half = first[0]["spatial_side_length"] / 2.0
    goals = np.asarray(first[0]["goals"])
    for index, (entry, exit_face) in enumerate(zip(first[0]["entry_faces"], first[0]["exit_faces"], strict=True)):
        assert {entry, exit_face} == {"x_min", "x_max"}
        assert abs(positions[index, 0]) == half
        assert abs(goals[index, 0]) == half
        assert positions[index, 0] == -goals[index, 0]
    fractions = np.asarray(first[0]["initial_speed_fractions"])
    assert np.all((0.4 <= fractions) & (fractions <= 0.65))


def test_time_aligned_cpa_uses_common_arrival_time():
    positions = np.asarray([[-10, 0, 0], [0, -10, 0]], dtype=float)
    goals = np.asarray([[10, 0, 0], [0, 10, 0]], dtype=float)
    velocities = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=float)
    edges, pairs = predicted_interaction_data(positions, velocities, goals, threshold=1.0, horizon=50.0)
    assert edges == [[0, 1]]
    assert pairs[0]["predicted_cpa_distance"] == 0.0
    assert pairs[0]["predicted_cpa_time"] == 10.0


def test_path_stretch_excludes_timeout_episodes(tmp_path):
    config = load_config(overrides={"environment": {"agent_count": 2, "max_steps": 1}, "learning": {"batch_size": 8}})
    trainer = SafeRLTrainer(config, build_env(config), output_dir=tmp_path)
    trainer.backend.act = lambda observation, deterministic: np.zeros((2, 3), dtype=np.float32)
    failed = {
        "scenario_id": "timeout",
        "positions": [[0, 0, 0], [20, 0, 0]],
        "goals": [[100, 0, 0], [-100, 0, 0]],
        "reference_route_lengths": [100, 120],
    }
    result = evaluate_scenario_bank(trainer, [failed])
    assert result["primary_metrics"]["episode_success_rate"] == 0.0
    assert result["primary_metrics"]["path_stretch"] is None
    assert result["episode_metrics"][0]["truncated"]


def test_mixed_n_generator_persists_capabilities_motion_and_conditions():
    bank = generate_random_traffic_bank(
        [2, 4], 8, seed=11,
        initial_speed_range=(2.0, 4.0),
        acceleration_range=(1.0, 3.0),
        velocity_limit_range=(8.0, 12.0),
        los_prediction_threshold=75.0,
    )
    assert {item["actual_agent_count"] for item in bank} == {2, 4}
    for item in bank:
        count = item["actual_agent_count"]
        assert len(item["maximum_accelerations"]) == count
        assert len(item["maximum_velocities"]) == count
        assert np.all(np.linalg.norm(item["velocities"], axis=1) >= 2.0)
        assert item["distance_thresholds"] == {"d_safe": 10.0, "d_warn": 30.0, "d_eng": 50.0}
        assert item["generator_version"] == 6
        assert item["predicted_los_edges"]
        assert item["predicted_interaction_edges"] == item["predicted_los_edges"]


def test_fixed_dimensions_do_not_change_with_agent_count():
    for count in (2, 4, 8, 16):
        item = generate_random_traffic_bank(count, 1, seed=100 + count)[0]
        assert item["spatial_dimensions"] == [280.0, 280.0, 56.0]
        assert item["spatial_volume_factor"] == 1.0
        assert item["scenario_descriptor"]["traffic_density"] == count / (280.0 * 280.0 * 56.0)


def test_density_controlled_mode_remains_available():
    item = generate_random_traffic_bank(
        8, 1, seed=88, spatial_scaling_mode="density_controlled",
        spatial_volume_scale_range=(1.0, 1.0),
    )[0]
    assert item["spatial_scaling_mode"] == "density_controlled"
    assert item["spatial_dimensions"] == [280.0, 280.0 * np.sqrt(2.0), 56.0 * np.sqrt(2.0)]


def test_primary_statistics_are_available_overall_and_conditioned_on_n():
    records = []
    for n, success in ((2, 1.0), (3, 0.0)):
        records.append({
            "method": "full", "training_seed": 0, "actual_agent_count": n,
            "episode_success": success, "separation_violation_pair_ticks": 0,
            "active_pair_ticks": 10, "path_stretch": 1.1 if success else None,
            "pure_intervention_squared_sum": 1.0, "active_agent_ticks": 10,
            "transmitted_bytes": 20, "active_agent_seconds": 1.0,
        })
    result = aggregate_primary_by_method_and_n(records, bootstrap_samples=20, seed=2)
    assert set(result["by_n"]) == {"2", "3"}
    assert result["overall"]["full"]["episode_success_rate"]["mean"] == 0.5
    assert result["by_n"]["2"]["full"]["episode_success_rate"]["sample_count"] == 1
    assert "episode_success_rate" in result["macro_by_n"]["full"]
