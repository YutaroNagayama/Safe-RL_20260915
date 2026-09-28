import numpy as np

from uav_safe_marl import build_env, load_config


def test_default_waypoint_capture_radius_is_five_metres():
    config = load_config()
    assert config.environment.goal_tolerance == 5.0
    assert config.scenario.interaction_threshold == 75.0
    assert config.scenario.spatial_scaling_mode == "fixed"
    assert (config.scenario.longitudinal_length, config.scenario.transverse_size) == (280.0, 280.0)
    assert (config.scenario.altitude_min, config.scenario.altitude_max) == (-28.0, 28.0)
    assert (config.scenario.volume_scale_min, config.scenario.volume_scale_max) == (1.0, 1.0)
    assert config.objective.goal_reward == 15.0
    assert config.objective.absolute_distance_scale == 280.0
    assert config.objective.w_absolute_distance == 0.002


def test_gymnasium_reset_and_step_contract():
    env = build_env(load_config(overrides={"environment": {"max_steps": 1}}))
    observation, info = env.reset(seed=42)
    assert env.observation_space.contains(observation)
    result = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert len(result) == 5
    next_observation, reward, terminated, truncated, step_info = result
    assert env.observation_space.contains(next_observation)
    assert isinstance(reward, float) and isinstance(terminated, bool) and truncated
    assert {"cost", "nominal_action", "safe_action", "d_safe_violation_count"} <= set(step_info)


def test_reached_agent_exits_interactions_on_the_following_tick():
    config = load_config(overrides={"environment": {"agent_count": 2, "max_steps": 3}})
    env = build_env(config)
    scenario = {
        "scenario_id": "one-exits",
        "positions": [[0, 0, 0], [40, 0, 0]],
        "velocities": [[0, 0, 0], [0, 0, 0]],
        "goals": [[1, 0, 0], [-40, 0, 0]],
        "reference_route_lengths": [1, 80],
    }
    env.reset(options={"scenario": scenario, "predicted_los_edges": [(0, 1)]})
    _, _, terminated, truncated, first = env.step(np.zeros((2, 3), dtype=np.float32))
    assert not terminated and not truncated
    np.testing.assert_array_equal(first["active_mask_after"], [False, True])
    assert first["pair_tick_count"] == 1
    exited_position = env.states[0].position.copy()
    observation, _, _, _, second = env.step(np.zeros((2, 3), dtype=np.float32))
    assert second["active_agent_count"] == 1
    assert second["pair_tick_count"] == 0
    np.testing.assert_array_equal(observation[0], np.zeros(14))
    np.testing.assert_array_equal(env.states[0].position, exited_position)
    np.testing.assert_array_equal(env.states[0].velocity, np.zeros(3))
    assert second["safety"][1]["active_neighbours"] == ()


def test_task_completion_takes_precedence_over_timeout():
    config = load_config(overrides={"environment": {"agent_count": 2, "max_steps": 1}})
    env = build_env(config)
    scenario = {
        "positions": [[0, 0, 0], [20, 0, 0]],
        "goals": [[1, 0, 0], [21, 0, 0]],
        "reference_route_lengths": [1, 1],
    }
    env.reset(options={"scenario": scenario})
    _, _, terminated, truncated, _ = env.step(np.zeros((2, 3), dtype=np.float32))
    assert terminated and not truncated


def test_pure_intervention_excludes_runtime_speed_correction():
    config = load_config(overrides={"environment": {"agent_count": 1, "max_steps": 2}, "control": {"actor_mode": "full"}})
    env = build_env(config)
    scenario = {
        "positions": [[0, 0, 0]],
        "velocities": [[12, 0, 0]],
        "goals": [[100, 0, 0]],
        "reference_route_lengths": [100],
    }
    env.reset(options={"scenario": scenario})
    _, _, _, _, info = env.step(np.array([[1, 0, 0]], dtype=np.float32))
    assert info["pure_intervention_squared_sum"] == 0.0
    assert info["runtime_correction_squared_sum"] > 0.0


def test_cbf_off_still_computes_counterfactual_pure_intervention():
    config = load_config(overrides={
        "environment": {"agent_count": 2, "max_steps": 2},
        "safety": {"enabled": False},
        "control": {"actor_mode": "full"},
    })
    env = build_env(config)
    scenario = {
        "positions": [[0, 0, 0], [8, 0, 0]],
        "goals": [[-100, 0, 0], [100, 0, 0]],
        "reference_route_lengths": [100, 92],
    }
    env.reset(options={"scenario": scenario})
    _, _, _, _, info = env.step(np.zeros((2, 3), dtype=np.float32))
    assert info["pure_intervention_squared_sum"] > 0.0
    assert info["runtime_correction_squared_sum"] == 0.0
