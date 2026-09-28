import numpy as np
import torch

from uav_safe_marl import build_env, load_config
from uav_safe_marl.control import (
    compose_action_numpy,
    reference_action_numpy,
    reference_action_torch,
)
from uav_safe_marl.core.types import AgentState
from uav_safe_marl.graph.observations import direct_node_features


def states():
    return [
        AgentState(
            np.array([0.0, 0.0, 0.0]), np.array([1.0, -1.0, 0.5]),
            np.array([100.0, 20.0, -10.0]), np.full(3, 3.0), np.full(3, 12.0),
        ),
        AgentState(
            np.array([10.0, 5.0, 0.0]), np.array([-2.0, 0.5, 0.0]),
            np.array([-80.0, 5.0, 0.0]), np.full(3, 2.0), np.full(3, 8.0),
        ),
    ]


def test_numpy_torch_reference_guidance_parity():
    value = states()
    reference = reference_action_numpy(value, 0.75, 0.8)
    features = direct_node_features(value, reference_actions=reference)
    torch_reference = reference_action_torch(torch.as_tensor(features, dtype=torch.float64), 0.75, 0.8)
    np.testing.assert_allclose(torch_reference.numpy(), reference, atol=1e-12)


def test_residual_is_bounded_and_zero_residual_equals_reference():
    limits = np.array([[2.0, 3.0, 4.0]])
    reference = np.array([[0.5, -0.5, 1.0]])
    residual, command, nominal = compose_action_numpy(
        np.array([[1.0, -1.0, 0.25]]), reference, limits, "residual", 0.5
    )
    np.testing.assert_allclose(residual, [[1.0, -1.5, 0.5]])
    assert np.all(np.abs(residual) <= 0.5 * limits + 1e-12)
    np.testing.assert_allclose(command, reference + residual)
    np.testing.assert_allclose(nominal, np.clip(command, -limits, limits))

    zero_residual, zero_command, zero_nominal = compose_action_numpy(
        np.zeros((1, 3)), reference, limits, "residual", 0.5
    )
    np.testing.assert_array_equal(zero_residual, np.zeros((1, 3)))
    np.testing.assert_allclose(zero_command, reference)
    np.testing.assert_allclose(zero_nominal, reference)


def test_environment_exposes_complete_residual_action_chain():
    config = load_config(overrides={"environment": {"agent_count": 1, "max_steps": 1}})
    env = build_env(config)
    env.reset(options={"scenario": {
        "positions": [[0.0, 0.0, 0.0]], "velocities": [[0.0, 0.0, 0.0]],
        "goals": [[100.0, 0.0, 0.0]], "reference_route_lengths": [100.0],
    }})
    normalized = np.array([[0.2, -0.4, 0.0]])
    _, _, _, _, info = env.step(normalized)
    np.testing.assert_allclose(info["residual_action"], 0.5 * 3.0 * normalized)
    np.testing.assert_allclose(info["command_action"], info["reference_action"] + info["residual_action"])
    np.testing.assert_allclose(info["nominal_action"], np.clip(info["command_action"], -3.0, 3.0))
    assert info["reward_terms"]["residual_penalty"] > 0.0


def test_reference_only_mode_ignores_actor_input_but_keeps_physical_filter():
    config = load_config(overrides={
        "control": {"actor_mode": "reference_only"},
        "safety": {"enabled": False},
        "environment": {"agent_count": 1, "max_steps": 1},
    })
    env = build_env(config)
    env.reset()
    _, _, _, _, first = env.step(np.ones((1, 3)))
    env.reset()
    _, _, _, _, second = env.step(-np.ones((1, 3)))
    np.testing.assert_allclose(first["nominal_action"], second["nominal_action"])
    np.testing.assert_array_equal(first["normalized_residual_action"], np.zeros((1, 3)))
