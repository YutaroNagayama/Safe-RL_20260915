import numpy as np

from uav_safe_marl.core.types import AgentState
from uav_safe_marl.objectives.cost import distance_cost
from uav_safe_marl.objectives.intervention import intervention_cost
from uav_safe_marl.objectives.reward import GoalProgressReward


def state(x, goal=(100.0, 0.0, 0.0)):
    return AgentState(np.asarray(x, float), np.zeros(3), np.asarray(goal, float), np.ones(3), np.ones(3) * 10)


def test_distance_cost_averages_each_agents_closest_peer_cost():
    states = [state((0, 0, 0)), state((15, 0, 0)), state((30, 0, 0))]
    assert distance_cost(states, 10.0, 20.0) == 0.25


def test_distance_cost_does_not_sum_multiple_peer_costs_per_agent():
    states = [state((0, 0, 0)), state((15, 0, 0)), state((-15, 0, 0))]
    assert distance_cost(states, 10.0, 20.0) == 0.25


def test_distance_cost_selects_the_highest_risk_peer_per_agent():
    states = [state((0, 0, 0)), state((15, 0, 0)), state((-12, 0, 0))]
    # Per-agent maxima are 0.64, 0.25, and 0.64 respectively.
    assert np.isclose(distance_cost(states, 10.0, 20.0), 0.51)


def test_distance_cost_averages_over_all_active_agents():
    two = [state((0, 0, 0)), state((15, 0, 0))]
    four = [state((0, 0, 0)), state((15, 0, 0)), state((100, 0, 0)), state((200, 0, 0))]
    assert distance_cost(two, 10.0, 20.0) == 0.25
    assert distance_cost(four, 10.0, 20.0) == 0.125


def test_distance_cost_excludes_inactive_agents_before_nearest_selection():
    states = [state((0, 0, 0)), state((1, 0, 0)), state((15, 0, 0))]
    assert distance_cost(states, 10.0, 20.0, np.array([True, False, True])) == 0.25
    assert distance_cost(states, 10.0, 20.0, np.array([True, False, False])) == 0.0


def test_intervention_is_mean_physical_squared_norm():
    corrections = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
    assert intervention_cost(corrections) == 12.5


def test_reward_has_goal_progress_and_time_only():
    model = GoalProgressReward(goal_reward=10.0, w_progress=2.0, w_time=0.5, absolute_distance_enabled=False)
    reward, terms = model.compute([state((0, 0, 0))], [state((1, 0, 0))], np.array([False]))
    assert reward == 1.5
    assert set(terms) == {"goal_reward", "progress", "time_penalty", "absolute_distance_penalty", "mean_normalized_goal_distance", "residual_penalty", "mean_normalized_residual_squared"}
    assert terms["absolute_distance_penalty"] == 0.0


def test_absolute_goal_distance_penalty_uses_active_mean_and_does_not_clip():
    model = GoalProgressReward(0.0, 0.0, 0.0, absolute_distance_scale=280.0, w_absolute_distance=0.002)
    previous = [state((0, 0, 0), goal=(280, 0, 0)), state((0, 0, 0), goal=(560, 0, 0))]
    current = [state((0, 0, 0), goal=(280, 0, 0)), state((0, 0, 0), goal=(560, 0, 0))]
    reward, terms = model.compute(previous, current, np.array([False, False]), np.array([True, True]))
    assert terms["mean_normalized_goal_distance"] == 1.5
    assert terms["absolute_distance_penalty"] == 0.003
    assert reward == -0.003

    _, first_only = model.compute(previous, current, np.array([False, False]), np.array([True, False]))
    assert first_only["absolute_distance_penalty"] == 0.002


def test_goal_reward_is_normalized_by_present_agent_count():
    model = GoalProgressReward(15.0, 0.0, 0.0, absolute_distance_enabled=False)
    states = [state((0, 0, 0)), state((0, 0, 0))]
    _, one = model.compute(states, states, np.array([True, False]))
    _, both = model.compute(states, states, np.array([True, True]))
    assert one["goal_reward"] == 7.5
    assert both["goal_reward"] == 15.0


def test_directional_progress_is_normalized_and_signed():
    previous = [state((0, 0, 0))]
    forward = [state((1, 0, 0))]
    backward = [state((-1, 0, 0))]
    model = GoalProgressReward(0.0, 1.0, 0.0, "directional", dt=0.1)
    _, forward_terms = model.compute(previous, forward, np.array([False]))
    _, backward_terms = model.compute(previous, backward, np.array([False]))
    assert forward_terms["progress"] == 1.0
    assert backward_terms["progress"] == -1.0
