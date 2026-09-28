import numpy as np

from uav_safe_marl.core.types import AgentState, HOCBFConstraint
from uav_safe_marl.envs.dynamics import DoubleIntegratorDynamics
from uav_safe_marl.safety.hocbf import hocbf_margins, hocbf_terms
from uav_safe_marl.safety.qp_filter import HOCBFSafetyFilter
from uav_safe_marl.safety.responsibility import CapabilityResponsibilityAllocator
from uav_safe_marl.safety.execution import SafetyExecutor


def state(position, acceleration=3.0):
    return AgentState(np.asarray(position, float), np.zeros(3), np.zeros(3), np.ones(3) * acceleration, np.ones(3) * 20)


def make_filter(mode="equal"):
    allocator = CapabilityResponsibilityAllocator(mode, dt=0.1)
    return HOCBFSafetyFilter(10.0, 25.0, 1.0, 1.0, 0.1, allocator)


def test_hocbf_pdf_equation():
    r, v, h, q = hocbf_terms(state((0, 0, 0)), state((8, 0, 0)), 10.0, 1.0, 1.0)
    np.testing.assert_allclose(r, [-8, 0, 0]); np.testing.assert_allclose(v, 0)
    assert h == -36.0 and q == -36.0


def test_safe_nominal_action_has_zero_correction():
    states = [state((0, 0, 0)), state((20, 0, 0))]
    result = make_filter().filter(0, states, np.zeros(3))
    np.testing.assert_allclose(result.safe_action, 0.0, atol=1e-9)
    np.testing.assert_allclose(result.intervention_action, 0.0, atol=1e-9)


def test_unsafe_nominal_is_corrected_away_and_satisfies_hocbf():
    states = [state((0, 0, 0)), state((8, 0, 0))]
    result = make_filter().filter(0, states, np.zeros(3))
    assert result.safe_action[0] < 0.0
    constraint = make_filter().constraints_for(0, states, np.zeros(3))[0]
    assert constraint.a @ result.safe_action >= constraint.b - 1e-8
    np.testing.assert_allclose(result.safe_action[0], -1.125, atol=1e-8)


def test_sampled_hocbf_preserves_separation_across_supported_base_ticks():
    """Regression check for the continuous-HOCBF-at-each-tick design (R7)."""
    for dt in (0.05, 0.1, 0.2):
        states = [
            AgentState(np.array([-10.0, 0, 0]), np.array([2.0, 0, 0]), np.array([10.0, 0, 0]), np.ones(3) * 3, np.ones(3) * 20),
            AgentState(np.array([10.0, 0, 0]), np.array([-2.0, 0, 0]), np.array([-10.0, 0, 0]), np.ones(3) * 3, np.ones(3) * 20),
        ]
        assert min(hocbf_margins(states[0], states[1], 10.0, 1.0)) >= 0.0
        safety_filter = HOCBFSafetyFilter(10.0, 25.0, 1.0, 1.0, dt, CapabilityResponsibilityAllocator("capability", dt))
        dynamics = DoubleIntegratorDynamics()
        minimum_distance = float("inf")
        for _ in range(round(8.0 / dt)):
            actions = np.vstack([safety_filter.filter(i, states, np.zeros(3)).safe_action for i in range(2)])
            states = dynamics.step(states, actions, dt)
            minimum_distance = min(minimum_distance, np.linalg.norm(states[0].position - states[1].position))
        assert minimum_distance >= 10.0


def test_learning_intervention_retries_all_constraints_without_runtime_bounds():
    own = state((0, 0, 0), acceleration=3.0)
    safety_filter = make_filter()
    constraints = [
        HOCBFConstraint(1, np.array([1.0, 0, 0]), 4.0, 4.0),
        HOCBFConstraint(2, np.array([0.0, 1, 0]), 1.0, 1.0),
    ]
    safety_filter.constraints_for = lambda *_: constraints
    result = safety_filter.filter(0, [own], np.zeros(3))
    assert result.emergency
    np.testing.assert_allclose(result.intervention_action, [4, 1, 0], atol=1e-7)


def test_norm_speed_constraint_rejects_diagonal_box_feasible_velocity():
    moving = AgentState(np.zeros(3), np.array([8.0, 8.0, 0]), np.ones(3), np.ones(3) * 3, np.ones(3) * 12)
    allocator = CapabilityResponsibilityAllocator("capability", dt=0.1, speed_constraint_mode="norm")
    safety_filter = HOCBFSafetyFilter(10, 25, 1, 1, 0.1, allocator, speed_constraint_mode="norm")
    result = safety_filter.filter(0, [moving], np.array([3.0, 3.0, 0]))
    assert np.linalg.norm(moving.velocity + 0.1 * result.safe_action) <= 12.0 + 1e-7


def test_pair_precomputation_and_spawn_process_backend_match_serial_filter():
    states = [
        AgentState(np.array([-9.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]), np.zeros(3), np.full(3, 3.0), np.full(3, 12.0)),
        AgentState(np.array([9.0, 0.0, 0.0]), np.array([-2.0, 0.0, 0.0]), np.zeros(3), np.full(3, 2.0), np.full(3, 10.0)),
    ]
    active = np.ones(2, dtype=bool)
    nominal = np.array([[1.0, 0.2, 0.0], [-1.0, -0.1, 0.0]])
    allocator = CapabilityResponsibilityAllocator("capability", 0.1, speed_constraint_mode="norm")
    safety_filter = HOCBFSafetyFilter(10.0, 50.0, 1.0, 1.0, 0.1, allocator, True, "norm")
    expected = [safety_filter.filter(i, states, nominal[i], active) for i in range(2)]
    executor = SafetyExecutor(safety_filter, "process", 2, 1)
    actual, engaged_agents, engaged_pairs = executor.solve(states, nominal, active, True)
    executor.close()
    assert engaged_agents == 2 and engaged_pairs == 1
    for first, second in zip(expected, actual, strict=True):
        np.testing.assert_allclose(first.safe_action, second.safe_action)
        np.testing.assert_allclose(first.intervention_action, second.intervention_action)
        assert first.active_neighbours == second.active_neighbours
        assert first.infeasible == second.infeasible
