import numpy as np

from uav_safe_marl.core.types import AgentState
from uav_safe_marl.safety.responsibility import CapabilityResponsibilityAllocator
from uav_safe_marl.safety.qp_backend import ActiveSetQPSolver
from uav_safe_marl.safety.responsibility import effective_bounds


def state(position, limit):
    return AgentState(np.asarray(position, float), np.zeros(3), np.zeros(3), np.ones(3) * limit, np.ones(3) * 20)


def test_pair_responsibility_is_complementary_and_capability_aware():
    allocator = CapabilityResponsibilityAllocator("capability", dt=0.1)
    rho_i, rho_j = allocator.allocate_pair(state((0, 0, 0), 3), state((8, 0, 0), 1), q_ij=-10)
    assert abs(rho_i + rho_j - 1.0) < 1e-12
    assert rho_i > rho_j
    reverse_j, reverse_i = allocator.allocate_pair(state((8, 0, 0), 1), state((0, 0, 0), 3), q_ij=-10)
    np.testing.assert_allclose([rho_i, rho_j], [reverse_i, reverse_j])


def test_zero_distance_and_zero_capability_fallback_to_equal():
    allocator = CapabilityResponsibilityAllocator("capability", dt=0.1)
    rho = allocator.allocate_pair(state((0, 0, 0), 0), state((0, 0, 0), 0), q_ij=-1)
    assert rho == (0.5, 0.5)


def test_insufficient_combined_capability_is_detected():
    allocator = CapabilityResponsibilityAllocator("capability", dt=0.1)
    allocation = allocator.allocate_pair_details(state((0, 0, 0), 0.01), state((1, 0, 0), 0.01), q_ij=-100)
    assert not allocation.feasible
    np.testing.assert_allclose(allocation.rho_first, allocation.kappa_first / (allocation.kappa_first + allocation.kappa_second))


def test_infeasible_pair_preserves_asymmetric_capability_ratio():
    allocator = CapabilityResponsibilityAllocator("capability", dt=0.1)
    allocation = allocator.allocate_pair_details(state((0, 0, 0), 9), state((1, 0, 0), 1), q_ij=-100)
    assert not allocation.feasible
    assert allocation.rho_first > allocation.rho_second
    np.testing.assert_allclose(allocation.rho_first + allocation.rho_second, 1.0)


def test_norm_speed_capability_uses_the_same_effective_set_as_runtime_qp():
    first = AgentState(np.zeros(3), np.array([-8.4, -8.4, 0.0]), np.ones(3), np.ones(3) * 3.0, np.ones(3) * 12.0)
    second = AgentState(np.array([1.0, 1.0, 0.0]), np.zeros(3), np.ones(3), np.ones(3) * 3.0, np.ones(3) * 12.0)
    allocator = CapabilityResponsibilityAllocator("capability", dt=0.1, speed_constraint_mode="norm")
    allocation = allocator.allocate_pair_details(first, second, q_ij=-1.0)
    coefficient = 2.0 * (first.position - second.position)
    lower, upper = effective_bounds(first, 0.1, "norm")
    maximizer, feasible = ActiveSetQPSolver().maximize_linear(
        coefficient, [], lower, upper,
        velocity=first.velocity, speed_limit=12.0, dt=0.1,
    )
    assert feasible
    np.testing.assert_allclose(allocation.kappa_first, max(0.0, coefficient @ maximizer), atol=1e-8)
    box_lower, box_upper = effective_bounds(first, 0.1, "box")
    old_box_kappa = max(0.0, coefficient @ np.where(coefficient >= 0.0, box_upper, box_lower))
    assert allocation.kappa_first < old_box_kappa
