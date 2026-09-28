import numpy as np

from uav_safe_marl.core.types import HOCBFConstraint
from uav_safe_marl.safety.conflict_resolver import PriorityConflictResolver


def constraint(neighbour, a, b, urgency):
    return HOCBFConstraint(neighbour, np.asarray(a, float), b, urgency)


def test_priority_uses_maximal_feasible_prefix_and_stops_after_first_conflict():
    resolver = PriorityConflictResolver()
    constraints = [constraint(1, [1, 0, 0], 2, 2), constraint(2, [-1, 0, 0], 2, 1), constraint(3, [0, 1, 0], 1, 0.5)]
    result = resolver.resolve(np.zeros(3), constraints, -np.ones(3) * 3, np.ones(3) * 3)
    assert result.active_neighbours == (1,)
    assert result.skipped_neighbours == (3,)
    # Stage 1 maximizes the next-priority barrier (-x) while preserving x>=2;
    # Stage 2 therefore projects the nominal action onto the lexicographic x=2 face.
    np.testing.assert_allclose(result.safe_action, [2, 0, 0], atol=1e-6)


def test_emergency_uses_best_bound_when_top_constraint_alone_is_impossible():
    result = PriorityConflictResolver().resolve(np.zeros(3), [constraint(1, [1, 0, 0], 5, 5)], -np.ones(3) * 3, np.ones(3) * 3)
    assert result.emergency and result.safe_action[0] == 3


def test_impossible_top_constraint_is_never_replaced_by_feasible_lower_one():
    constraints = [constraint(1, [1, 0, 0], 5, 5), constraint(2, [0, 1, 0], 1, 1)]
    result = PriorityConflictResolver().resolve(np.zeros(3), constraints, -np.ones(3) * 3, np.ones(3) * 3)
    assert result.emergency
    assert result.active_neighbours == (1,)
    assert result.skipped_neighbours == (2,)


def test_learning_resolver_uses_minimum_slack_then_minimum_correction():
    constraints = [constraint(1, [1, 0, 0], 2, 2), constraint(2, [-1, 0, 0], 2, 1)]
    result = PriorityConflictResolver().resolve_learning(np.zeros(3), constraints)
    assert result.infeasible
    np.testing.assert_allclose(result.safe_action, [2, 0, 0], atol=1e-6)
