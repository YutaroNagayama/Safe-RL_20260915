"""Priority resolver for infeasible multi-neighbour HOCBF constraints."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from uav_safe_marl.core.types import HOCBFConstraint, SafetyResult
from .qp_backend import ActiveSetQPSolver


class PriorityConflictResolver:
    """Lexicographic maximal-feasible-prefix resolver for runtime and learning."""

    def __init__(self, solver: ActiveSetQPSolver | None = None) -> None:
        self.solver = solver or ActiveSetQPSolver()

    def resolve(self, nominal: np.ndarray, constraints: Sequence[HOCBFConstraint], lower: np.ndarray, upper: np.ndarray, *, velocity: np.ndarray | None = None, speed_limit: float | None = None, dt: float | None = None) -> SafetyResult:
        """Resolve a bounded runtime conflict and optimize the first failed barrier."""
        ordered = sorted(constraints, key=lambda item: (-item.urgency, item.neighbour_id))
        accepted: list[HOCBFConstraint] = []
        failed_index = None
        for index, constraint in enumerate(ordered):
            _, feasible = self.solver.solve(nominal, [*accepted, constraint], lower, upper, velocity=velocity, speed_limit=speed_limit, dt=dt)
            if feasible:
                accepted.append(constraint)
            else:
                failed_index = index
                break
        if failed_index is None:
            current, _ = self.solver.solve(nominal, accepted, lower, upper, velocity=velocity, speed_limit=speed_limit, dt=dt)
            return SafetyResult(current, current.copy(), tuple(c.neighbour_id for c in accepted))

        failed = ordered[failed_index]
        stage_one, stage_one_feasible = self.solver.maximize_linear(
            failed.a, accepted, lower, upper,
            velocity=velocity, speed_limit=speed_limit, dt=dt,
        )
        if not stage_one_feasible:
            stage_one = np.clip(nominal, lower, upper)
        achieved = float(failed.a @ stage_one)
        relaxed = HOCBFConstraint(failed.neighbour_id, failed.a, achieved, failed.urgency, failed.responsibility_feasible)
        current, stage_two_feasible = self.solver.solve(
            nominal, [*accepted, relaxed], lower, upper,
            velocity=velocity, speed_limit=speed_limit, dt=dt,
        )
        if not stage_two_feasible:
            current = stage_one
        emergency = failed_index == 0
        return SafetyResult(
            safe_action=current,
            intervention_action=current.copy(),
            active_neighbours=tuple(c.neighbour_id for c in accepted) if not emergency else (failed.neighbour_id,),
            # The failed constraint is handled as the lexicographic soft face;
            # only strictly lower-priority constraints are omitted.
            skipped_neighbours=tuple(item.neighbour_id for item in ordered[failed_index + 1:]),
            infeasible=True,
            resolver_used=True,
            emergency=emergency,
            max_urgency=max((c.urgency for c in ordered), default=0.0),
        )

    def resolve_learning(self, nominal: np.ndarray, constraints: Sequence[HOCBFConstraint]) -> SafetyResult:
        """Resolve unbounded-input HOCBF conflicts through minimum slack."""
        ordered = sorted(constraints, key=lambda item: (-item.urgency, item.neighbour_id))
        accepted: list[HOCBFConstraint] = []
        failed_index = None
        for index, constraint in enumerate(ordered):
            _, feasible = self.solver.solve(nominal, [*accepted, constraint])
            if feasible:
                accepted.append(constraint)
            else:
                failed_index = index
                break
        if failed_index is None:
            action, _ = self.solver.solve(nominal, accepted)
            return SafetyResult(action, action.copy(), tuple(c.neighbour_id for c in accepted))

        failed = ordered[failed_index]

        def feasible_with_slack(slack: float) -> tuple[np.ndarray, bool]:
            relaxed = HOCBFConstraint(failed.neighbour_id, failed.a, failed.b - slack, failed.urgency, failed.responsibility_feasible)
            return self.solver.solve(nominal, [*accepted, relaxed])

        lower_slack, upper_slack = 0.0, 1.0
        _, feasible = feasible_with_slack(upper_slack)
        while not feasible and upper_slack < 1e12:
            upper_slack *= 2.0
            _, feasible = feasible_with_slack(upper_slack)
        if not feasible:
            action, _ = self.solver.solve(nominal, accepted)
            return SafetyResult(action, action.copy(), tuple(c.neighbour_id for c in accepted), tuple(item.neighbour_id for item in ordered[failed_index:]), True, True, failed_index == 0, max((c.urgency for c in ordered), default=0.0))
        for _ in range(70):
            middle = 0.5 * (lower_slack + upper_slack)
            _, feasible = feasible_with_slack(middle)
            if feasible:
                upper_slack = middle
            else:
                lower_slack = middle
        action, _ = feasible_with_slack(upper_slack + self.solver.tolerance)
        return SafetyResult(
            safe_action=action,
            intervention_action=action.copy(),
            active_neighbours=tuple(c.neighbour_id for c in accepted),
            skipped_neighbours=tuple(item.neighbour_id for item in ordered[failed_index + 1:]),
            infeasible=True,
            resolver_used=True,
            emergency=failed_index == 0,
            max_urgency=max((c.urgency for c in ordered), default=0.0),
        )
