"""Optional independent numerical oracle for the runtime 3-D QP backend."""

import numpy as np
import pytest

cp = pytest.importorskip("cvxpy", reason="install .[verify] for independent QP validation")

from uav_safe_marl.core.types import HOCBFConstraint
from uav_safe_marl.safety.qp_backend import ActiveSetQPSolver


@pytest.mark.parametrize("with_speed_ball", [False, True])
def test_random_projection_matches_cvxpy(with_speed_ball):
    rng = np.random.default_rng(20260922)
    solver = ActiveSetQPSolver(tolerance=1e-7)
    for _ in range(30):
        feasible_seed = rng.uniform(-1.0, 1.0, 3)
        lower, upper = np.full(3, -3.0), np.full(3, 3.0)
        rows = rng.normal(size=(5, 3))
        constraints = [
            HOCBFConstraint(index, row, float(row @ feasible_seed - rng.uniform(0.05, 1.0)), 0.0)
            for index, row in enumerate(rows)
        ]
        nominal = rng.normal(size=3) * 4.0
        kwargs = {}
        if with_speed_ball:
            kwargs = {"velocity": np.zeros(3), "speed_limit": 2.5, "dt": 1.0}
        internal, feasible = solver.solve(nominal, constraints, lower, upper, **kwargs)
        assert feasible

        action = cp.Variable(3)
        cvx_constraints = [rows @ action >= np.asarray([item.b for item in constraints]), action >= lower, action <= upper]
        if with_speed_ball:
            cvx_constraints.append(cp.norm(action, 2) <= 2.5)
        problem = cp.Problem(cp.Minimize(cp.sum_squares(action - nominal)), cvx_constraints)
        problem.solve(solver=cp.CLARABEL)
        assert problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
        # Independent conic and active-set solvers terminate under different
        # feasibility tolerances near a sphere/half-space intersection.
        np.testing.assert_allclose(internal, action.value, atol=1e-4, rtol=1e-4)


def test_random_linear_maximization_matches_cvxpy():
    rng = np.random.default_rng(20260923)
    solver = ActiveSetQPSolver(tolerance=1e-7)
    for _ in range(30):
        lower, upper = np.full(3, -2.0), np.full(3, 2.0)
        objective = rng.normal(size=3)
        seed = rng.uniform(-1.0, 1.0, 3)
        rows = rng.normal(size=(4, 3))
        constraints = [
            HOCBFConstraint(index, row, float(row @ seed - rng.uniform(0.05, 0.8)), 0.0)
            for index, row in enumerate(rows)
        ]
        internal, feasible = solver.maximize_linear(objective, constraints, lower, upper)
        assert feasible
        action = cp.Variable(3)
        problem = cp.Problem(
            cp.Maximize(objective @ action),
            [rows @ action >= np.asarray([item.b for item in constraints]), action >= lower, action <= upper],
        )
        problem.solve(solver=cp.CLARABEL)
        assert problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
        np.testing.assert_allclose(objective @ internal, problem.value, atol=2e-5, rtol=2e-5)
