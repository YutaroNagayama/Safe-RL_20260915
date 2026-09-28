"""Dependency-light convex QP backend for three-dimensional acceleration."""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np

from uav_safe_marl.core.types import HOCBFConstraint


class ActiveSetQPSolver:
    """Project onto linear halfspaces and optional bounds by active-set enumeration.

    The control dimension is fixed at three, so at an optimum at most three
    linearly independent inequalities are active. This exact small-dimensional
    backend avoids coupling safety logic to a particular external solver.
    """

    def __init__(self, tolerance: float = 1e-8) -> None:
        self.tolerance = tolerance

    def _system(self, constraints, lower, upper):
        rows = [np.asarray(c.a, dtype=np.float64) for c in constraints]
        rhs = [float(c.b) for c in constraints]
        if lower is not None:
            for axis in range(3):
                unit = np.zeros(3); unit[axis] = 1.0
                rows.append(unit); rhs.append(float(lower[axis]))
        if upper is not None:
            for axis in range(3):
                unit = np.zeros(3); unit[axis] = -1.0
                rows.append(unit); rhs.append(float(-upper[axis]))
        return (np.vstack(rows), np.asarray(rhs)) if rows else (np.empty((0, 3)), np.empty(0))

    @staticmethod
    def _ball(velocity, speed_limit, dt):
        if velocity is None or speed_limit is None or dt is None:
            return None
        if dt <= 0.0 or speed_limit <= 0.0:
            raise ValueError("dt and speed_limit must be positive")
        return -np.asarray(velocity, dtype=np.float64) / dt, float(speed_limit) / dt

    def _feasible(self, point, matrix, vector, ball) -> bool:
        linear = bool(np.all(matrix @ point >= vector - self.tolerance))
        return linear and (ball is None or np.linalg.norm(point - ball[0]) <= ball[1] + self.tolerance)

    @staticmethod
    def _affine_geometry(active: np.ndarray, rhs: np.ndarray, center: np.ndarray, query: np.ndarray):
        if len(active) == 0:
            return center.copy(), query.copy(), np.eye(3)
        gram = active @ active.T
        if np.linalg.matrix_rank(gram, tol=1e-10) < len(active):
            return None
        inverse_rhs = np.linalg.solve(gram, rhs - active @ center)
        sphere_center = center + active.T @ inverse_rhs
        inverse_query = np.linalg.solve(gram, rhs - active @ query)
        query_projection = query + active.T @ inverse_query
        _, _, vh = np.linalg.svd(active)
        null_basis = vh[len(active):].T
        return sphere_center, query_projection, null_basis

    def _sphere_candidate(self, active, rhs, ball, query, objective=None):
        geometry = self._affine_geometry(active, rhs, ball[0], query)
        if geometry is None:
            return []
        sphere_center, query_projection, null_basis = geometry
        residual = ball[1] ** 2 - float(np.sum((sphere_center - ball[0]) ** 2))
        if residual < -self.tolerance or null_basis.shape[1] == 0:
            return []
        radius = np.sqrt(max(0.0, residual))
        if objective is None:
            direction = query_projection - sphere_center
        else:
            direction = null_basis @ (null_basis.T @ objective)
        norm = float(np.linalg.norm(direction))
        if norm > 1e-12:
            return [sphere_center + radius * direction / norm]
        return [sphere_center + sign * radius * null_basis[:, 0] for sign in (-1.0, 1.0)]

    def solve(self, nominal: np.ndarray, constraints: Sequence[HOCBFConstraint], lower: np.ndarray | None = None, upper: np.ndarray | None = None, *, velocity: np.ndarray | None = None, speed_limit: float | None = None, dt: float | None = None) -> tuple[np.ndarray, bool]:
        nominal = np.asarray(nominal, dtype=np.float64)
        if nominal.shape != (3,):
            raise ValueError("nominal must have shape (3,)")
        matrix, vector = self._system(constraints, lower, upper)
        ball = self._ball(velocity, speed_limit, dt)
        if len(matrix) == 0 and ball is None:
            return nominal.copy(), True

        candidates: list[np.ndarray] = [nominal] if self._feasible(nominal, matrix, vector, ball) else []
        for count in range(1, min(3, len(matrix)) + 1):
            for indices in combinations(range(len(matrix)), count):
                active = matrix[list(indices)]
                gram = active @ active.T
                if np.linalg.matrix_rank(gram, tol=1e-10) < count:
                    continue
                multipliers = np.linalg.solve(gram, vector[list(indices)] - active @ nominal)
                if np.any(multipliers < -self.tolerance):
                    continue
                point = nominal + active.T @ multipliers
                if self._feasible(point, matrix, vector, ball):
                    candidates.append(point)
        if ball is not None:
            for count in range(0, min(2, len(matrix)) + 1):
                for indices in combinations(range(len(matrix)), count):
                    active = matrix[list(indices)]
                    rhs = vector[list(indices)]
                    for point in self._sphere_candidate(active, rhs, ball, nominal):
                        if self._feasible(point, matrix, vector, ball):
                            candidates.append(point)
        if not candidates:
            return nominal.copy(), False
        best = min(candidates, key=lambda point: float(np.sum((point - nominal) ** 2)))
        return best, True

    def maximize_linear(self, objective: np.ndarray, constraints: Sequence[HOCBFConstraint], lower: np.ndarray, upper: np.ndarray, *, velocity: np.ndarray | None = None, speed_limit: float | None = None, dt: float | None = None) -> tuple[np.ndarray, bool]:
        """Maximize a linear barrier value over the bounded effective set."""
        direction = np.asarray(objective, dtype=np.float64)
        matrix, vector = self._system(constraints, lower, upper)
        ball = self._ball(velocity, speed_limit, dt)
        candidates: list[np.ndarray] = []
        for count in range(1, min(3, len(matrix)) + 1):
            for indices in combinations(range(len(matrix)), count):
                active, rhs = matrix[list(indices)], vector[list(indices)]
                if np.linalg.matrix_rank(active, tol=1e-10) < count:
                    continue
                if count == 3:
                    point = np.linalg.solve(active, rhs)
                    if self._feasible(point, matrix, vector, ball):
                        candidates.append(point)
                elif ball is not None:
                    for point in self._sphere_candidate(active, rhs, ball, ball[0], direction):
                        if self._feasible(point, matrix, vector, ball):
                            candidates.append(point)
        if ball is not None:
            for point in self._sphere_candidate(np.empty((0, 3)), np.empty(0), ball, ball[0], direction):
                if self._feasible(point, matrix, vector, ball):
                    candidates.append(point)
        # Box-only linear optimum is a corner; include all corners explicitly.
        if lower is not None and upper is not None:
            for bits in range(8):
                point = np.array([upper[k] if bits & (1 << k) else lower[k] for k in range(3)])
                if self._feasible(point, matrix, vector, ball):
                    candidates.append(point)
        if not candidates:
            projected, feasible = self.solve(np.zeros(3), constraints, lower, upper, velocity=velocity, speed_limit=speed_limit, dt=dt)
            if feasible:
                candidates.append(projected)
        if not candidates:
            return np.zeros(3), False
        return max(candidates, key=lambda point: float(direction @ point)), True
