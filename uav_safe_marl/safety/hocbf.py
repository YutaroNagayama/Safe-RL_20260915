"""Relative-degree-two HOCBF equations for pairwise separation."""

from __future__ import annotations

import numpy as np

from uav_safe_marl.core.types import AgentState, HOCBFConstraint


def hocbf_terms(first: AgentState, second: AgentState, d_safe: float, k1: float, k2: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return ``r_ij``, ``v_ij``, ``h_ij``, and ``q_ij`` from PDF slide 25."""
    relative_position = first.position - second.position
    relative_velocity = first.velocity - second.velocity
    h_ij = float(relative_position @ relative_position - d_safe * d_safe)
    q_ij = float(
        2.0 * (relative_velocity @ relative_velocity)
        + 2.0 * (k1 + k2) * (relative_position @ relative_velocity)
        + k1 * k2 * h_ij
    )
    return relative_position, relative_velocity, h_ij, q_ij


def hocbf_margins(first: AgentState, second: AgentState, d_safe: float, k1: float) -> tuple[float, float]:
    """Return the relative-degree-two initial-set margins ``h`` and ``psi_1``."""
    relative_position = first.position - second.position
    relative_velocity = first.velocity - second.velocity
    h_ij = float(relative_position @ relative_position - d_safe * d_safe)
    psi_1 = float(2.0 * (relative_position @ relative_velocity) + k1 * h_ij)
    return h_ij, psi_1


def build_local_constraint(neighbour_id: int, relative_position: np.ndarray, q_ij: float, rho_ij: float, nominal: np.ndarray, responsibility_feasible: bool = True) -> HOCBFConstraint:
    """Build ``a^T u_i >= b`` for ``2 r^T u_i + rho q >= 0``."""
    a = 2.0 * np.asarray(relative_position, dtype=np.float64)
    b = -float(rho_ij * q_ij)
    norm = float(np.linalg.norm(a))
    urgency = max(0.0, b - float(a @ nominal)) / norm if norm > 1e-12 else (float("inf") if b > 0 else 0.0)
    return HOCBFConstraint(neighbour_id=neighbour_id, a=a, b=b, urgency=urgency, responsibility_feasible=responsibility_feasible)


def satisfies(action: np.ndarray, constraint: HOCBFConstraint, tolerance: float = 1e-8) -> bool:
    """Return whether an action satisfies one local HOCBF inequality."""
    return float(constraint.a @ action) >= constraint.b - tolerance
