"""Safety metric aggregation."""

from __future__ import annotations

import numpy as np

from uav_safe_marl.core.types import SafetyResult


def summarize_safety(results: list[SafetyResult], nominal_actions: np.ndarray, d_safe: float, minimum_distance: float) -> dict[str, float]:
    """Summarize runtime events and learning-only CBF correction magnitude."""
    if nominal_actions.shape != (len(results), 3):
        raise ValueError("nominal_actions must have shape (len(results), 3)")
    magnitudes = [np.linalg.norm(item.intervention_action - nominal_actions[index]) for index, item in enumerate(results)]
    return {
        "minimum_pair_distance": float(minimum_distance),
        "d_safe_violation_count": float(minimum_distance < d_safe),
        "cbf_intervention_count": float(sum(m > 1e-8 for m in magnitudes)),
        "intervention_magnitude": float(sum(magnitudes)),
        "qp_infeasible_count": float(sum(item.infeasible for item in results)),
        "conflict_resolution_count": float(sum(item.resolver_used for item in results)),
        "skipped_constraint_count": float(sum(len(item.skipped_neighbours) for item in results)),
        "emergency_fallback_count": float(sum(item.emergency for item in results)),
        "responsibility_infeasible_count": float(sum(item.responsibility_infeasible_count for item in results)),
        "max_urgency": float(max((item.max_urgency for item in results), default=0.0)),
    }
