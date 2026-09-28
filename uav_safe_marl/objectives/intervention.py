"""Safety-layer intervention cost."""

import numpy as np


def intervention_cost(corrections: np.ndarray, active_mask: np.ndarray | None = None) -> float:
    """PDF slide 29: mean squared physical acceleration correction per UAV."""
    values = np.asarray(corrections, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("corrections must have shape (agent_count, 3)")
    if active_mask is not None:
        active = np.asarray(active_mask, dtype=bool)
        if active.shape != (len(values),):
            raise ValueError("active_mask must match corrections")
        values = values[active]
    return float(np.mean(np.sum(values * values, axis=1))) if len(values) else 0.0
