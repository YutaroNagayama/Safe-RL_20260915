"""Flight-path metrics."""

import numpy as np


def path_lengths(trajectory: np.ndarray) -> np.ndarray:
    """Return one accumulated path length per agent for shape ``(T,N,3)``."""
    if trajectory.ndim != 3 or trajectory.shape[-1] != 3:
        raise ValueError("trajectory must have shape (time, agents, 3)")
    return np.linalg.norm(np.diff(trajectory, axis=0), axis=-1).sum(axis=0)

