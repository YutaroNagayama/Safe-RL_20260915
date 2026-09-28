"""3D trajectory plotting."""

from __future__ import annotations

import numpy as np


def plot_trajectories(trajectory: np.ndarray, ax=None):
    """Plot a ``(time, agents, 3)`` trajectory and return the Matplotlib axes."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install uav-safe-marl[plot] to plot trajectories") from exc
    if ax is None:
        ax = plt.figure().add_subplot(projection="3d")
    for agent in range(trajectory.shape[1]):
        ax.plot(*trajectory[:, agent, :].T, label=f"UAV {agent}")
    ax.set(xlabel="x", ylabel="y", zlabel="z"); ax.legend()
    return ax

