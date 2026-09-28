"""Learning-curve plotting."""

from __future__ import annotations

import json
from pathlib import Path
from time import sleep

import numpy as np


def plot_learning_curve(history: list[dict[str, float]], key: str = "episode_reward", ax=None):
    """Plot one recorded metric and return the Matplotlib axes."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install uav-safe-marl[plot] to plot learning curves") from exc
    if ax is None:
        _, ax = plt.subplots()
    ax.plot([row["episode"] for row in history], [row[key] for row in history])
    ax.set(xlabel="episode", ylabel=key)
    return ax


def plot_learning_dashboard(history, figure=None):
    """Plot the five Primary Metrics plus CMDP cost for live diagnosis."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install uav-safe-marl[plot] to plot learning curves") from exc
    if figure is None:
        figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    else:
        figure.clear(); axes = figure.subplots(2, 3)
    panels = [
        ("episode_success", "Episode Success"),
        ("separation_violation_rate", "Separation Violation"),
        ("path_stretch", "Path Stretch (successful)"),
        ("pure_safety_intervention_magnitude", "Pure Safety Intervention"),
        ("communication_load", "Communication Load"),
        ("discounted_episode_cost", "Discounted Cost"),
    ]
    episodes = [row["episode"] for row in history]
    for axis, (key, title) in zip(axes.flat, panels, strict=True):
        values = [np.nan if row.get(key) is None else row.get(key, np.nan) for row in history]
        axis.plot(episodes, values, alpha=0.35)
        if len(values) >= 10:
            kernel = np.ones(10) / 10
            axis.plot(episodes[9:], np.convolve(np.asarray(values, dtype=float), kernel, mode="valid"), linewidth=2)
        if key == "discounted_episode_cost" and history:
            axis.axhline(history[-1].get("configured_d_cost", np.nan), color="red", linestyle="--", label="d_cost")
            axis.legend()
        axis.set(title=title, xlabel="episode", ylabel=key)
    figure.tight_layout()
    return figure


def watch_learning_metrics(path: str | Path, refresh_seconds: float = 1.0) -> None:
    """Continuously refresh a local dashboard from the canonical JSONL log."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install uav-safe-marl[plot] to watch learning") from exc
    source = Path(path)
    plt.ion()
    figure = plt.figure(figsize=(14, 8))
    while plt.fignum_exists(figure.number):
        if source.exists():
            history = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
            if history:
                plot_learning_dashboard(history, figure)
        plt.pause(0.05)
        sleep(max(0.0, refresh_seconds - 0.05))
