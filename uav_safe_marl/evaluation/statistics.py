"""Seed-clustered statistics for the five fixed Primary Metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np


def _seed_metrics(records: list[dict[str, Any]]) -> dict[str, float | None]:
    active_ticks = sum(row["active_agent_ticks"] for row in records)
    successful = [row["path_stretch"] for row in records if row["path_stretch"] is not None]
    return {
        "episode_success_rate": float(np.mean([row["episode_success"] for row in records])),
        "separation_violation_rate": sum(row["separation_violation_pair_ticks"] for row in records) / max(1, sum(row["active_pair_ticks"] for row in records)),
        "path_stretch": float(np.mean(successful)) if successful else None,
        "pure_safety_intervention_magnitude": sum(row["pure_intervention_squared_sum"] for row in records) / max(1, active_ticks),
        "communication_load": sum(row["transmitted_bytes"] for row in records) / max(1e-12, sum(row["active_agent_seconds"] for row in records)),
    }


def aggregate_primary_by_method(records: Iterable[dict[str, Any]], bootstrap_samples: int = 2000, seed: int = 0) -> dict[str, dict[str, dict[str, float | int | None]]]:
    """Return mean, median, and seed-clustered bootstrap CI by method."""
    rows_all = list(records)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows_all:
        grouped[(str(row["method"]), int(row["training_seed"]))].append(row)
    by_method: dict[str, list[tuple[int, dict[str, float | None]]]] = defaultdict(list)
    for (method, training_seed), rows in grouped.items():
        by_method[method].append((training_seed, _seed_metrics(rows)))
    rng = np.random.default_rng(seed)
    output: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for method, seed_items in by_method.items():
        output[method] = {}
        for metric in seed_items[0][1]:
            values = np.asarray([row[metric] for _, row in seed_items if row[metric] is not None], dtype=np.float64)
            sample_count = int(sum(1 for row in rows_all if str(row["method"]) == method and (metric != "path_stretch" or row.get("path_stretch") is not None)))
            seed_values = {str(training_seed): None if row[metric] is None else float(row[metric]) for training_seed, row in seed_items}
            if not len(values):
                output[method][metric] = {"mean": None, "median": None, "ci95_low": None, "ci95_high": None, "training_seed_count": 0, "sample_count": sample_count, "by_training_seed": seed_values}
                continue
            indices = rng.integers(0, len(values), size=(bootstrap_samples, len(values)))
            bootstrap_means = values[indices].mean(axis=1)
            output[method][metric] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "ci95_low": float(np.quantile(bootstrap_means, 0.025)),
                "ci95_high": float(np.quantile(bootstrap_means, 0.975)),
                "training_seed_count": int(len(values)),
                "sample_count": sample_count,
                "by_training_seed": seed_values,
            }
    return output


def aggregate_primary_by_method_and_n(records: Iterable[dict[str, Any]], bootstrap_samples: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Produce overall, conditional-N, and equal-N macro summaries."""
    rows = list(records)
    overall = aggregate_primary_by_method(rows, bootstrap_samples, seed)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["actual_agent_count"])].append(row)
    by_n: dict[str, Any] = {
        str(n): aggregate_primary_by_method(group, bootstrap_samples, seed + n)
        for n, group in sorted(grouped.items())
    }
    macro: dict[str, dict[str, float | None]] = {}
    for method in overall:
        metric_names = overall[method]
        macro[method] = {}
        for metric in metric_names:
            values = [summary[method][metric]["mean"] for summary in by_n.values() if method in summary and summary[method][metric]["mean"] is not None]
            macro[method][metric] = float(np.mean(values)) if values else None
    return {"overall": overall, "by_n": by_n, "macro_by_n": macro}
