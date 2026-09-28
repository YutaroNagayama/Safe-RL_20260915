"""Persist scenario banks and raw evaluation rollouts without losing recomputability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from uav_safe_marl.runners.evaluator import evaluate
from .scenarios import interaction_descriptor, predicted_interaction_data


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _exact_scenario(value: dict[str, Any]) -> dict[str, Any]:
    """Resolve legacy/manual inputs into a replayable exact-scenario schema."""
    scenario = dict(value)
    positions = np.asarray(scenario["positions"], dtype=np.float64)
    goals = np.asarray(scenario["goals"], dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1:] != (3,) or goals.shape != positions.shape:
        raise ValueError("scenario positions/goals must have shape (N, 3)")
    count = len(positions)
    velocities = np.asarray(scenario.get("velocities", np.zeros_like(positions)), dtype=np.float64)
    if velocities.shape != positions.shape:
        raise ValueError("scenario velocities must have shape (N, 3)")

    def values(name: str, fallback: float) -> np.ndarray:
        result = np.asarray(scenario.get(name, fallback), dtype=np.float64)
        if result.ndim == 0:
            result = np.full(count, float(result))
        if result.shape != (count,) or np.any(result <= 0.0):
            raise ValueError(f"{name} must be positive with shape (N,)")
        return result

    thresholds = dict(scenario.get("distance_thresholds", {"d_safe": 10.0, "d_warn": 30.0, "d_eng": 50.0}))
    if set(thresholds) != {"d_safe", "d_warn", "d_eng"} or not 0 < float(thresholds["d_safe"]) < float(thresholds["d_warn"]) <= float(thresholds["d_eng"]):
        raise ValueError("invalid distance_thresholds")
    los_threshold = float(scenario.get("los_prediction_threshold", thresholds["d_eng"]))
    horizon = float(scenario.get("prediction_horizon", 50.0))
    maximum_accelerations = values("maximum_accelerations", 3.0)
    maximum_velocities = values("maximum_velocities", 12.0)
    computed_edges, computed_pairs = predicted_interaction_data(positions, velocities, goals, los_threshold, horizon)
    scenario.update({
        "actual_agent_count": count,
        "agent_count": count,
        "positions": positions,
        "velocities": velocities,
        "goals": goals,
        "maximum_accelerations": maximum_accelerations,
        "maximum_velocities": maximum_velocities,
        "reference_route_lengths": np.asarray(scenario.get("reference_route_lengths", np.linalg.norm(goals - positions, axis=1)), dtype=np.float64),
        "reference_routes": scenario.get("reference_routes", np.stack((positions, goals), axis=1)),
        "distance_thresholds": {key: float(thresholds[key]) for key in ("d_safe", "d_warn", "d_eng")},
        "los_prediction_threshold": los_threshold,
        "prediction_horizon": horizon,
        "generator_version": scenario.get("generator_version", "manual_v1"),
        "scenario_seed": scenario.get("scenario_seed"),
    })
    scenario.setdefault("predicted_los_edges", computed_edges)
    scenario.setdefault("predicted_interaction_edges", scenario["predicted_los_edges"])
    if "predicted_pair_data" not in scenario:
        supplied = {tuple(sorted(map(int, edge))) for edge in scenario["predicted_interaction_edges"]}
        for pair in computed_pairs:
            pair["candidate"] = (int(pair["first"]), int(pair["second"])) in supplied
        scenario["predicted_pair_data"] = computed_pairs
    if "scenario_descriptor" not in scenario:
        dimensions = np.asarray(scenario.get("spatial_dimensions", [1.0, 1.0, 1.0]), dtype=np.float64)
        volume = float(scenario.get("spatial_volume", np.prod(dimensions)))
        descriptor = interaction_descriptor(
            count, scenario["predicted_pair_data"], max(volume, 1e-12), maximum_velocities, maximum_accelerations,
            float(thresholds["d_safe"]), float(thresholds["d_warn"]), float(thresholds["d_eng"]),
        )
        descriptor.update({
            "spatial_scaling_mode": scenario.get("spatial_scaling_mode", "manual"),
            "spatial_dimensions": dimensions.tolist(),
            "spatial_volume": volume,
            "spatial_volume_factor": float(scenario.get("spatial_volume_factor", 1.0)),
            "initial_cbf_edge_count": sum(
                float(pair["initial_distance"]) <= float(thresholds["d_eng"])
                for pair in scenario["predicted_pair_data"]
            ),
        })
        scenario["scenario_descriptor"] = descriptor
    if np.asarray(scenario["reference_route_lengths"]).shape != (count,):
        raise ValueError("reference_route_lengths must have shape (N,)")
    return scenario


def save_scenario_bank(path: str | Path, scenarios: Iterable[dict[str, Any]]) -> Path:
    """Save exact generated scenarios so model comparisons never rely on RNG alone."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": 3, "scenarios": [_jsonable(_exact_scenario(item)) for item in scenarios]}
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def load_scenario_bank(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format_version") not in {1, 2, 3} or not isinstance(payload.get("scenarios"), list):
        raise ValueError("unsupported scenario-bank format")
    identifiers = [item.get("scenario_id") for item in payload["scenarios"]]
    if any(identifier is None for identifier in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("scenario_id values must be present and unique")
    return payload["scenarios"]


def _aggregate(episodes: list[dict[str, Any]], dt: float) -> dict[str, float | None]:
    successful = [item["path_stretch"] for item in episodes if item["path_stretch"] is not None]
    active_ticks = sum(item["active_agent_ticks"] for item in episodes)
    return {
        "episode_success_rate": float(np.mean([item["episode_success"] for item in episodes])),
        "separation_violation_rate": sum(item["separation_violation_pair_ticks"] for item in episodes) / max(1, sum(item["active_pair_ticks"] for item in episodes)),
        "path_stretch": float(np.mean(successful)) if successful else None,
        "pure_safety_intervention_magnitude": sum(item["pure_intervention_squared_sum"] for item in episodes) / max(1, active_ticks),
        "communication_load": sum(item["transmitted_bytes"] for item in episodes) / max(dt, active_ticks * dt),
    }


def evaluate_scenario_bank(trainer: Any, scenarios: Iterable[dict[str, Any]], method: str = "model", training_seed: int | None = None) -> dict[str, Any]:
    """Run one deterministic paired rollout for every exact scenario."""
    combined = {"episode_metrics": [], "rollouts": [], "trajectories": []}
    rewards: list[float] = []
    costs: list[float] = []
    for scenario in scenarios:
        result = evaluate(trainer, episodes=1, reset_options={"scenario": scenario})
        for metrics in result["episode_metrics"]:
            metrics["method"] = method
            metrics["training_seed"] = trainer.config.seed if training_seed is None else training_seed
        combined["episode_metrics"].extend(result["episode_metrics"])
        combined["rollouts"].extend(result["rollouts"])
        combined["trajectories"].extend(result["trajectories"])
        rewards.append(result["mean_reward"])
        costs.append(result["mean_cost"])
    combined["mean_reward"] = float(np.mean(rewards))
    combined["mean_cost"] = float(np.mean(costs))
    return combine_evaluation_results([combined], trainer.config.environment.dt_base)


def combine_evaluation_results(results: Iterable[dict[str, Any]], dt: float) -> dict[str, Any]:
    """Merge serial or process-parallel shards in their supplied order."""
    merged = {"episode_metrics": [], "rollouts": [], "trajectories": []}
    rewards: list[float] = []
    costs: list[float] = []
    for result in results:
        merged["episode_metrics"].extend(result["episode_metrics"])
        merged["rollouts"].extend(result["rollouts"])
        merged["trajectories"].extend(result["trajectories"])
        if "mean_reward" in result:
            rewards.extend([float(result["mean_reward"])] * len(result["episode_metrics"]))
            costs.extend([float(result["mean_cost"])] * len(result["episode_metrics"]))
    if not merged["episode_metrics"]:
        raise ValueError("scenario bank cannot be empty")
    merged["primary_metrics"] = _aggregate(merged["episode_metrics"], dt)
    counts = sorted({int(row["actual_agent_count"]) for row in merged["episode_metrics"]})
    merged["primary_metrics_by_n"] = {
        str(count): _aggregate([row for row in merged["episode_metrics"] if int(row["actual_agent_count"]) == count], dt)
        for count in counts
    }
    merged["primary_metrics_macro_by_n"] = {
        metric: float(np.mean([values[metric] for values in merged["primary_metrics_by_n"].values() if values[metric] is not None]))
        if any(values[metric] is not None for values in merged["primary_metrics_by_n"].values()) else None
        for metric in merged["primary_metrics"]
    }
    # Every shard returned by evaluate contains one mean over its episodes.  The
    # parallel implementation uses equal one-scenario shards, while the serial
    # path supplies exact per-scenario rewards above.
    merged["mean_reward"] = float(np.mean(rewards)) if rewards else 0.0
    merged["mean_cost"] = float(np.mean(costs)) if costs else 0.0
    return merged


def save_evaluation(result: dict[str, Any], output_dir: str | Path) -> Path:
    """Save summaries as JSON and each full rollout as a compressed NPZ."""
    root = Path(output_dir)
    rollout_dir = root / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "primary_metrics": result["primary_metrics"],
        "primary_metrics_by_n": result.get("primary_metrics_by_n", {}),
        "primary_metrics_macro_by_n": result.get("primary_metrics_macro_by_n", {}),
        "mean_reward": result["mean_reward"],
        "mean_cost": result["mean_cost"],
        "episode_count": len(result["episode_metrics"]),
    }
    (root / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    with (root / "episode_metrics.jsonl").open("w", encoding="utf-8") as stream:
        for metrics in result["episode_metrics"]:
            stream.write(json.dumps(_jsonable(metrics), ensure_ascii=False) + "\n")
    manifest = []
    for index, rollout in enumerate(result["rollouts"]):
        scenario_id = rollout["reset_info"].get("scenario_id") or f"episode_{index:04d}"
        filename = f"rollout_{index:04d}.npz"
        path = rollout_dir / filename
        np.savez_compressed(
            path,
            positions=rollout["positions"],
            velocities=rollout["velocities"],
            active_masks=rollout["active_masks"],
            reference_actions=rollout["reference_actions"],
            normalized_residual_actions=rollout["normalized_residual_actions"],
            residual_actions=rollout["residual_actions"],
            command_actions=rollout["command_actions"],
            nominal_actions=rollout["nominal_actions"],
            safe_actions=rollout["safe_actions"],
            intervention_actions=rollout["intervention_actions"],
            pure_cbf_actions=rollout["pure_cbf_actions"],
            newly_reached_masks=rollout["newly_reached_masks"],
            pairwise_distances=rollout["pairwise_distances"],
            actor_adjacencies=rollout["actor_adjacencies"],
            cbf_adjacencies=rollout["cbf_adjacencies"],
            information_ages=rollout["information_ages"],
            transmitted_byte_deltas=rollout["transmitted_byte_deltas"],
            rewards=rollout["rewards"],
            costs=rollout["costs"],
            reward_goal=np.asarray([item["goal_reward"] for item in rollout["reward_components"]]),
            reward_progress=np.asarray([item["progress"] for item in rollout["reward_components"]]),
            reward_time_penalty=np.asarray([item["time_penalty"] for item in rollout["reward_components"]]),
            reward_absolute_distance_penalty=np.asarray([item["absolute_distance_penalty"] for item in rollout["reward_components"]]),
            reward_residual_penalty=np.asarray([item["residual_penalty"] for item in rollout["reward_components"]]),
            mean_normalized_goal_distance=np.asarray([item["mean_normalized_goal_distance"] for item in rollout["reward_components"]]),
            goal_distances=rollout["goal_distances"],
            distance_cost=np.asarray([item["distance_cost"] for item in rollout["cost_components"]]),
            intervention_cost=np.asarray([item["intervention_cost"] for item in rollout["cost_components"]]),
            reached_step=rollout["reached_step"],
            initial_positions=rollout["reset_info"]["initial_positions"],
            initial_velocities=rollout["reset_info"]["initial_velocities"],
            goals=rollout["reset_info"]["goals"],
            reference_route_lengths=rollout["reset_info"]["reference_route_lengths"],
            maximum_accelerations=rollout["maximum_accelerations"],
            maximum_velocities=rollout["maximum_velocities"],
            actual_agent_count=np.asarray(rollout["reset_info"]["actual_agent_count"]),
            d_safe=np.asarray(rollout["distance_thresholds"]["d_safe"]),
            d_warn=np.asarray(rollout["distance_thresholds"]["d_warn"]),
            d_eng=np.asarray(rollout["distance_thresholds"]["d_eng"]),
            candidate_adjacency=rollout["candidate_adjacency"],
            actual_minimum_pair_distances=rollout["actual_minimum_pair_distances"],
            cbf_ever_adjacency=rollout["cbf_ever_adjacency"],
            candidate_miss_adjacency=rollout["candidate_miss_adjacency"],
            candidate_edge_lifetime_ticks=rollout["candidate_edge_lifetime_ticks"],
        )
        manifest.append({"scenario_id": scenario_id, "rollout": f"rollouts/{filename}"})
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return root
