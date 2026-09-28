"""Seeded scenario-bank generation for paired method comparisons."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def _time_aligned_cpa(
    first_position: np.ndarray,
    first_goal: np.ndarray,
    first_speed: float,
    second_position: np.ndarray,
    second_goal: np.ndarray,
    second_speed: float,
    prediction_horizon: float,
) -> tuple[float, float]:
    """Return CPA distance/time while both reference trajectories are active."""
    first_route = first_goal - first_position
    second_route = second_goal - second_position
    first_length = float(np.linalg.norm(first_route))
    second_length = float(np.linalg.norm(second_route))
    first_velocity = np.zeros(3) if first_speed <= 1e-12 else first_route / first_length * first_speed
    second_velocity = np.zeros(3) if second_speed <= 1e-12 else second_route / second_length * second_speed
    first_duration = prediction_horizon if first_speed <= 1e-12 else first_length / first_speed
    second_duration = prediction_horizon if second_speed <= 1e-12 else second_length / second_speed
    horizon = max(0.0, min(prediction_horizon, first_duration, second_duration))
    relative_position = first_position - second_position
    relative_velocity = first_velocity - second_velocity
    denominator = float(relative_velocity @ relative_velocity)
    cpa_time = 0.0 if denominator <= 1e-12 else float(np.clip(-(relative_position @ relative_velocity) / denominator, 0.0, horizon))
    cpa_distance = float(np.linalg.norm(relative_position + cpa_time * relative_velocity))
    return cpa_distance, cpa_time


def predicted_interaction_data(
    positions: np.ndarray,
    velocities: np.ndarray,
    goals: np.ndarray,
    threshold: float,
    horizon: float,
) -> tuple[list[list[int]], list[dict[str, float | int | bool]]]:
    """Compute time-aligned pair CPA data and the preflight screening graph."""
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    goals = np.asarray(goals, dtype=np.float64)
    speeds = np.linalg.norm(velocities, axis=1)
    edges: list[list[int]] = []
    pairs: list[dict[str, float | int | bool]] = []
    for first in range(len(positions)):
        for second in range(first + 1, len(positions)):
            distance, time = _time_aligned_cpa(
                positions[first], goals[first], float(speeds[first]),
                positions[second], goals[second], float(speeds[second]), horizon,
            )
            candidate = distance <= threshold
            if candidate:
                edges.append([first, second])
            pairs.append({
                "first": first,
                "second": second,
                "initial_distance": float(np.linalg.norm(positions[first] - positions[second])),
                "relative_initial_speed": float(np.linalg.norm(velocities[first] - velocities[second])),
                "predicted_cpa_distance": distance,
                "predicted_cpa_time": time,
                "candidate": candidate,
            })
    return edges, pairs


def _predicted_edges(positions: np.ndarray, velocities: np.ndarray, goals: np.ndarray, threshold: float, horizon: float) -> list[list[int]]:
    """Compatibility wrapper returning only candidate pairs."""
    return predicted_interaction_data(positions, velocities, goals, threshold, horizon)[0]


def interaction_descriptor(
    count: int,
    pair_data: Sequence[dict[str, float | int | bool]],
    volume: float,
    capability_velocity: np.ndarray,
    capability_acceleration: np.ndarray,
    d_safe: float,
    d_warn: float,
    d_eng: float,
) -> dict:
    """Describe a generated graph without constraining its complexity."""
    candidates = [pair for pair in pair_data if bool(pair["candidate"])]
    edge_count = len(candidates)
    possible = count * (count - 1) // 2
    degrees = np.zeros(count, dtype=np.int64)
    for pair in candidates:
        degrees[int(pair["first"])] += 1
        degrees[int(pair["second"])] += 1
    minimum_pair = min(pair_data, key=lambda pair: float(pair["predicted_cpa_distance"]), default=None)
    earliest = min((float(pair["predicted_cpa_time"]) for pair in candidates), default=None)

    def spread(values: np.ndarray) -> dict[str, float]:
        mean = float(np.mean(values))
        return {
            "minimum": float(np.min(values)), "maximum": float(np.max(values)), "mean": mean,
            "standard_deviation": float(np.std(values)),
            "coefficient_of_variation": float(np.std(values) / mean) if mean > 0.0 else 0.0,
        }

    def candidate_fraction(limit: float, strict: bool = False) -> float:
        if not candidates:
            return 0.0
        selected = sum(
            float(pair["predicted_cpa_distance"]) < limit if strict
            else float(pair["predicted_cpa_distance"]) <= limit
            for pair in candidates
        )
        return float(selected / len(candidates))

    return {
        "traffic_density": float(count / volume),
        "predicted_edge_count": edge_count,
        "mean_degree": float(2 * edge_count / count),
        "max_degree": int(degrees.max(initial=0)),
        "edge_ratio": float(edge_count / possible) if possible else 0.0,
        "all_to_all": bool(possible > 0 and edge_count == possible),
        "candidate_empty": edge_count == 0,
        "minimum_predicted_cpa": None if minimum_pair is None else float(minimum_pair["predicted_cpa_distance"]),
        "minimum_cpa_pair": None if minimum_pair is None else [int(minimum_pair["first"]), int(minimum_pair["second"])],
        "minimum_cpa_time": None if minimum_pair is None else float(minimum_pair["predicted_cpa_time"]),
        "earliest_candidate_cpa_time": earliest,
        "descriptor_d_safe": float(d_safe),
        "descriptor_d_warn": float(d_warn),
        "descriptor_d_eng": float(d_eng),
        "candidate_fraction_cpa_le_d_eng": candidate_fraction(d_eng),
        "candidate_fraction_cpa_le_d_warn": candidate_fraction(d_warn),
        "candidate_fraction_cpa_lt_d_safe": candidate_fraction(d_safe, strict=True),
        "has_candidate_cpa_le_d_eng": any(float(pair["predicted_cpa_distance"]) <= d_eng for pair in candidates),
        "has_candidate_cpa_le_d_warn": any(float(pair["predicted_cpa_distance"]) <= d_warn for pair in candidates),
        "has_candidate_cpa_lt_d_safe": any(float(pair["predicted_cpa_distance"]) < d_safe for pair in candidates),
        "velocity_capability": spread(np.asarray(capability_velocity, dtype=np.float64)),
        "acceleration_capability": spread(np.asarray(capability_acceleration, dtype=np.float64)),
    }


def _face_point(rng: np.random.Generator, face: str, longitudinal_length: float, transverse_width: float, altitude_range: tuple[float, float]) -> np.ndarray:
    """Sample one point on an entry/exit face of an imaginary corridor."""
    x = -longitudinal_length / 2.0 if face == "x_min" else longitudinal_length / 2.0
    return np.array([x, rng.uniform(-transverse_width / 2.0, transverse_width / 2.0), rng.uniform(*altitude_range)])


def _boundary_crossing_routes(
    rng: np.random.Generator,
    count: int,
    longitudinal_length: float,
    transverse_width: float,
    altitude_range: tuple[float, float],
    minimum_distance: float,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Generate separated entry points and opposite-face exit waypoints."""
    starts: list[np.ndarray] = []
    goals: list[np.ndarray] = []
    entry_faces: list[str] = []
    exit_faces: list[str] = []
    for _ in range(count):
        for _ in range(100_000):
            entry = "x_min" if rng.random() < 0.5 else "x_max"
            exit_face = "x_max" if entry == "x_min" else "x_min"
            start = _face_point(rng, entry, longitudinal_length, transverse_width, altitude_range)
            goal = _face_point(rng, exit_face, longitudinal_length, transverse_width, altitude_range)
            if all(np.linalg.norm(start - other) >= minimum_distance for other in starts) and all(np.linalg.norm(goal - other) >= minimum_distance for other in goals):
                starts.append(start); goals.append(goal); entry_faces.append(entry); exit_faces.append(exit_face)
                break
        else:
            raise RuntimeError("could not sample separated boundary routes; increase spatial size")
    return np.vstack(starts), np.vstack(goals), entry_faces, exit_faces


def generate_random_traffic_bank(
    agent_count: int | Sequence[int],
    scenario_count: int,
    seed: int,
    *,
    base_size: float = 280.0,
    transverse_size: float = 280.0,
    spatial_scaling_mode: str = "fixed",
    spatial_volume_scale_range: tuple[float, float] = (0.7, 1.3),
    altitude_range: tuple[float, float] = (-28.0, 28.0),
    minimum_initial_distance: float = 20.0,
    d_safe: float = 10.0,
    d_warn: float = 30.0,
    d_eng: float = 50.0,
    los_prediction_threshold: float = 75.0,
    prediction_horizon: float = 50.0,
    initial_speed_range: tuple[float, float] | None = None,
    initial_speed_fraction_range: tuple[float, float] = (0.4, 0.65),
    acceleration_range: tuple[float, float] = (3.0, 3.0),
    velocity_limit_range: tuple[float, float] = (12.0, 12.0),
    require_predicted_conflict: bool = True,
) -> list[dict]:
    """Generate exact boundary-crossing scenarios with natural graph complexity."""
    counts = [int(agent_count)] if isinstance(agent_count, (int, np.integer)) else [int(value) for value in agent_count]
    if not counts or min(counts) < 2 or scenario_count < 1:
        raise ValueError("agent_count choices must be >=2 and scenario_count positive")
    if not 0 < d_safe < d_warn <= d_eng < los_prediction_threshold:
        raise ValueError("require 0 < d_safe < d_warn <= d_eng < los_prediction_threshold")
    if spatial_scaling_mode not in {"fixed", "density_controlled"}:
        raise ValueError("spatial_scaling_mode must be fixed or density_controlled")
    if base_size <= 0.0 or transverse_size <= 0.0 or minimum_initial_distance <= 0.0:
        raise ValueError("spatial dimensions and minimum distance must be positive")
    if altitude_range[1] <= altitude_range[0]:
        raise ValueError("altitude_range must be increasing")
    if spatial_volume_scale_range[0] <= 0.0 or spatial_volume_scale_range[1] < spatial_volume_scale_range[0]:
        raise ValueError("invalid spatial_volume_scale_range")
    for name, bounds in (("initial_speed_fraction_range", initial_speed_fraction_range), ("acceleration_range", acceleration_range), ("velocity_limit_range", velocity_limit_range)):
        if bounds[0] < 0 or bounds[1] < bounds[0] or (name != "initial_speed_fraction_range" and bounds[0] <= 0):
            raise ValueError(f"invalid {name}")
    if initial_speed_fraction_range[1] > 1.0:
        raise ValueError("initial speed fractions cannot exceed one")
    if initial_speed_range is not None:
        if initial_speed_range[0] < 0 or initial_speed_range[1] < initial_speed_range[0]:
            raise ValueError("invalid initial_speed_range")
        if initial_speed_range[1] > velocity_limit_range[0]:
            raise ValueError("initial speeds must not exceed the minimum generated velocity limit")

    root = np.random.SeedSequence(seed)
    scenarios = []
    for index, child in enumerate(root.spawn(scenario_count)):
        rng = np.random.default_rng(child)
        # N is sampled once and held fixed across candidate-zero rejection.
        count = int(rng.choice(counts))
        rejected_without_candidate = 0
        for attempt in range(1, 1_001):
            if spatial_scaling_mode == "fixed":
                volume_factor = 1.0
                transverse_width = transverse_size
                scaled_altitude = altitude_range
            else:
                volume_factor = float(rng.uniform(*spatial_volume_scale_range))
                cross_scale = math.sqrt((count / 4.0) * volume_factor)
                transverse_width = transverse_size * cross_scale
                altitude_center = 0.5 * (altitude_range[0] + altitude_range[1])
                altitude_half_span = 0.5 * (altitude_range[1] - altitude_range[0]) * cross_scale
                scaled_altitude = (altitude_center - altitude_half_span, altitude_center + altitude_half_span)
            positions, goals, entry_faces, exit_faces = _boundary_crossing_routes(rng, count, base_size, transverse_width, scaled_altitude, minimum_initial_distance)
            route_lengths = np.linalg.norm(goals - positions, axis=1)
            directions = (goals - positions) / np.maximum(route_lengths[:, None], 1e-12)
            accelerations = rng.uniform(*acceleration_range, size=count)
            velocity_limits = rng.uniform(*velocity_limit_range, size=count)
            if initial_speed_range is None:
                speed_fractions = rng.uniform(*initial_speed_fraction_range, size=count)
                speeds = speed_fractions * velocity_limits
            else:
                speeds = rng.uniform(*initial_speed_range, size=count)
                speed_fractions = speeds / velocity_limits
            velocities = directions * speeds[:, None]
            edges, pair_data = predicted_interaction_data(positions, velocities, goals, los_prediction_threshold, prediction_horizon)
            if require_predicted_conflict and not edges:
                rejected_without_candidate += 1
                continue
            break
        else:
            raise RuntimeError("could not generate a scenario satisfying the candidate requirement")

        spatial_dimensions = [base_size, transverse_width, scaled_altitude[1] - scaled_altitude[0]]
        volume = float(np.prod(spatial_dimensions))
        descriptor = interaction_descriptor(
            count, pair_data, volume, velocity_limits, accelerations, d_safe, d_warn, d_eng
        )
        descriptor.update({
            "spatial_scaling_mode": spatial_scaling_mode,
            "spatial_dimensions": spatial_dimensions,
            "spatial_volume": volume,
            "spatial_volume_factor": volume_factor,
            "generation_attempts": attempt,
            "candidate_zero_rejections": rejected_without_candidate,
            "initial_cbf_edge_count": sum(float(pair["initial_distance"]) <= d_eng for pair in pair_data),
        })
        scenarios.append({
            "scenario_id": f"interaction_traffic_n{count}_root{seed}_{index:04d}",
            "generator": "boundary_interaction_traffic_v6", "generator_version": 6,
            "geometry": "boundary_crossing_corridor",
            "spatial_scaling_mode": spatial_scaling_mode,
            "scenario_seed_entropy": child.entropy, "scenario_seed_spawn_key": list(child.spawn_key),
            "generation_attempts": attempt, "candidate_zero_rejections": rejected_without_candidate,
            "agent_count": count, "actual_agent_count": count,
            "spatial_side_length": base_size, "spatial_dimensions": spatial_dimensions,
            "spatial_volume": volume, "spatial_volume_factor": volume_factor,
            "positions": positions.tolist(), "velocities": velocities.tolist(), "goals": goals.tolist(),
            "entry_faces": entry_faces, "exit_faces": exit_faces,
            "maximum_accelerations": accelerations.tolist(), "maximum_velocities": velocity_limits.tolist(),
            "initial_speed_fractions": speed_fractions.tolist(),
            "reference_route_lengths": route_lengths.tolist(),
            "reference_routes": np.stack((positions, goals), axis=1).tolist(),
            "reference_arrival_times": (route_lengths / speeds).tolist(),
            "predicted_los_edges": edges, "predicted_interaction_edges": edges,
            "predicted_pair_data": pair_data, "scenario_descriptor": descriptor,
            "distance_thresholds": {"d_safe": d_safe, "d_warn": d_warn, "d_eng": d_eng},
            "los_prediction_threshold": los_prediction_threshold,
            "interaction_prediction_threshold": los_prediction_threshold,
            "prediction_horizon": prediction_horizon,
            "altitude_range": list(scaled_altitude),
            "minimum_initial_separation": minimum_initial_distance,
        })
    return scenarios
