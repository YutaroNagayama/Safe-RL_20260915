"""Fixed exit-waypoint guidance shared by rollout and differentiable updates."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from uav_safe_marl.core.types import AgentState
from uav_safe_marl.graph.feature_layout import NODE_FEATURES


def reference_action_numpy(
    states: Sequence[AgentState],
    speed_fraction: float,
    velocity_gain: float,
    active_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return component-wise bounded acceleration toward each exit waypoint."""
    active = np.ones(len(states), dtype=bool) if active_mask is None else np.asarray(active_mask, dtype=bool)
    result = np.zeros((len(states), 3), dtype=np.float64)
    for index, state in enumerate(states):
        if not active[index]:
            continue
        displacement = np.asarray(state.goal - state.position, dtype=np.float64)
        distance = float(np.linalg.norm(displacement))
        if distance <= 1e-12:
            continue
        speed_limit = float(np.min(state.velocity_limit))
        target_velocity = speed_fraction * speed_limit * displacement / distance
        raw = velocity_gain * (target_velocity - state.velocity)
        result[index] = np.clip(raw, -state.acceleration_limit, state.acceleration_limit)
    return result


def reference_action_torch(node_features, speed_fraction: float, velocity_gain: float):
    """Torch equivalent of :func:`reference_action_numpy` for policy updates.

    Node layout is ``position, velocity, goal-position, a_max, v_max, u_ref/a_max``.
    The explicit normalized reference is retained in replay for diagnosis, while
    this function reconstructs the fixed controller from collection-time state.
    """
    velocity = node_features[..., NODE_FEATURES.velocity]
    displacement = node_features[..., NODE_FEATURES.goal_displacement]
    acceleration_limit = node_features[..., NODE_FEATURES.acceleration_limit].clamp_min(1e-12)
    speed_limit = node_features[..., NODE_FEATURES.velocity_limit].clamp_min(0.0)
    distance = displacement.norm(dim=-1, keepdim=True)
    direction = displacement / distance.clamp_min(1e-12)
    target_velocity = speed_fraction * speed_limit * direction
    raw = velocity_gain * (target_velocity - velocity)
    reference = raw.clamp(min=-acceleration_limit, max=acceleration_limit)
    return reference * (distance > 1e-12).to(reference.dtype)


def compose_action_numpy(
    normalized_action: np.ndarray,
    reference_action: np.ndarray,
    acceleration_limits: np.ndarray,
    actor_mode: str,
    residual_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return physical residual, raw command, and acceleration-bounded nominal."""
    normalized = np.clip(np.asarray(normalized_action, dtype=np.float64), -1.0, 1.0)
    limits = np.asarray(acceleration_limits, dtype=np.float64)
    reference = np.asarray(reference_action, dtype=np.float64)
    if actor_mode == "residual":
        residual = residual_fraction * limits * normalized
        command = reference + residual
    elif actor_mode == "reference_only":
        residual = np.zeros_like(normalized)
        command = reference.copy()
    elif actor_mode == "full":
        residual = limits * normalized
        command = residual.copy()
    else:
        raise ValueError(f"unsupported actor_mode: {actor_mode}")
    nominal = np.clip(command, -limits, limits)
    return residual, command, nominal


def compose_action_torch(
    normalized_action,
    node_features,
    actor_mode: str,
    residual_fraction: float,
    speed_fraction: float,
    velocity_gain: float,
):
    """Differentiably reconstruct the nominal action seen by centralized critics."""
    limits = node_features[..., NODE_FEATURES.acceleration_limit].clamp_min(1e-12)
    reference = reference_action_torch(node_features, speed_fraction, velocity_gain)
    if actor_mode == "residual":
        residual = residual_fraction * limits * normalized_action
        command = reference + residual
    elif actor_mode == "reference_only":
        residual = normalized_action * 0.0
        command = reference
    elif actor_mode == "full":
        residual = limits * normalized_action
        command = residual
    else:
        raise ValueError(f"unsupported actor_mode: {actor_mode}")
    return command.clamp(min=-limits, max=limits)
