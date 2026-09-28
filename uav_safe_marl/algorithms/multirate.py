"""Utilities that keep base-tick Bellman updates consistent with action hold."""

from __future__ import annotations

from contextlib import contextmanager

from uav_safe_marl.control import compose_action_torch
from uav_safe_marl.graph.feature_layout import NODE_FEATURES
from uav_safe_marl.policies.base import require_torch

torch = require_torch()


def held_nominal_action(state, held_normalized, config, graph_mode: bool):
    nodes = state["node_features"] if graph_mode else state
    action = compose_action_torch(
        held_normalized, nodes, config.control.actor_mode,
        config.control.residual_acceleration_fraction,
        config.control.reference_speed_fraction,
        config.control.reference_velocity_gain,
    )
    mask = state["node_mask"].unsqueeze(-1) if graph_mode else nodes[..., NODE_FEATURES.active_mask]
    action = action * mask
    return action if graph_mode else action.flatten(1)


def select_decision_action(policy_action, held_action, decision_mask):
    """Use a new policy action only on rows where an Actor decision exists."""
    shape = [decision_mask.shape[0]] + [1] * (policy_action.ndim - 1)
    return decision_mask.reshape(shape) * policy_action + (1.0 - decision_mask.reshape(shape)) * held_action


def augment_critic_state(state, held_normalized, phase, actor_period: int, graph_mode: bool):
    """Add held residual and phase to the centralized state, never Actor input."""
    nodes = state["node_features"] if graph_mode else state
    phase_value = phase / float(max(1, actor_period))
    phase_feature = phase_value[:, None, :].expand(-1, nodes.shape[1], -1)
    # At a decision state the previous hold has no control authority and must not
    # leak the newly selected action into the state side of Q(s, a).
    hold_mask = (phase > 0.0)[:, None, :].to(nodes.dtype)
    timing = torch.cat((held_normalized * hold_mask, phase_feature), dim=-1)
    timing = timing * (state["node_mask"].unsqueeze(-1) if graph_mode else nodes[..., NODE_FEATURES.active_mask])
    augmented_nodes = torch.cat((nodes, timing), dim=-1)
    if not graph_mode:
        return augmented_nodes
    result = dict(state)
    result["node_features"] = augmented_nodes
    return result


@contextmanager
def frozen_parameters(*modules):
    parameters = [parameter for module in modules for parameter in module.parameters()]
    previous = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, enabled in zip(parameters, previous, strict=True):
            parameter.requires_grad_(enabled)
