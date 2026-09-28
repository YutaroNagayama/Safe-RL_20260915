"""Graph observations shared by online execution and graph-valued replay."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from uav_safe_marl.communication.message import P2PMessage
from uav_safe_marl.core.types import AgentState
from .feature_layout import EDGE_DIRECT_WIDTH


GraphObservation = dict[str, np.ndarray]


def direct_node_features(
    states: Sequence[AgentState],
    active_mask: np.ndarray | None = None,
    reference_actions: np.ndarray | None = None,
) -> np.ndarray:
    """Return state plus exact own capability for each padded agent slot."""
    references = np.zeros((len(states), 3), dtype=np.float64) if reference_actions is None else np.asarray(reference_actions, dtype=np.float64)
    if references.shape != (len(states), 3):
        raise ValueError("reference_actions must have shape (agent_count, 3)")
    features = np.vstack([
        np.concatenate((
            state.position,
            state.velocity,
            state.goal - state.position,
            [float(np.min(state.acceleration_limit)), float(np.min(state.velocity_limit))],
            references[index] / np.maximum(state.acceleration_limit, 1e-12),
        ))
        for index, state in enumerate(states)
    ])
    if active_mask is not None:
        features = features * np.asarray(active_mask, dtype=bool)[:, None]
    return features


def build_actor_graph(
    states: Sequence[AgentState],
    cache: Mapping[int, Mapping[int, P2PMessage]],
    embedding_dim: int,
    active_mask: np.ndarray | None = None,
    reference_actions: np.ndarray | None = None,
) -> GraphObservation:
    """Build directed receiver<-sender P2P edges without global-state leakage."""
    count = len(states)
    active = np.ones(count, dtype=bool) if active_mask is None else np.asarray(active_mask, dtype=bool)
    if active.shape != (count,):
        raise ValueError("active_mask must have shape (agent_count,)")
    adjacency = np.zeros((count, count), dtype=bool)
    edge_features = np.zeros((count, count, EDGE_DIRECT_WIDTH + embedding_dim), dtype=np.float64)
    for receiver_id, receiver in enumerate(states):
        if not active[receiver_id]:
            continue
        for sender_id, message in cache.get(receiver_id, {}).items():
            if not 0 <= sender_id < count or sender_id == receiver_id or not active[sender_id]:
                continue
            delayed = np.asarray(message.embedding[:embedding_dim], dtype=np.float64)
            delayed = np.pad(delayed, (0, embedding_dim - delayed.size))
            adjacency[receiver_id, sender_id] = True
            edge_features[receiver_id, sender_id] = np.concatenate((
                message.position - receiver.position,
                message.velocity - receiver.velocity,
                [float(message.age_steps)],
                delayed,
            ))
    return {
        "node_features": direct_node_features(states, active, reference_actions),
        "adjacency": adjacency,
        "edge_features": edge_features,
        "node_mask": active.copy(),
    }


def stack_graph_observations(observations: Sequence[GraphObservation], maximum: int | None = None) -> GraphObservation:
    """Pad variable-agent graphs into a masked dense batch."""
    if not observations:
        raise ValueError("cannot stack an empty graph batch")
    batch = len(observations)
    maximum = maximum or max(item["node_features"].shape[0] for item in observations)
    node_width = observations[0]["node_features"].shape[-1]
    edge_width = observations[0]["edge_features"].shape[-1]
    nodes = np.zeros((batch, maximum, node_width), dtype=np.float64)
    adjacency = np.zeros((batch, maximum, maximum), dtype=bool)
    edges = np.zeros((batch, maximum, maximum, edge_width), dtype=np.float64)
    mask = np.zeros((batch, maximum), dtype=bool)
    for index, graph in enumerate(observations):
        count = graph["node_features"].shape[0]
        nodes[index, :count] = graph["node_features"]
        adjacency[index, :count, :count] = graph["adjacency"]
        edges[index, :count, :count] = graph["edge_features"]
        mask[index, :count] = graph["node_mask"]
    return {"node_features": nodes, "adjacency": adjacency, "edge_features": edges, "node_mask": mask}


def copy_graph(graph: GraphObservation) -> GraphObservation:
    return {key: np.asarray(value).copy() for key, value in graph.items()}
