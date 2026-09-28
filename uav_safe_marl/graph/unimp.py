"""One-round, two-hop-equivalent variable-neighbour aggregation."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from uav_safe_marl.communication.message import P2PMessage
from uav_safe_marl.core.types import AgentState
from .feature_layout import EDGE_DIRECT_WIDTH, NODE_FEATURES


class UniMPEncoder:
    """Aggregate a variable message set into fixed mean/max statistics.

    Each message already contains the sender's previous neighbour embedding, so
    one communication round exposes two-hop-equivalent information as required
    by PDF slide 20. A neural projection may consume this fixed representation.
    """

    def __init__(self, enabled: bool = True, embedding_dim: int = 23) -> None:
        self.enabled = enabled
        self.embedding_dim = embedding_dim

    @property
    def output_dim(self) -> int:
        """Fixed actor input width."""
        # The final scalar is an explicit actual-agent mask for the padded
        # pooled/concat baseline.
        return NODE_FEATURES.pooled_direct_width + 2 * (EDGE_DIRECT_WIDTH + self.embedding_dim)

    def encode(self, self_state: AgentState, messages: Sequence[P2PMessage], active: bool = True, reference_action: np.ndarray | None = None) -> np.ndarray:
        reference = np.zeros(3, dtype=np.float64) if reference_action is None else np.asarray(reference_action, dtype=np.float64)
        direct = np.concatenate((
            self_state.position,
            self_state.velocity,
            self_state.goal - self_state.position,
            [float(np.min(self_state.acceleration_limit)), float(np.min(self_state.velocity_limit))],
            reference / np.maximum(self_state.acceleration_limit, 1e-12),
            [float(active)],
        ))
        if not self.enabled or not messages:
            return np.concatenate((direct, np.zeros(self.output_dim - direct.size))) if active else np.zeros(self.output_dim)
        features = []
        for message in messages:
            delayed = np.pad(message.embedding[: self.embedding_dim], (0, max(0, self.embedding_dim - message.embedding.size)))
            base = np.concatenate((message.position - self_state.position, message.velocity - self_state.velocity, [float(message.age_steps)], delayed))
            features.append(base)
        matrix = np.vstack(features)
        aggregate = np.concatenate((matrix.mean(axis=0), matrix.max(axis=0)))
        return np.concatenate((direct, aggregate)) if active else np.zeros(self.output_dim)
