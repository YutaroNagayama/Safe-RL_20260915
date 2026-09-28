"""Variable-neighbour graph utilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from uav_safe_marl.communication.message import P2PMessage


def message_features(
    messages: Sequence[P2PMessage],
    own_position: np.ndarray,
    embedding_dim: int = 23,
) -> np.ndarray:
    """Stack fixed-width direct state, delayed embedding, and information age."""
    if embedding_dim < 0:
        raise ValueError("embedding_dim must be non-negative")
    if not messages:
        return np.empty((0, 7 + embedding_dim), dtype=np.float64)
    rows = []
    for message in messages:
        embedding = message.embedding[:embedding_dim]
        embedding = np.pad(embedding, (0, embedding_dim - embedding.size))
        rows.append(np.concatenate((message.position - own_position, message.velocity, embedding, [float(message.age_steps)])))
    return np.vstack(rows)
