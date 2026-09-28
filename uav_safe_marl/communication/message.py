"""P2P message schema."""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True, frozen=True)
class P2PMessage:
    """Direct state plus a causally delayed neighbour embedding and its age."""

    sender_id: int
    receiver_id: int
    position: np.ndarray
    velocity: np.ndarray
    embedding: np.ndarray
    age_steps: int
    sent_step: int

    @property
    def byte_size(self) -> int:
        """Approximate numeric payload size, excluding transport headers."""
        return int(self.position.nbytes + self.velocity.nbytes + self.embedding.nbytes + 16)
