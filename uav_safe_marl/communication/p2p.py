"""Configurable prior/reactive P2P information sharing."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from uav_safe_marl.core.types import AgentState
from .delay import DelayQueue
from .message import P2PMessage


class P2PCommunication:
    """Causal P2P transport: receive old messages before sending new ones."""

    def __init__(self, mode: str, delay_steps: int, d_eng: float, actor_period_k: int) -> None:
        self.mode, self.delay_steps, self.d_eng, self.actor_period_k = mode, delay_steps, d_eng, actor_period_k
        self.queue = DelayQueue()
        self.message_count = 0
        self.transmitted_bytes = 0
        self.predicted_los_edges: set[tuple[int, int]] = set()

    def reset(self, predicted_los_edges: Sequence[tuple[int, int]] = ()) -> None:
        self.queue.clear()
        self.message_count = self.transmitted_bytes = 0
        self.predicted_los_edges = {tuple(sorted((int(first), int(second)))) for first, second in predicted_los_edges if first != second}

    def _linked(self, sender_id: int, receiver_id: int, first: AgentState, second: AgentState) -> bool:
        if self.mode == "none":
            return False
        if self.mode == "prior_share":
            return tuple(sorted((sender_id, receiver_id))) in self.predicted_los_edges
        return np.linalg.norm(first.position - second.position) <= self.d_eng

    def receive(self, step: int, active_mask: np.ndarray | None = None) -> dict[int, list[P2PMessage]]:
        """Deliver only messages sent on an earlier tick."""
        delivered: dict[int, list[P2PMessage]] = defaultdict(list)
        active = None if active_mask is None else np.asarray(active_mask, dtype=bool)
        for message in self.queue.pop(step):
            if active is not None and (not active[message.sender_id] or not active[message.receiver_id]):
                continue
            delivered[message.receiver_id].append(message)
        return dict(delivered)

    def send(self, states: Sequence[AgentState], embeddings: np.ndarray | None, step: int, active_mask: np.ndarray | None = None) -> None:
        """Enqueue current messages with at least one tick of causal delay."""
        include_embedding = embeddings is not None and step % self.actor_period_k == 0
        effective_delay = max(1, self.delay_steps)
        active = np.ones(len(states), dtype=bool) if active_mask is None else np.asarray(active_mask, dtype=bool)
        for sender_id, sender in enumerate(states):
            if not active[sender_id]:
                continue
            for receiver_id, receiver in enumerate(states):
                if sender_id == receiver_id or not active[receiver_id] or not self._linked(sender_id, receiver_id, sender, receiver):
                    continue
                embedding = np.asarray(embeddings[sender_id], dtype=np.float64).copy() if include_embedding else np.empty(0)
                message = P2PMessage(sender_id, receiver_id, sender.position.copy(), sender.velocity.copy(), embedding, effective_delay, step)
                self.queue.push(step + effective_delay, message)
                self.message_count += 1
                self.transmitted_bytes += message.byte_size

    def exchange(self, states: Sequence[AgentState], embeddings: np.ndarray | None, step: int) -> dict[int, list[P2PMessage]]:
        """Compatibility wrapper preserving receive-before-send causality."""
        delivered = self.receive(step)
        self.send(states, embeddings, step)
        return delivered
