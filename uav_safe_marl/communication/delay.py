"""Deterministic delayed-message queue."""

from collections import defaultdict
from typing import DefaultDict

from .message import P2PMessage


class DelayQueue:
    def __init__(self) -> None:
        self._queue: DefaultDict[int, list[P2PMessage]] = defaultdict(list)

    def push(self, delivery_step: int, message: P2PMessage) -> None:
        self._queue[delivery_step].append(message)

    def pop(self, step: int) -> list[P2PMessage]:
        return self._queue.pop(step, [])

    def clear(self) -> None:
        self._queue.clear()

