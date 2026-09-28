"""Protocols that keep experiment modules replaceable."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from .types import AgentState, HOCBFConstraint, SafetyResult, Transition


class DynamicsModel(Protocol):
    def step(self, states: Sequence[AgentState], accelerations: NDArray[np.float64], dt: float, active_mask: NDArray[np.bool_] | None = None) -> list[AgentState]: ...


class CommunicationModel(Protocol):
    def reset(self, predicted_los_edges: Sequence[tuple[int, int]] | None = None) -> None: ...
    def receive(self, step: int, active_mask: NDArray[np.bool_] | None = None) -> Any: ...
    def send(self, states: Sequence[AgentState], embeddings: NDArray[np.float64] | None, step: int, active_mask: NDArray[np.bool_] | None = None) -> None: ...
    def exchange(self, states: Sequence[AgentState], embeddings: NDArray[np.float64] | None, step: int) -> Any: ...


class InformationEncoder(Protocol):
    def encode(self, self_state: AgentState, messages: Sequence[Any]) -> NDArray[np.float64]: ...


class ActorBackend(Protocol):
    def act(self, observations: Any, deterministic: bool) -> NDArray[np.float64]: ...
    def update(self, batch: dict[str, Any]) -> dict[str, float]: ...


class Critic(Protocol):
    def __call__(self, states: Any, actions: Any) -> Any: ...


class RewardModel(Protocol):
    def compute(self, previous: Sequence[AgentState], current: Sequence[AgentState], reached: NDArray[np.bool_], active_mask: NDArray[np.bool_] | None = None) -> tuple[float, dict[str, float]]: ...


class CostModel(Protocol):
    def compute(self, states: Sequence[AgentState], corrections: NDArray[np.float64], active_mask: NDArray[np.bool_] | None = None) -> tuple[float, dict[str, float]]: ...


class ResponsibilityAllocator(Protocol):
    def allocate_pair(self, i: AgentState, j: AgentState, q_ij: float) -> tuple[float, float]: ...


class ConflictResolver(Protocol):
    def resolve(
        self,
        nominal: NDArray[np.float64],
        constraints: Sequence[HOCBFConstraint],
        lower: NDArray[np.float64],
        upper: NDArray[np.float64],
        *,
        velocity: NDArray[np.float64] | None = None,
        speed_limit: float | None = None,
        dt: float | None = None,
    ) -> SafetyResult: ...


class SafetyFilter(Protocol):
    def filter(self, agent_id: int, states: Sequence[AgentState], nominal: NDArray[np.float64], active_mask: NDArray[np.bool_] | None = None) -> SafetyResult: ...


class QPSolverBackend(Protocol):
    def solve(
        self,
        nominal: NDArray[np.float64],
        constraints: Sequence[HOCBFConstraint],
        lower: NDArray[np.float64] | None,
        upper: NDArray[np.float64] | None,
        *,
        velocity: NDArray[np.float64] | None = None,
        speed_limit: float | None = None,
        dt: float | None = None,
    ) -> tuple[NDArray[np.float64], bool]: ...


class ReplayBuffer(Protocol):
    def add(self, transition: Transition) -> None: ...
    def sample(self, batch_size: int) -> dict[str, Any]: ...


class SafeRLTrainer(Protocol):
    def train(self, episodes: int, reset_options: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...
