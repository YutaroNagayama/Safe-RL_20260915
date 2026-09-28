"""Initial-state replay that reproduces the configured training distribution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from uav_safe_marl.graph.observations import copy_graph, stack_graph_observations


@dataclass(slots=True)
class InitialStateRecord:
    observation: Any
    agent_count: int
    scenario_id: str | None


class InitialStateBuffer:
    """Store exact episode-start observations for episodic dual estimation."""

    def __init__(self, capacity: int, seed: int = 0) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._records: list[InitialStateRecord] = []
        self._next = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._records)

    @staticmethod
    def _copy_observation(observation: Any) -> Any:
        return copy_graph(observation) if isinstance(observation, dict) else np.asarray(observation).copy()

    def add(self, observation: Any, agent_count: int, scenario_id: str | None = None) -> None:
        record = InitialStateRecord(self._copy_observation(observation), int(agent_count), scenario_id)
        if len(self._records) < self.capacity:
            self._records.append(record)
        else:
            self._records[self._next] = record
            self._next = (self._next + 1) % self.capacity

    def occupancy_by_agent_count(self) -> dict[int, int]:
        result: dict[int, int] = {}
        for record in self._records:
            result[record.agent_count] = result.get(record.agent_count, 0) + 1
        return result

    def supports(self, distribution: Mapping[int, float] | None = None) -> bool:
        if not self._records:
            return False
        if not distribution:
            return True
        available = self.occupancy_by_agent_count()
        return all(available.get(int(count), 0) > 0 for count, probability in distribution.items() if probability > 0.0)

    def sample(self, batch_size: int, distribution: Mapping[int, float] | None = None) -> dict[str, Any]:
        if batch_size < 1 or not self._records:
            raise ValueError("cannot sample an empty initial-state buffer")
        indices_by_n: dict[int, list[int]] = {}
        for index, record in enumerate(self._records):
            indices_by_n.setdefault(record.agent_count, []).append(index)
        if distribution:
            if not self.supports(distribution):
                raise ValueError("initial-state buffer does not yet cover the configured start distribution")
            counts = np.asarray(sorted(int(n) for n, p in distribution.items() if p > 0.0), dtype=np.int64)
            probabilities = np.asarray([float(distribution[int(n)]) for n in counts], dtype=np.float64)
            probabilities /= probabilities.sum()
        else:
            counts = np.asarray(sorted(indices_by_n), dtype=np.int64)
            probabilities = np.asarray([len(indices_by_n[int(n)]) for n in counts], dtype=np.float64)
            probabilities /= probabilities.sum()
        sampled_n = self._rng.choice(counts, size=batch_size, replace=True, p=probabilities)
        selected = [
            self._records[int(self._rng.choice(indices_by_n[int(count)]))]
            for count in sampled_n
        ]
        graph_mode = isinstance(selected[0].observation, dict)
        if graph_mode:
            maximum = max(record.observation["node_features"].shape[0] for record in selected)
            state = stack_graph_observations([record.observation for record in selected], maximum)
        else:
            state = np.stack([record.observation for record in selected])
        sampled_counts = {int(n): int(np.sum(sampled_n == n)) for n in counts}
        return {
            "state": state,
            "actual_agent_count": sampled_n,
            "sampled_count_by_n": sampled_counts,
            "scenario_id": [record.scenario_id for record in selected],
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "records": self._records,
            "next": self._next,
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError("initial-state buffer capacity differs from checkpoint")
        self._records = list(state["records"])
        self._next = int(state["next"])
        self._rng.bit_generator.state = state["rng_state"]
