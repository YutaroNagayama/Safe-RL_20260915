"""Seeded fixed-capacity replay buffer."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from uav_safe_marl.core.types import Transition
from uav_safe_marl.graph.observations import copy_graph, stack_graph_observations


class NumpyReplayBuffer:
    """Store both nominal and executed safe actions for correct CTDE training."""

    def __init__(self, capacity: int, seed: int = 0) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._data: list[Transition | None] = [None] * self.capacity
        self._size = 0
        self._start = 0
        self._rng = np.random.default_rng(seed)
        self._occupancy_by_n: dict[int, int] = {}
        self._n_step_cache: dict[tuple[int, float, int, int], list[tuple[int, int, int, float, float]]] = {}

    def __len__(self) -> int:
        return self._size

    def _physical_index(self, logical_index: int) -> int:
        if not 0 <= logical_index < self._size:
            raise IndexError(logical_index)
        return (self._start + logical_index) % self.capacity

    def _at(self, logical_index: int) -> Transition:
        item = self._data[self._physical_index(logical_index)]
        assert item is not None
        return item

    def add(self, transition: Transition) -> None:
        # Freeze the information set that was actually observable at collection
        # time. In graph mode this includes delayed peer z values and their ages;
        # they must never be recomputed by the current Actor Encoder on sampling.
        state = copy_graph(transition.state) if isinstance(transition.state, dict) else np.asarray(transition.state).copy()
        next_state = copy_graph(transition.next_state) if isinstance(transition.next_state, dict) else np.asarray(transition.next_state).copy()
        frozen = Transition(
            state=state,
            nominal_action=np.asarray(transition.nominal_action).copy(),
            safe_action=np.asarray(transition.safe_action).copy(),
            reward=float(transition.reward),
            cost=float(transition.cost),
            next_state=next_state,
            terminated=bool(transition.terminated),
            truncated=bool(transition.truncated),
            extras=dict(transition.extras),
            reference_action=None if transition.reference_action is None else np.asarray(transition.reference_action).copy(),
            normalized_residual_action=None if transition.normalized_residual_action is None else np.asarray(transition.normalized_residual_action).copy(),
            residual_action=None if transition.residual_action is None else np.asarray(transition.residual_action).copy(),
            command_action=None if transition.command_action is None else np.asarray(transition.command_action).copy(),
            actor_phase=int(transition.actor_phase),
            is_actor_decision_tick=bool(transition.is_actor_decision_tick),
            held_residual_action=None if transition.held_residual_action is None else np.asarray(transition.held_residual_action).copy(),
            next_actor_phase=int(transition.next_actor_phase),
            next_is_actor_decision_tick=bool(transition.next_is_actor_decision_tick),
            next_held_residual_action=None if transition.next_held_residual_action is None else np.asarray(transition.next_held_residual_action).copy(),
        )
        count_value = frozen.extras.get("actual_agent_count")
        count = None if count_value is None else int(count_value)
        if self._size < self.capacity:
            self._data[(self._start + self._size) % self.capacity] = frozen
            self._size += 1
        else:
            previous = self._data[self._start]
            if previous is not None and previous.extras.get("actual_agent_count") is not None:
                previous_count = int(previous.extras["actual_agent_count"])
                remaining = self._occupancy_by_n.get(previous_count, 0) - 1
                if remaining > 0:
                    self._occupancy_by_n[previous_count] = remaining
                else:
                    self._occupancy_by_n.pop(previous_count, None)
            self._data[self._start] = frozen
            self._start = (self._start + 1) % self.capacity
        if count is not None:
            self._occupancy_by_n[count] = self._occupancy_by_n.get(count, 0) + 1
        self._n_step_cache.clear()

    def sample(self, batch_size: int) -> dict[str, Any]:
        if batch_size < 1 or batch_size > self._size:
            raise ValueError("batch_size must be in [1, len(buffer)]")
        selected = [self._at(int(i)) for i in self._rng.choice(self._size, batch_size, replace=False)]
        return self._collate(selected)

    def _collate(self, selected: list[Transition]) -> dict[str, Any]:
        """Collate already-selected transitions without changing their order."""
        graph_mode = isinstance(selected[0].state, dict)
        if graph_mode:
            maximum = max(max(x.state["node_features"].shape[0], x.next_state["node_features"].shape[0]) for x in selected)
            state = stack_graph_observations([x.state for x in selected], maximum)
            next_state = stack_graph_observations([x.next_state for x in selected], maximum)

            def padded_action(name: str) -> np.ndarray:
                result = np.zeros((len(selected), maximum, 3), dtype=np.float64)
                for index, transition in enumerate(selected):
                    value = np.asarray(getattr(transition, name))
                    result[index, : len(value)] = value
                return result
            nominal_action = padded_action("nominal_action")
            safe_action = padded_action("safe_action")
        else:
            state = np.stack([x.state for x in selected])
            next_state = np.stack([x.next_state for x in selected])
            nominal_action = np.stack([x.nominal_action for x in selected])
            safe_action = np.stack([x.safe_action for x in selected])
        result = {
            "state": state,
            "nominal_action": nominal_action,
            "safe_action": safe_action,
            "reward": np.asarray([x.reward for x in selected], dtype=np.float64)[:, None],
            "cost": np.asarray([x.cost for x in selected], dtype=np.float64)[:, None],
            "next_state": next_state,
            "terminated": np.asarray([x.terminated for x in selected], dtype=np.float64)[:, None],
            "truncated": np.asarray([x.truncated for x in selected], dtype=np.float64)[:, None],
            # Kept for older analysis consumers; this is deliberately identical
            # to terminated rather than terminated OR truncated.
            "done": np.asarray([x.terminated for x in selected], dtype=np.float64)[:, None],
            "bootstrap_mask": 1.0 - np.asarray([x.terminated for x in selected], dtype=np.float64)[:, None],
            "actor_phase": np.asarray([x.actor_phase for x in selected], dtype=np.float64)[:, None],
            "is_actor_decision_tick": np.asarray([x.is_actor_decision_tick for x in selected], dtype=np.float64)[:, None],
            "next_actor_phase": np.asarray([x.next_actor_phase for x in selected], dtype=np.float64)[:, None],
            "next_is_actor_decision_tick": np.asarray([x.next_is_actor_decision_tick for x in selected], dtype=np.float64)[:, None],
        }
        for name in ("reference_action", "normalized_residual_action", "residual_action", "command_action", "held_residual_action", "next_held_residual_action"):
            if all(getattr(item, name) is not None for item in selected):
                if graph_mode:
                    values = np.zeros((len(selected), maximum, 3), dtype=np.float64)
                    for index, transition in enumerate(selected):
                        value = np.asarray(getattr(transition, name))
                        values[index, :len(value)] = value
                else:
                    values = np.stack([np.asarray(getattr(item, name)) for item in selected])
                result[name] = values
        return result

    def _n_step_candidates(self, n_step: int, gamma: float) -> list[tuple[int, int, int, float, float]]:
        """Return valid within-episode n-step windows in logical buffer order.

        Each tuple is ``(start, end, horizon, discounted_cost, bootstrap_mask)``.
        Time-limit truncation ends the stored trajectory but retains bootstrap;
        true MDP termination disables it. Windows never cross an episode edge.
        """
        if n_step < 1:
            raise ValueError("n_step must be positive")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        key = (int(n_step), float(gamma), self._size, self._start)
        cached = self._n_step_cache.get(key)
        if cached is not None:
            return cached
        candidates: list[tuple[int, int, int, float, float]] = []
        for start in range(self._size):
            discounted_cost = 0.0
            discount = 1.0
            end = start
            complete = False
            for offset in range(n_step):
                logical = start + offset
                if logical >= self._size:
                    break
                transition = self._at(logical)
                discounted_cost += discount * float(transition.cost)
                discount *= gamma
                end = logical
                if transition.terminated or transition.truncated:
                    complete = True
                    break
                if offset + 1 == n_step:
                    complete = True
            if complete:
                final = self._at(end)
                candidates.append((start, end, end - start + 1, discounted_cost, 0.0 if final.terminated else 1.0))
        self._n_step_cache[key] = candidates
        return candidates

    def sample_n_step(self, batch_size: int, n_step: int, gamma: float) -> dict[str, Any]:
        """Sample sequence-safe n-step Cost targets from the frozen replay."""
        candidates = self._n_step_candidates(n_step, gamma)
        if batch_size < 1 or batch_size > len(candidates):
            raise ValueError("batch_size must be in [1, number of valid n-step windows]")
        picked = [candidates[int(i)] for i in self._rng.choice(len(candidates), batch_size, replace=False)]
        starts = [self._at(item[0]) for item in picked]
        ends = [self._at(item[1]) for item in picked]
        result = self._collate(starts)
        end_batch = self._collate(ends)
        result.update({
            "n_step_cost": np.asarray([item[3] for item in picked], dtype=np.float64)[:, None],
            "n_step_discount": np.asarray([gamma ** item[2] for item in picked], dtype=np.float64)[:, None],
            "n_step_bootstrap_mask": np.asarray([item[4] for item in picked], dtype=np.float64)[:, None],
            "n_step_horizon": np.asarray([item[2] for item in picked], dtype=np.int64)[:, None],
            "n_step_next_state": end_batch["next_state"],
            "n_step_next_actor_phase": end_batch["next_actor_phase"],
            "n_step_next_is_actor_decision_tick": end_batch["next_is_actor_decision_tick"],
            "n_step_next_held_residual_action": end_batch["next_held_residual_action"],
        })
        return result

    def occupancy_by_agent_count(self) -> dict[int, int]:
        """Count currently retained transitions by collection-time active N."""
        return dict(self._occupancy_by_n)

    def sampling_state_dict(self) -> dict[str, Any]:
        """Return the small RNG-only state needed for exact update resumption."""
        return deepcopy(self._rng.bit_generator.state)

    def load_sampling_state_dict(self, state: dict[str, Any]) -> None:
        """Restore sampling RNG without duplicating the potentially huge replay."""
        self._rng.bit_generator.state = deepcopy(state)

    def state_dict(self) -> dict[str, Any]:
        """Return an exact, pickle-safe training-resume snapshot."""
        return {
            "capacity": self.capacity,
            "data": self._data,
            "size": self._size,
            "start": self._start,
            "rng_state": self._rng.bit_generator.state,
            "occupancy_by_n": self._occupancy_by_n,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError("replay capacity differs from checkpoint")
        data = list(state["data"])
        if len(data) != self.capacity:
            raise ValueError("invalid replay snapshot length")
        self._data = data
        self._size = int(state["size"])
        self._start = int(state["start"])
        self._rng.bit_generator.state = state["rng_state"]
        self._occupancy_by_n = {int(k): int(v) for k, v in state.get("occupancy_by_n", {}).items()}
        self._n_step_cache.clear()
