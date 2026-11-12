"""Seeded fixed-capacity replay buffer."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from uav_safe_marl.core.types import Transition
from uav_safe_marl.graph.observations import stack_graph_observations


class NumpyReplayBuffer:
    """Store both nominal and executed safe actions for correct CTDE training."""

    def __init__(self, capacity: int, seed: int = 0, storage_dtype: Any = np.float64) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.storage_dtype = np.dtype(storage_dtype)
        if self.storage_dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
            raise ValueError("storage_dtype must be float32 or float64")
        self._data: list[Transition | None] = [None] * self.capacity
        self._size = 0
        self._start = 0
        self._rng = np.random.default_rng(seed)
        self._occupancy_by_n: dict[int, int] = {}

    def _freeze_array(self, value: Any) -> np.ndarray:
        """Freeze replay inputs in the configured storage precision.

        Exact mode retains the historical float64 snapshot contract. Fast mode
        uses float32, which is also the precision consumed by the learners, to
        halve RAM and checkpoint traffic.
        """
        array = np.asarray(value)
        dtype = self.storage_dtype if np.issubdtype(array.dtype, np.floating) else array.dtype
        return np.asarray(array, dtype=dtype).copy()

    def _freeze_observation(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._freeze_array(item) for key, item in value.items()}
        return self._freeze_array(value)

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
        state = self._freeze_observation(transition.state)
        next_state = self._freeze_observation(transition.next_state)
        frozen = Transition(
            state=state,
            nominal_action=self._freeze_array(transition.nominal_action),
            safe_action=self._freeze_array(transition.safe_action),
            reward=float(transition.reward),
            cost=float(transition.cost),
            next_state=next_state,
            terminated=bool(transition.terminated),
            truncated=bool(transition.truncated),
            extras=dict(transition.extras),
            reference_action=None if transition.reference_action is None else self._freeze_array(transition.reference_action),
            normalized_residual_action=None if transition.normalized_residual_action is None else self._freeze_array(transition.normalized_residual_action),
            residual_action=None if transition.residual_action is None else self._freeze_array(transition.residual_action),
            command_action=None if transition.command_action is None else self._freeze_array(transition.command_action),
            actor_phase=int(transition.actor_phase),
            is_actor_decision_tick=bool(transition.is_actor_decision_tick),
            held_residual_action=None if transition.held_residual_action is None else self._freeze_array(transition.held_residual_action),
            next_actor_phase=int(transition.next_actor_phase),
            next_is_actor_decision_tick=bool(transition.next_is_actor_decision_tick),
            next_held_residual_action=None if transition.next_held_residual_action is None else self._freeze_array(transition.next_held_residual_action),
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
                result = np.zeros((len(selected), maximum, 3), dtype=self.storage_dtype)
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
            "reward": np.asarray([x.reward for x in selected], dtype=self.storage_dtype)[:, None],
            "cost": np.asarray([x.cost for x in selected], dtype=self.storage_dtype)[:, None],
            "next_state": next_state,
            "terminated": np.asarray([x.terminated for x in selected], dtype=self.storage_dtype)[:, None],
            "truncated": np.asarray([x.truncated for x in selected], dtype=self.storage_dtype)[:, None],
            # Kept for older analysis consumers; this is deliberately identical
            # to terminated rather than terminated OR truncated.
            "done": np.asarray([x.terminated for x in selected], dtype=self.storage_dtype)[:, None],
            "bootstrap_mask": 1.0 - np.asarray([x.terminated for x in selected], dtype=self.storage_dtype)[:, None],
            "actor_phase": np.asarray([x.actor_phase for x in selected], dtype=self.storage_dtype)[:, None],
            "is_actor_decision_tick": np.asarray([x.is_actor_decision_tick for x in selected], dtype=self.storage_dtype)[:, None],
            "next_actor_phase": np.asarray([x.next_actor_phase for x in selected], dtype=self.storage_dtype)[:, None],
            "next_is_actor_decision_tick": np.asarray([x.next_is_actor_decision_tick for x in selected], dtype=self.storage_dtype)[:, None],
        }
        for name in ("reference_action", "normalized_residual_action", "residual_action", "command_action", "held_residual_action", "next_held_residual_action"):
            if all(getattr(item, name) is not None for item in selected):
                if graph_mode:
                    values = np.zeros((len(selected), maximum, 3), dtype=self.storage_dtype)
                    for index, transition in enumerate(selected):
                        value = np.asarray(getattr(transition, name))
                        values[index, :len(value)] = value
                else:
                    values = np.stack([np.asarray(getattr(item, name)) for item in selected])
                result[name] = values
        return result

    def _n_step_window(self, start: int, n_step: int, gamma: float) -> tuple[int, int, int, float, float] | None:
        """Build one valid within-episode n-step window.

        The tuple is ``(start, end, horizon, discounted_cost, bootstrap_mask)``.
        Time-limit truncation ends the stored trajectory but retains bootstrap;
        true MDP termination disables it. Windows never cross an episode edge.
        """
        discounted_cost = 0.0
        discount = 1.0
        end = start
        for offset in range(n_step):
            logical = start + offset
            if logical >= self._size:
                return None
            transition = self._at(logical)
            discounted_cost += discount * float(transition.cost)
            discount *= gamma
            end = logical
            if transition.terminated or transition.truncated:
                return (start, end, end - start + 1, discounted_cost, 0.0 if transition.terminated else 1.0)
        final = self._at(end)
        return (start, end, n_step, discounted_cost, 0.0 if final.terminated else 1.0)

    def _n_step_candidate_count(self, n_step: int) -> int:
        """Count valid starts in O(n), not O(replay size).

        Every stored start is valid except the unfinished suffix at the newest
        end of the replay.  That suffix contains at most ``n_step - 1`` starts.
        Episode boundaries inside the buffer complete, rather than invalidate,
        a window.
        """
        if self._size == 0:
            return 0
        trailing = 0
        for offset in range(min(n_step - 1, self._size)):
            transition = self._at(self._size - 1 - offset)
            if transition.terminated or transition.truncated:
                break
            trailing += 1
        return self._size - trailing

    def sample_n_step(self, batch_size: int, n_step: int, gamma: float) -> dict[str, Any]:
        """Sample sequence-safe n-step Cost targets from the frozen replay."""
        if n_step < 1:
            raise ValueError("n_step must be positive")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        candidate_count = self._n_step_candidate_count(n_step)
        if batch_size < 1 or batch_size > candidate_count:
            raise ValueError("batch_size must be in [1, number of valid n-step windows]")
        picked = [
            self._n_step_window(int(i), n_step, gamma)
            for i in self._rng.choice(candidate_count, batch_size, replace=False)
        ]
        if any(item is None for item in picked):
            raise RuntimeError("n-step candidate accounting produced an incomplete window")
        picked = [item for item in picked if item is not None]
        starts = [self._at(item[0]) for item in picked]
        ends = [self._at(item[1]) for item in picked]
        result = self._collate(starts)
        # The n-step target consumes only the end transition's next fields.
        if isinstance(ends[0].state, dict):
            maximum = max(max(x.state["node_features"].shape[0], x.next_state["node_features"].shape[0]) for x in ends)
            next_state = stack_graph_observations([x.next_state for x in ends], maximum)
        else:
            maximum = None
            next_state = np.stack([x.next_state for x in ends])
        if any(x.next_held_residual_action is None for x in ends):
            raise KeyError("next_held_residual_action")
        if maximum is not None:
            next_held = np.zeros((len(ends), maximum, 3), dtype=self.storage_dtype)
            for index, transition in enumerate(ends):
                held = np.asarray(transition.next_held_residual_action)
                next_held[index, :len(held)] = held
        else:
            next_held = np.stack([np.asarray(x.next_held_residual_action) for x in ends])
        result.update({
            "n_step_cost": np.asarray([item[3] for item in picked], dtype=self.storage_dtype)[:, None],
            "n_step_discount": np.asarray([gamma ** item[2] for item in picked], dtype=self.storage_dtype)[:, None],
            "n_step_bootstrap_mask": np.asarray([item[4] for item in picked], dtype=self.storage_dtype)[:, None],
            "n_step_horizon": np.asarray([item[2] for item in picked], dtype=np.int64)[:, None],
            "n_step_next_state": next_state,
            "n_step_next_actor_phase": np.asarray([x.next_actor_phase for x in ends], dtype=self.storage_dtype)[:, None],
            "n_step_next_is_actor_decision_tick": np.asarray([x.next_is_actor_decision_tick for x in ends], dtype=self.storage_dtype)[:, None],
            "n_step_next_held_residual_action": next_held,
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
            "storage_dtype": self.storage_dtype.str,
            "data": self._data,
            "size": self._size,
            "start": self._start,
            "rng_state": self._rng.bit_generator.state,
            "occupancy_by_n": self._occupancy_by_n,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError("replay capacity differs from checkpoint")
        saved_dtype = np.dtype(state.get("storage_dtype", np.float64))
        if saved_dtype != self.storage_dtype:
            raise ValueError("replay storage dtype differs from checkpoint")
        data = list(state["data"])
        if len(data) != self.capacity:
            raise ValueError("invalid replay snapshot length")
        self._data = data
        self._size = int(state["size"])
        self._start = int(state["start"])
        self._rng.bit_generator.state = state["rng_state"]
        self._occupancy_by_n = {int(k): int(v) for k, v in state.get("occupancy_by_n", {}).items()}
