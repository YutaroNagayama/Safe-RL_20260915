"""Resource-aware execution of independent per-agent safety QPs."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from time import perf_counter
from typing import Sequence

import numpy as np

from uav_safe_marl.core.types import AgentState, HOCBFConstraint, SafetyResult
from .qp_filter import HOCBFSafetyFilter


_WORKER_FILTER: HOCBFSafetyFilter | None = None


def _initialize_worker(safety_filter: HOCBFSafetyFilter) -> None:
    global _WORKER_FILTER
    _WORKER_FILTER = safety_filter


def _solve_job(job: tuple[int, Sequence[AgentState], np.ndarray, Sequence[HOCBFConstraint], np.ndarray, bool]) -> tuple[int, SafetyResult, float]:
    if _WORKER_FILTER is None:
        raise RuntimeError("safety worker was not initialized")
    agent_id, states, nominal, constraints, active, enabled = job
    started = perf_counter()
    if enabled:
        result = _WORKER_FILTER.filter_with_constraints(agent_id, states, nominal, constraints, active)
    else:
        result = _WORKER_FILTER.diagnose_with_constraints(agent_id, states, nominal, constraints, active)
    return agent_id, result, perf_counter() - started


class SafetyExecutor:
    """Run stateless agent QPs serially or in a persistent spawn process pool."""

    def __init__(self, safety_filter: HOCBFSafetyFilter, backend: str, workers: int, parallel_min_agents: int) -> None:
        self.safety_filter = safety_filter
        self.backend = backend
        self.workers = int(workers) if workers else max(1, min(8, (os.cpu_count() or 2) - 1))
        self.parallel_min_agents = int(parallel_min_agents)
        self._pool: ProcessPoolExecutor | None = None
        self.last_backend = "serial"
        self.last_worker_time_sum = 0.0
        self.last_max_agent_time = 0.0

    def _use_processes(self, active_count: int) -> bool:
        return self.workers > 1 and self.backend != "serial" and (
            self.backend == "process" or active_count >= self.parallel_min_agents
        )

    def _ensure_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=mp.get_context("spawn"),
                initializer=_initialize_worker,
                initargs=(self.safety_filter,),
            )
        return self._pool

    def solve(self, states: Sequence[AgentState], nominal: np.ndarray, active: np.ndarray, enabled: bool) -> tuple[list[SafetyResult], int, int]:
        constraints = self.safety_filter.constraints_for_all(states, nominal, active)
        active_ids = np.flatnonzero(active).tolist()
        results = [SafetyResult(np.zeros(3), np.zeros(3)) for _ in states]
        jobs = [(i, states, nominal[i], constraints[i], active, enabled) for i in active_ids]
        if self._use_processes(len(active_ids)):
            solved = list(self._ensure_pool().map(_solve_job, jobs))
            self.last_backend = "process"
        else:
            solved = []
            for i in active_ids:
                started = perf_counter()
                if enabled:
                    result = self.safety_filter.filter_with_constraints(i, states, nominal[i], constraints[i], active)
                else:
                    result = self.safety_filter.diagnose_with_constraints(i, states, nominal[i], constraints[i], active)
                solved.append((i, result, perf_counter() - started))
            self.last_backend = "serial"
        durations = []
        for agent_id, result, elapsed in solved:
            results[agent_id] = result
            durations.append(elapsed)
        self.last_worker_time_sum = float(sum(durations))
        self.last_max_agent_time = float(max(durations, default=0.0))
        engaged_agents = sum(bool(item) for item in constraints)
        directed_constraints = sum(len(item) for item in constraints)
        return results, engaged_agents, directed_constraints // 2

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=False)
            self._pool = None

