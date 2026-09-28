"""Spawn-safe fixed-scenario evaluation for Windows and Unix hosts."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import atexit
import shutil
import tempfile
from typing import Any, Sequence

from uav_safe_marl.experiments.loading import load_trainer_checkpoint
from uav_safe_marl.runners.evaluator import evaluate
from .artifacts import combine_evaluation_results


_EVAL_TRAINER = None
_EVAL_METHOD = "model"
_EVAL_SEED = 0


def _initialize_evaluator(checkpoint: str, device: str, method: str, training_seed: int, config_overrides: dict[str, Any] | None) -> None:
    global _EVAL_TRAINER, _EVAL_METHOD, _EVAL_SEED
    worker_dir = Path(tempfile.mkdtemp(prefix="uav_safe_marl_eval_worker_"))
    atexit.register(shutil.rmtree, worker_dir, ignore_errors=True)
    _EVAL_TRAINER = load_trainer_checkpoint(checkpoint, worker_dir, device=device, nested_parallelism=False, config_overrides=config_overrides)
    _EVAL_METHOD, _EVAL_SEED = method, int(training_seed)


def _evaluate_one(item: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    if _EVAL_TRAINER is None:
        raise RuntimeError("evaluation worker was not initialized")
    index, scenario = item
    result = evaluate(_EVAL_TRAINER, episodes=1, reset_options={"scenario": scenario})
    for metrics in result["episode_metrics"]:
        metrics["method"] = _EVAL_METHOD
        metrics["training_seed"] = _EVAL_SEED
    return index, result


def evaluate_checkpoint_parallel(checkpoint: str | Path, scenarios: Sequence[dict[str, Any]], *, workers: int, method: str = "model", training_seed: int = 0, device: str = "cpu", config_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate exact scenarios in spawn workers and merge in bank order."""
    if workers < 1:
        raise ValueError("workers must be positive")
    if not scenarios:
        raise ValueError("scenario bank cannot be empty")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=_initialize_evaluator,
        initargs=(str(Path(checkpoint).resolve()), device, method, int(training_seed), config_overrides),
    ) as pool:
        indexed = list(pool.map(_evaluate_one, enumerate(scenarios)))
    ordered = [result for _, result in sorted(indexed, key=lambda item: item[0])]
    first = ordered[0]
    duration = float(first["episode_metrics"][0]["episode_duration"])
    steps = max(1, len(first["rollouts"][0]["rewards"]))
    return combine_evaluation_results(ordered, duration / steps)
