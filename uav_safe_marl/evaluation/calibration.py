"""Frozen-checkpoint Cost Critic calibration against Monte Carlo rollouts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from uav_safe_marl.runners.evaluator import evaluate


def _frozen_parameters(trainer: Any) -> dict[str, dict[str, Any]]:
    state = trainer.backend.state_dict()
    return {
        group: {name: value.detach().cpu().clone() for name, value in state[group].items()}
        for group in ("actor", "cost_critics")
    }


def _assert_frozen(trainer: Any, before: dict[str, dict[str, Any]]) -> None:
    state = trainer.backend.state_dict()
    for group, parameters in before.items():
        after = state[group]
        if any(not np.array_equal(value.numpy(), after[name].detach().cpu().numpy()) for name, value in parameters.items()):
            raise RuntimeError(f"{group} parameters changed during frozen evaluation")


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    if not rows:
        raise ValueError("calibration rows cannot be empty")
    actual = np.asarray([row["discounted_episode_cost"] for row in rows], dtype=np.float64)
    mean = np.asarray([row["start_state_cost_mean"] for row in rows], dtype=np.float64)
    ucb = np.asarray([row["start_state_cost_ucb"] for row in rows], dtype=np.float64)
    correlation = float(np.corrcoef(mean, actual)[0, 1]) if len(rows) > 1 and np.std(mean) > 0 and np.std(actual) > 0 else None
    return {
        "count": len(rows),
        "mc_return_mean": float(actual.mean()),
        "mc_return_median": float(np.median(actual)),
        "mean_estimate_bias": float(np.mean(mean - actual)),
        "ucb_estimate_bias": float(np.mean(ucb - actual)),
        "mean_estimate_mae": float(np.mean(np.abs(mean - actual))),
        "mean_estimate_rmse": float(np.sqrt(np.mean((mean - actual) ** 2))),
        "ucb_estimate_mae": float(np.mean(np.abs(ucb - actual))),
        "correlation": correlation,
        "empirical_ucb_coverage": float(np.mean(actual <= ucb)),
        "mc_return_p75": float(np.quantile(actual, 0.75)),
        "mc_return_p90": float(np.quantile(actual, 0.90)),
        "mc_return_p95": float(np.quantile(actual, 0.95)),
        "mc_return_max": float(actual.max()),
    }


def calibrate_cost_critic(
    trainer: Any,
    scenarios: Iterable[dict[str, Any]],
    *,
    repeats: int = 1,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Calibrate a frozen Cost Critic under its learned continuation policy.

    For SAC, both the initial action and continuation actions are stochastic.
    The exact initial action evaluated by Q(s_0, a_0) is forced into the first
    rollout step; subsequent decision ticks sample normally. Runtime HOCBF is
    preserved. Mean-action deployment evaluation is intentionally separate.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")
    scenario_list = list(scenarios)
    if not scenario_list:
        raise ValueError("scenario bank cannot be empty")
    backend_name = trainer.config.policy.backend.upper()
    execution_mode = "stochastic" if backend_name == "SAC" else "mean"
    continuation_semantics = "stochastic_policy" if backend_name == "SAC" else "deterministic_policy"
    before = _frozen_parameters(trainer)
    rows: list[dict[str, Any]] = []
    for repeat in range(repeats):
        for scenario in scenario_list:
            result = evaluate(
                trainer,
                episodes=1,
                reset_options={"scenario": scenario},
                policy_execution_mode=execution_mode,
                pair_initial_action_with_cost=True,
            )
            row = dict(result["episode_metrics"][0])
            if not row["paired_initial_action_matches_rollout"] or not row["paired_initial_nominal_matches_rollout"]:
                raise RuntimeError("Cost estimate and rollout did not use the same initial action")
            row.update({
                "repeat": repeat,
                "training_seed": trainer.config.seed,
                "runtime_hocbf": trainer.config.safety.enabled,
                "calibration_kind": "cost_critic_policy_calibration",
                "critic_continuation_policy": continuation_semantics,
                "initial_action_conditioning": "same_sample_for_q_and_rollout",
                "execution_mode": execution_mode,
                "critic_rollout_semantics_match": True,
                "gamma_cost": trainer.config.learning.gamma_cost,
                "beta_ucb": trainer.config.learning.beta_ucb,
                "coverage_is_empirical_diagnostic": True,
            })
            rows.append(row)
    _assert_frozen(trainer, before)
    counts = sorted({int(row["actual_agent_count"]) for row in rows})
    result = {
        "definition": "frozen Cost Critic calibration with action-conditioned matched-policy Monte Carlo returns; coverage is empirical, not a confidence guarantee",
        "calibration_kind": "cost_critic_policy_calibration",
        "critic_continuation_policy": continuation_semantics,
        "initial_action_conditioning": "same_sample_for_q_and_rollout",
        "runtime_hocbf": trainer.config.safety.enabled,
        "execution_mode": execution_mode,
        "critic_rollout_semantics_match": True,
        "overall": _aggregate(rows),
        "by_agent_count": {
            str(count): _aggregate([row for row in rows if int(row["actual_agent_count"]) == count])
            for count in counts
        },
        "rows": rows,
    }
    if output_dir is not None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        summary = {key: value for key, value in result.items() if key != "rows"}
        (root / "cost_calibration_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        with (root / "cost_calibration.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return result


def evaluate_mean_policy_cost_diagnostic(
    trainer: Any,
    scenarios: Iterable[dict[str, Any]],
    *,
    repeats: int = 1,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate deterministic deployment cost without calling it calibration.

    SAC Cost Critics are trained for stochastic continuation, so their values
    are not compared with these all-mean-action returns as calibrated targets.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")
    scenario_list = list(scenarios)
    if not scenario_list:
        raise ValueError("scenario bank cannot be empty")
    before = _frozen_parameters(trainer)
    rows: list[dict[str, Any]] = []
    for repeat in range(repeats):
        for scenario in scenario_list:
            result = evaluate(
                trainer,
                episodes=1,
                reset_options={"scenario": scenario},
                policy_execution_mode="mean",
                pair_initial_action_with_cost=False,
            )
            episode = result["episode_metrics"][0]
            rows.append({
                "scenario_id": episode["scenario_id"],
                "actual_agent_count": episode["actual_agent_count"],
                "repeat": repeat,
                "training_seed": trainer.config.seed,
                "runtime_hocbf": trainer.config.safety.enabled,
                "execution_mode": "mean",
                "diagnostic_kind": "mean_policy_deployment_cost",
                "critic_rollout_semantics_match": trainer.config.policy.backend.upper() != "SAC",
                "undiscounted_episode_cost": episode["undiscounted_episode_cost"],
                "discounted_episode_cost": episode["discounted_episode_cost"],
                "episode_success": episode["episode_success"],
                "separation_violation_rate": episode["separation_violation_rate"],
                "pure_safety_intervention_magnitude": episode["pure_safety_intervention_magnitude"],
            })
    _assert_frozen(trainer, before)
    discounted = np.asarray([row["discounted_episode_cost"] for row in rows], dtype=np.float64)
    result = {
        "definition": "deterministic mean-action deployment diagnostic; not SAC Cost Critic calibration",
        "diagnostic_kind": "mean_policy_deployment_cost",
        "runtime_hocbf": trainer.config.safety.enabled,
        "execution_mode": "mean",
        "critic_rollout_semantics_match": trainer.config.policy.backend.upper() != "SAC",
        "count": len(rows),
        "discounted_episode_cost_mean": float(discounted.mean()),
        "discounted_episode_cost_median": float(np.median(discounted)),
        "rows": rows,
    }
    if output_dir is not None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        summary = {key: value for key, value in result.items() if key != "rows"}
        (root / "mean_policy_diagnostic_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        with (root / "mean_policy_diagnostic.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return result
