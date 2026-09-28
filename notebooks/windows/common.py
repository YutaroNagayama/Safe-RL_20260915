"""Shared Windows-notebook workflow for fixed-bank training and evaluation.

Keep imports at module scope limited to the standard library so that the first
notebook can create the project virtual environment before third-party modules
are required.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping


METHOD_LABELS = ("A", "B", "C", "D", "E")


def find_project_root(start: str | Path) -> Path:
    """Find the repository root from a notebook or VS Code working directory."""
    start_path = Path(start).resolve()
    for candidate in (start_path, *start_path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("pyproject.toml が見つかりません。VS Codeでリポジトリ全体を開いてください。")


def _venv_python(project_root: Path) -> Path:
    suffix = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    return project_root / ".venv" / suffix


def ensure_environment(project_root: str | Path) -> Path:
    """Create/install the local venv and require the matching notebook kernel."""
    root = Path(project_root).resolve()
    venv_root = root / ".venv"
    venv_python = _venv_python(root)
    marker = venv_root / ".uav_safe_marl_ready"
    if not venv_python.exists():
        creator = ["py", "-3.11"] if os.name == "nt" and shutil.which("py") else [sys.executable]
        subprocess.check_call([*creator, "-m", "venv", str(venv_root)])
    requirement_time = (root / "pyproject.toml").stat().st_mtime
    if not marker.exists() or marker.stat().st_mtime < requirement_time:
        subprocess.check_call([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
        torch_index = os.environ.get("UAV_TORCH_INDEX_URL", "").strip()
        if torch_index:
            subprocess.check_call([str(venv_python), "-m", "pip", "install", "torch", "--index-url", torch_index])
        subprocess.check_call([str(venv_python), "-m", "pip", "install", "-e", ".[learn,plot,dev]"], cwd=root)
        subprocess.check_call([
            str(venv_python), "-m", "ipykernel", "install", "--user",
            "--name", "uav-safe-marl", "--display-name", "UAV Safe MARL (.venv)",
        ])
        marker.write_text(json.dumps({"python": str(venv_python)}, indent=2), encoding="utf-8")
    if Path(sys.executable).resolve() != venv_python.resolve():
        raise RuntimeError(
            "初回セットアップは完了しました。VS Code右上のKernelから "
            "'UAV Safe MARL (.venv)' を選び、このBookを先頭から再実行してください。"
        )
    os.chdir(root)
    return venv_python


def load_settings(project_root: str | Path) -> dict[str, Any]:
    import yaml

    path = Path(project_root) / "notebooks" / "windows" / "experiment_settings.yaml"
    with path.open(encoding="utf-8") as stream:
        settings = yaml.safe_load(stream) or {}
    _validate_settings(settings)
    return settings


def deep_merge(*mappings: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mapping in mappings:
        for key, value in mapping.items():
            if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
    return result


def _validate_settings(settings: Mapping[str, Any]) -> None:
    scale = settings.get("experiment", {}).get("scale")
    if scale not in settings.get("profiles", {}):
        raise ValueError(f"Unknown experiment.scale: {scale!r}")
    methods = settings.get("methods", {})
    if tuple(methods) != METHOD_LABELS:
        raise ValueError(f"methods must be ordered exactly as {METHOD_LABELS}")
    profile = settings["profiles"][scale]
    if not profile.get("training_seeds") or not profile.get("agent_counts"):
        raise ValueError("The selected profile needs training_seeds and agent_counts")


def _scenario_kwargs(settings: Mapping[str, Any]) -> dict[str, Any]:
    scenario = settings["scenario"]
    base = settings["base_config"]
    safety = base["safety"]
    return {
        "base_size": scenario["longitudinal_length"],
        "transverse_size": scenario["transverse_size"],
        "spatial_scaling_mode": "fixed",
        "spatial_volume_scale_range": (1.0, 1.0),
        "altitude_range": (scenario["altitude_min"], scenario["altitude_max"]),
        "minimum_initial_distance": scenario["minimum_initial_separation"],
        "d_safe": safety["d_safe"],
        "d_warn": safety["d_warn"],
        "d_eng": safety["d_eng"],
        "los_prediction_threshold": scenario["interaction_threshold"],
        "prediction_horizon": scenario["prediction_horizon"],
        "initial_speed_fraction_range": (
            scenario["initial_speed_fraction_min"], scenario["initial_speed_fraction_max"]
        ),
        "acceleration_range": (scenario["acceleration_min"], scenario["acceleration_max"]),
        "velocity_limit_range": (scenario["velocity_limit_min"], scenario["velocity_limit_max"]),
        "require_predicted_conflict": scenario["require_predicted_candidate"],
    }


def _result_root(project_root: Path, settings: Mapping[str, Any]) -> Path:
    experiment = settings["experiment"]
    date = datetime.now().strftime("%Y%m%d")
    return project_root / experiment["results_root"] / f"result_{int(experiment['result_index'])}_{date}"


def _compute_max_steps(scenarios: list[dict[str, Any]], dt: float, multiplier: float) -> int:
    import numpy as np

    maximum_reference_time = max(
        float(np.max(
            np.asarray(item["reference_route_lengths"], dtype=float)
            / np.maximum(np.linalg.norm(np.asarray(item["velocities"], dtype=float), axis=1), 1e-6)
        ))
        for item in scenarios
    )
    return int(math.ceil(multiplier * maximum_reference_time / dt))


def _round_robin_by_agent_count(bank: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Shuffle within N, then interleave N strata to keep every prefix balanced."""
    import numpy as np

    rng = np.random.default_rng(seed)
    groups: dict[int, list[dict[str, Any]]] = {}
    for scenario in bank:
        groups.setdefault(int(scenario["actual_agent_count"]), []).append(scenario)
    for count, values in groups.items():
        groups[count] = [values[index] for index in rng.permutation(len(values))]
    ordered: list[dict[str, Any]] = []
    maximum = max(map(len, groups.values()), default=0)
    for index in range(maximum):
        for count in sorted(groups):
            if index < len(groups[count]):
                ordered.append(groups[count][index])
    return ordered


def create_or_load_experiment(project_root: str | Path) -> dict[str, Any]:
    """Create immutable shared scenario banks, or reuse today's existing bank."""
    import numpy as np
    from uav_safe_marl.evaluation import generate_random_traffic_bank, load_scenario_bank, save_scenario_bank

    root = Path(project_root).resolve()
    settings = load_settings(root)
    profile = settings["profiles"][settings["experiment"]["scale"]]
    result_root = _result_root(root, settings)
    shared = result_root / "shared"
    resolved_path = shared / "resolved_experiment.json"
    train_path = shared / "training_bank.json"
    eval_path = shared / "evaluation_bank.json"
    existing = resolved_path.is_file() and train_path.is_file() and eval_path.is_file()
    if existing and settings["experiment"].get("regenerate_scenarios", False):
        raise FileExistsError(
            f"{result_root} は既に存在します。scenarioを作り直す場合は result_index を増やしてください。"
        )
    if existing:
        training_bank = load_scenario_bank(train_path)
        evaluation_bank = load_scenario_bank(eval_path)
        context = json.loads(resolved_path.read_text(encoding="utf-8"))
        if int(context.get("schema_version", 0)) != 2:
            raise RuntimeError(
                "既存experimentは旧scenario供給順です。結果を保持したままresult_indexを増やし、"
                "00_generate_scenarios.ipynbで新しいbankを生成してください。"
            )
    else:
        shared.mkdir(parents=True, exist_ok=True)
        for label in METHOD_LABELS:
            (result_root / label).mkdir(parents=True, exist_ok=True)
        kwargs = _scenario_kwargs(settings)
        training_bank: list[dict[str, Any]] = []
        evaluation_bank: list[dict[str, Any]] = []
        for count in profile["agent_counts"]:
            training_bank.extend(generate_random_traffic_bank(
                int(count), int(profile["training_scenarios_per_n"]),
                seed=int(settings["scenario"]["seed_training"]) + int(count), **kwargs,
            ))
            evaluation_bank.extend(generate_random_traffic_bank(
                int(count), int(profile["evaluation_scenarios_per_n"]),
                seed=int(settings["scenario"]["seed_evaluation"]) + int(count), **kwargs,
            ))
        training_bank = _round_robin_by_agent_count(
            training_bank, int(settings["scenario"]["seed_training"])
        )
        dt = float(settings["base_config"]["environment"]["dt_base"])
        max_steps = _compute_max_steps(
            [*training_bank, *evaluation_bank], dt, float(settings["scenario"]["timeout_multiplier"])
        )
        save_scenario_bank(train_path, training_bank)
        save_scenario_bank(eval_path, evaluation_bank)
        context = {
            "schema_version": 2,
            "created_at": datetime.now().astimezone().isoformat(),
            "result_root": str(result_root.relative_to(root)),
            "selected_profile": settings["experiment"]["scale"],
            "profile": deepcopy(profile),
            "max_steps": max_steps,
            "training_episode_count": len(training_bank),
            "evaluation_episode_count": len(evaluation_bank),
            "settings": settings,
        }
        resolved_path.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
    pointer = root / "notebooks" / "windows" / "current_experiment.json"
    pointer.write_text(json.dumps({"result_root": str(result_root.relative_to(root))}, indent=2), encoding="utf-8")
    return {**context, "result_root_path": result_root, "training_bank": training_bank, "evaluation_bank": evaluation_bank}


def load_current_experiment(project_root: str | Path) -> dict[str, Any]:
    from uav_safe_marl.evaluation import load_scenario_bank

    root = Path(project_root).resolve()
    pointer_path = root / "notebooks" / "windows" / "current_experiment.json"
    if not pointer_path.is_file():
        raise FileNotFoundError("先に 00_generate_scenarios.ipynb を実行してください。")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    result_root = root / pointer["result_root"]
    context = json.loads((result_root / "shared" / "resolved_experiment.json").read_text(encoding="utf-8"))
    return {
        **context,
        "result_root_path": result_root,
        "training_bank": load_scenario_bank(result_root / "shared" / "training_bank.json"),
        "evaluation_bank": load_scenario_bank(result_root / "shared" / "evaluation_bank.json"),
    }


def resolved_method_config(project_root: str | Path, context: Mapping[str, Any], method: str, seed: int):
    from uav_safe_marl import load_config

    if method not in METHOD_LABELS:
        raise ValueError(f"Unknown method: {method}")
    settings = context["settings"]
    profile = context["profile"]
    overrides = deep_merge(
        settings["base_config"], settings["methods"][method]["overrides"],
        {
            "seed": int(seed),
            "device": settings["runtime"]["training_device"],
            "environment": {
                "agent_count": min(profile["agent_counts"]),
                "max_agent_count": max(profile["agent_counts"]),
                "max_steps": int(context["max_steps"]),
            },
            "scenario": {
                "training_agent_count_distribution": {
                    int(count): 1.0 / len(profile["agent_counts"])
                    for count in profile["agent_counts"]
                },
            },
            "monitoring": {
                "enabled": True, "progress_bar": True, "tensorboard": True,
                "auto_launch_tensorboard": False, "open_browser": False,
                "tensorboard_host": "127.0.0.1", "tensorboard_port": 0,
            },
        },
    )
    return load_config(Path(project_root) / "configs" / "full.yaml", overrides=overrides)


def method_summary(project_root: str | Path, method: str) -> dict[str, Any]:
    context = load_current_experiment(project_root)
    config = resolved_method_config(project_root, context, method, int(context["profile"]["training_seeds"][0]))
    return {
        "method": method,
        "name": context["settings"]["methods"][method]["name"],
        "result_root": str(context["result_root_path"]),
        "training_seeds": (
            context["profile"]["training_seeds"]
            if context["settings"]["methods"][method].get("train", True)
            else [context["profile"]["training_seeds"][0]]
        ),
        "training_scenario_bank_size": len(context["training_bank"]),
        "total_environment_steps_budget": config.training.total_environment_steps_budget,
        "evaluation_episodes_per_seed": len(context["evaluation_bank"]),
        "communication": config.communication.information_mode,
        "actor_representation": config.graph.actor_aggregation_backend,
        "critic_representation": config.graph.critic_aggregation_backend,
        "intervention_cost": config.objective.intervention_cost_enabled,
        "runtime_hocbf": config.safety.enabled,
        "actor_mode": config.control.actor_mode,
        "max_steps": config.environment.max_steps,
    }


def validate_experiment_stage(project_root: str | Path, expected_stage: str) -> dict[str, Any]:
    """Prevent a Pilot/Main notebook run against the wrong immutable snapshot."""
    if expected_stage not in {"smoke", "pilot", "main"}:
        raise ValueError("expected_stage must be smoke, pilot, or main")
    context = load_current_experiment(project_root)
    actual = context["selected_profile"]
    if actual != expected_stage:
        raise RuntimeError(
            f"このBookは {expected_stage!r} として実行しようとしていますが、現在のscenario bankは {actual!r} です。\n"
            "experiment_settings.yamlのscale/result_indexを確認し、00_generate_scenarios.ipynbを先に実行してください。"
        )
    budget = context["settings"]["base_config"]["training"].get("total_environment_steps_budget")
    if expected_stage in {"pilot", "main"} and budget is None:
        raise RuntimeError(
            f"{expected_stage}では total_environment_steps_budget が必須です。"
            "YAMLへ値を設定し、result_indexを増やして00を再実行してください。"
        )
    return {
        "stage": actual,
        "result_root": str(context["result_root_path"]),
        "environment_step_budget": budget,
        "training_seeds": context["profile"]["training_seeds"],
        "agent_counts": context["profile"]["agent_counts"],
        "training_bank_size": len(context["training_bank"]),
        "evaluation_bank_size": len(context["evaluation_bank"]),
    }


def training_diagnostic_report(project_root: str | Path, method: str = "A") -> dict[str, Any]:
    """Summarize Pilot-readiness signals without making a scientific go/no-go decision."""
    import math

    context = load_current_experiment(project_root)
    method_root = context["result_root_path"] / method
    expected_seeds = [int(value) for value in context["profile"]["training_seeds"]]
    expected_n = {int(value) for value in context["profile"]["agent_counts"]}
    budget = context["settings"]["base_config"]["training"].get("total_environment_steps_budget")
    reports: list[dict[str, Any]] = []
    warnings: list[str] = []
    diagnostic_keys = {
        "entropy_alpha", "log_alpha", "effective_target_entropy", "mean_policy_log_prob",
        "cost_critic_mean", "cost_critic_std", "cost_ucb_batch_mean",
        "fraction_cost_ucb_above_d_cost", "lambda_update_magnitude", "fraction_lambda_positive",
    }
    for seed in expected_seeds:
        seed_root = method_root / f"seed_{seed:03d}"
        rows = _read_jsonl(seed_root / "training" / "metrics.jsonl")
        completion_path = seed_root / "completed.json"
        completed = completion_path.is_file()
        if not rows:
            warnings.append(f"seed={seed}: training metricsがありません")
            reports.append({"seed": seed, "complete": completed, "episodes": 0})
            continue
        diagnostic_rows = [row for row in rows if diagnostic_keys <= row.keys()]
        alphas = [float(row["entropy_alpha"]) for row in diagnostic_rows]
        lambdas = [float(row["lambda"]) for row in rows if row.get("lambda") is not None]
        costs = [float(row["discounted_episode_cost"]) for row in rows]
        final = rows[-1]
        observed_n = {int(row["actual_agent_count"]) for row in rows}
        finite = all(math.isfinite(value) for value in [*alphas, *lambdas, *costs])
        if not completed:
            warnings.append(f"seed={seed}: completed.jsonがありません")
        if budget is not None and int(final["cumulative_environment_steps"]) < int(budget):
            warnings.append(f"seed={seed}: environment-step budgetへ未到達です")
        if not diagnostic_rows:
            warnings.append(f"seed={seed}: CAL/entropy診断値がまだありません（learning_starts未到達の可能性）")
        if observed_n != expected_n:
            warnings.append(f"seed={seed}: 観測N={sorted(observed_n)}、期待N={sorted(expected_n)}")
        if not finite:
            warnings.append(f"seed={seed}: 非有限の診断値があります")
        reports.append({
            "seed": seed,
            "complete": completed,
            "episodes": len(rows),
            "environment_steps": int(final["cumulative_environment_steps"]),
            "budget_overshoot": int(final.get("environment_steps_budget_overshoot", 0)),
            "gradient_updates": int(final["gradient_updates"]),
            "observed_n": sorted(observed_n),
            "alpha_first": alphas[0] if alphas else None,
            "alpha_final": alphas[-1] if alphas else None,
            "alpha_min": min(alphas) if alphas else None,
            "lambda_final": lambdas[-1] if lambdas else None,
            "fraction_lambda_positive": final.get("fraction_lambda_positive"),
            "fraction_cost_ucb_above_d_cost": final.get("fraction_cost_ucb_above_d_cost"),
            "discounted_episode_cost_mean": sum(costs) / len(costs),
            "discounted_episode_cost_max": max(costs),
            "episode_success_rate_training": sum(float(row["episode_success"]) for row in rows) / len(rows),
            "transitions_collected_by_n": {
                str(n): int(final.get(f"transitions_collected_N{n}", 0)) for n in sorted(expected_n)
            },
            "replay_occupancy_by_n": {
                str(n): int(final.get(f"replay_occupancy_N{n}", 0)) for n in sorted(expected_n)
            },
        })
    return {
        "stage": context["selected_profile"],
        "method": method,
        "result_root": str(context["result_root_path"]),
        "configured_environment_step_budget": budget,
        "mechanically_complete": not warnings,
        "warnings": warnings,
        "seed_reports": reports,
        "note": "mechanically_completeは収束・研究妥当性の自動判定ではありません。TensorBoardと評価指標を人が確認してください。",
    }


def _evaluation_workers(settings: Mapping[str, Any]) -> int:
    configured = int(settings["runtime"].get("evaluation_workers", 0))
    if configured > 0:
        return configured
    return max(1, min(8, (os.cpu_count() or 2) // 2))


def _launch_tensorboard(logdir: Path, open_browser: bool) -> tuple[Any, str]:
    from tensorboard import program
    import webbrowser

    server = program.TensorBoard()
    server.configure(argv=[None, "--logdir", str(logdir), "--host", "127.0.0.1", "--port", "0"])
    url = server.launch()
    if open_browser:
        webbrowser.open(url, new=2)
    return server, url


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_reproducibility_snapshot(root: Path, shared: Path) -> None:
    """Record exact dependencies and source hashes without copying result data."""
    tracked: list[dict[str, Any]] = []
    roots = [root / "uav_safe_marl", root / "configs", root / "notebooks" / "windows", root / "tests"]
    for source_root in roots:
        if not source_root.exists():
            continue
        for path in sorted(source_root.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".ipynb"}:
                tracked.append({"path": str(path.relative_to(root)), "sha256": _sha256(path)})
    for path in (root / "pyproject.toml",):
        if path.is_file():
            tracked.append({"path": str(path.relative_to(root)), "sha256": _sha256(path)})
    aggregate = hashlib.sha256(json.dumps(tracked, sort_keys=True).encode()).hexdigest()
    (shared / "source_manifest.json").write_text(
        json.dumps({"aggregate_sha256": aggregate, "files": tracked}, indent=2), encoding="utf-8"
    )
    try:
        frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    except (OSError, subprocess.CalledProcessError):
        frozen = "# pip freeze unavailable\n"
    (shared / "environment_freeze.txt").write_text(frozen, encoding="utf-8")


def run_method(project_root: str | Path, method: str, run: bool = True) -> dict[str, Any]:
    """Train and evaluate one method against the current immutable banks."""
    from uav_safe_marl import build_env
    from uav_safe_marl.evaluation import evaluate_checkpoint_parallel, save_evaluation
    from uav_safe_marl.evaluation.statistics import aggregate_primary_by_method_and_n
    from uav_safe_marl.runners.trainer import SafeRLTrainer
    from uav_safe_marl.runtime import runtime_manifest

    root = Path(project_root).resolve()
    context = load_current_experiment(root)
    if int(context.get("schema_version", 0)) != 2:
        raise RuntimeError(
            "旧scenario bankでは新規学習を開始できません。result_indexを増やし、"
            "00_generate_scenarios.ipynbでround-robin bankを生成してください。"
        )
    settings = context["settings"]
    if not run:
        return {"status": "disabled", **method_summary(root, method)}
    method_root = context["result_root_path"] / method
    method_root.mkdir(parents=True, exist_ok=True)
    training_bank_path = context["result_root_path"] / "shared" / "training_bank.json"
    evaluation_bank_path = context["result_root_path"] / "shared" / "evaluation_bank.json"
    input_reference = {
        "method": method,
        "name": settings["methods"][method]["name"],
        "resolved_experiment": "../shared/resolved_experiment.json",
        "training_bank": "../shared/training_bank.json",
        "training_bank_sha256": _sha256(training_bank_path),
        "evaluation_bank": "../shared/evaluation_bank.json",
        "evaluation_bank_sha256": _sha256(evaluation_bank_path),
    }
    _write_reproducibility_snapshot(root, context["result_root_path"] / "shared")
    (method_root / "input_reference.json").write_text(
        json.dumps(input_reference, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tensorboard_server, tensorboard_url = _launch_tensorboard(
        method_root, bool(settings["runtime"].get("open_tensorboard_browser", True))
    )
    seed_reports = []
    trainable = bool(settings["methods"][method].get("train", True))
    method_seeds = context["profile"]["training_seeds"] if trainable else [context["profile"]["training_seeds"][0]]
    for seed in method_seeds:
        seed = int(seed)
        seed_root = method_root / f"seed_{seed:03d}"
        completion = seed_root / "completed.json"
        if completion.is_file() and settings["experiment"].get("skip_completed_seeds", True):
            seed_reports.append(json.loads(completion.read_text(encoding="utf-8")))
            print(f"[{method}] seed={seed}: completed.json があるためskip")
            continue
        latest_checkpoint = seed_root / "checkpoints" / "latest.pt"
        if seed_root.is_dir() and any(seed_root.iterdir()) and not latest_checkpoint.is_file():
            raise RuntimeError(
                f"未完了の出力が残っています: {seed_root}\n"
                "ログを混在させないため自動上書きしません。内容を退避するか result_index を増やしてください。"
            )
        seed_root.mkdir(parents=True, exist_ok=True)
        config = resolved_method_config(root, context, method, seed)
        (seed_root / "resolved_config.json").write_text(
            json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n=== {method}: training seed {seed} ===")
        trainer = SafeRLTrainer(config, build_env(config), output_dir=seed_root / "training")
        if latest_checkpoint.is_file():
            print(f"[{method}] seed={seed}: latest checkpointからtrainingを再開")
            trainer.load_checkpoint(latest_checkpoint, resume_training=True)
            trainer.logger.truncate_after_episode(trainer.completed_episodes)
        if trainable:
            episode_limit = None if config.training.total_environment_steps_budget is not None else len(context["training_bank"])
            trainer.train(
                episode_limit, scenarios=context["training_bank"], live_display=True,
                checkpoint_path=latest_checkpoint,
            )
        else:
            print(f"[{method}] fixed controller baseline: gradient trainingを行いません")
        checkpoint = seed_root / "checkpoints" / "final.pt"
        trainer.save_checkpoint(latest_checkpoint)
        trainer.save_checkpoint(checkpoint)
        trainer.env.close()
        print(f"=== {method}: evaluation seed {seed} ===")
        result = evaluate_checkpoint_parallel(
            checkpoint, context["evaluation_bank"], workers=_evaluation_workers(settings),
            method=method, training_seed=seed, device=settings["runtime"]["evaluation_device"],
        )
        save_evaluation(result, seed_root / "evaluation")
        hocbf_off_primary = None
        if bool(settings["runtime"].get("evaluate_hocbf_off_diagnostic", True)) and config.safety.enabled and config.control.actor_mode == "residual":
            print(f"=== {method}: same-checkpoint HOCBF-OFF diagnostic seed {seed} ===")
            diagnostic = evaluate_checkpoint_parallel(
                checkpoint, context["evaluation_bank"], workers=_evaluation_workers(settings),
                method=f"{method}_hocbf_off", training_seed=seed,
                device=settings["runtime"]["evaluation_device"],
                config_overrides={"safety": {"enabled": False}},
            )
            save_evaluation(diagnostic, seed_root / "evaluation_hocbf_off")
            hocbf_off_primary = diagnostic["primary_metrics"]
        report = {
            "status": "complete", "method": method, "training_seed": seed,
            "checkpoint": str(checkpoint.relative_to(context["result_root_path"])),
            "primary_metrics": result["primary_metrics"],
            "same_checkpoint_hocbf_off_primary_metrics": hocbf_off_primary,
            "training_progress": {
                "total_environment_steps_budget": config.training.total_environment_steps_budget,
                "total_environment_steps": trainer.total_environment_steps,
                "environment_steps_budget_overshoot": (
                    0 if config.training.total_environment_steps_budget is None else
                    max(0, trainer.total_environment_steps - config.training.total_environment_steps_budget)
                ),
                "gradient_updates": trainer.total_gradient_updates,
                "completed_episodes": trainer.completed_episodes,
                "transitions_collected_by_n": trainer.transitions_collected_by_n,
                "replay_occupancy_by_n": trainer.buffer.occupancy_by_agent_count(),
            },
            "hardware": runtime_manifest(), "completed_at": datetime.now().astimezone().isoformat(),
        }
        completion.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        seed_reports.append(report)
    records: list[dict[str, Any]] = []
    for seed in method_seeds:
        records.extend(_read_jsonl(method_root / f"seed_{int(seed):03d}" / "evaluation" / "episode_metrics.jsonl"))
    aggregate = aggregate_primary_by_method_and_n(records) if records else {}
    summary = {
        "method": method, "name": settings["methods"][method]["name"],
        "tensorboard_url": tensorboard_url, "seed_reports": seed_reports,
        "aggregate": aggregate,
    }
    (method_root / "method_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    # Retain the server object for the lifetime of the running notebook kernel.
    summary["_tensorboard_server"] = tensorboard_server
    return summary
