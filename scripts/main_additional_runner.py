"""F budget 1.5, B50 and paired legacy rollouts on the original main banks.

The engine package is deliberately unchanged so old checkpoint/source identity
continues to apply. New raw F/B runs live in separate child experiments.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
from time import perf_counter
from unittest.mock import patch
import zipfile

import numpy as np
import yaml

from scripts import main_notebook_runner as notebook
from scripts.main_incremental_checkpoint import IncrementalNotebookTrainer
from uav_safe_marl import build_env
from uav_safe_marl.evaluation import load_scenario_bank
from uav_safe_marl.evaluation.artifacts import combine_evaluation_results, save_evaluation
from uav_safe_marl.experiments import main
from uav_safe_marl.experiments.loading import load_trainer_checkpoint
from uav_safe_marl.policies.base import require_torch
from uav_safe_marl.runners.evaluator import evaluate
from uav_safe_marl.runners.trainer import SafeRLTrainer
from uav_safe_marl.runtime import runtime_manifest

NEW_METHODS = ("F", "B50")
ON_METHODS = ("A", "B75", "B50", "HBA", "F")
LEGACY_METHODS = ("A", "B75", "HBA", "C", "D75", "HBA_OFF", "E", "R")
ALL_METHODS = ("A", "B75", "B50", "HBA", "F", "C", "D75", "D50", "HBA_OFF", "F_OFF", "E", "R")
# Canonical method -> source experiment, raw training method, ON, old evaluation ID.
CONDITIONS = {
    "A": ("parent", "A", True, "A"), "C": ("parent", "A", False, "C"),
    "B75": ("parent", "B", True, "B"), "D75": ("parent", "B", False, "D"),
    "HBA": ("parent", "F", True, "F"), "HBA_OFF": ("parent", "F", False, "F_OFF"),
    "F": ("F", "F", True, "F"), "F_OFF": ("F", "F", False, "F_OFF"),
    "B50": ("B50", "B", True, "B"), "D50": ("B50", "B", False, "D"),
    "E": ("parent", "E", True, "E"), "R": ("parent", "R", False, "R"),
}


def implementation_hash():
    directory = Path(__file__).resolve().parent
    return main.digest({name: main.file_hash(directory / name) for name in (
        "main_additional_runner.py", "main_incremental_checkpoint.py", "main_notebook_runner.py")})


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _child_data(parent, method, target):
    data = deepcopy(parent)
    settings = data["settings"]
    settings["stage"] = f"additional_{method}"
    settings["online_steps"] = target
    settings["base_config"]["training"]["total_environment_steps_budget"] = target
    settings["evaluation_workers"] = 1
    if method == "F":
        settings["base_config"]["objective"]["intervention_cost_enabled"] = False
        settings["base_config"]["learning"]["d_cost"] = 1.5
    else:
        settings["base_config"]["communication"]["reactive_radius"] = 50.0
    data["settings_sha256"] = main.digest(settings)
    return data


def prepare_additional(parent, output, *, target=30000):
    parent, output = Path(parent).resolve(), Path(output).resolve()
    if not isinstance(target, int) or target < 1:
        raise ValueError("target must be a positive integer")
    if output == parent or output.is_relative_to(parent) or parent.is_relative_to(output):
        raise ValueError("追加実験の保存先は旧実験と別のディレクトリにしてください")
    original = main.context(parent)  # exact source and every original bank hash
    settings = original["settings"]
    if (settings["base_config"]["learning"]["d_cost"] != 6
            or not settings["base_config"]["objective"]["intervention_cost_enabled"]
            or settings["base_config"]["communication"]["reactive_radius"] != 75):
        raise ValueError("Parent must be the original A/B75/HBA experiment")
    expected = {"schema_version": 1, "parent_root": str(parent), "target": target,
        "parent_manifest_sha256": main.file_hash(parent / "experiment.json"),
        "source_sha256": original["source_sha256"], "implementation_sha256": implementation_hash(),
        "bank_hashes": original["bank_hashes"], "training_seeds": settings["training_seeds"],
        "development_ids": original["development_ids"], "conditions": CONDITIONS,
        "new_methods": list(NEW_METHODS)}
    # JSON normalisation also makes tuple/list registry identity stable.
    expected = json.loads(json.dumps(expected))
    if (output / "suite.json").is_file():
        if _read(output / "suite.json") != expected:
            raise ValueError("Additional experiment identity changed; choose a new output directory")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError("Incomplete/nonempty additional experiment directory")
    for method in NEW_METHODS:
        child = output / "experiments" / method
        data = _child_data(original, method, target)
        for name, digest in expected["bank_hashes"].items():
            source, destination = parent / "banks" / f"{name}.json", child / "banks" / f"{name}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copyfile(source, destination)
            if main.file_hash(destination) != digest:
                raise ValueError(f"Shared bank changed: {method}/{name}")
        if (child / "experiment.json").exists() and _read(child / "experiment.json") != data:
            raise ValueError(f"Child experiment changed: {method}")
        main.write(child / "experiment.json", data)
        setting_file = child / "settings.yaml"
        if setting_file.exists() and yaml.safe_load(setting_file.read_text(encoding="utf-8")) != data["settings"]:
            raise ValueError(f"Child settings changed: {method}")
        setting_file.write_text(yaml.safe_dump(data["settings"], allow_unicode=True, sort_keys=False), encoding="utf-8")
    # Retain exact original banks/provenance in the portable suite as well.
    for name in expected["bank_hashes"]:
        destination = output / "banks" / f"{name}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(parent / "banks" / f"{name}.json", destination)
        if main.file_hash(destination) != expected["bank_hashes"][name]:
            raise ValueError(f"Suite bank changed: {name}")
    if (output / "parent_experiment.json").exists() and _read(output / "parent_experiment.json") != original:
        raise ValueError("Parent snapshot changed")
    main.write(output / "parent_experiment.json", original)
    main.write(output / "suite.json", expected)
    return expected


def context(output):
    output = Path(output).resolve()
    data = _read(output / "suite.json")
    if data["implementation_sha256"] != implementation_hash():
        raise ValueError("Additional runner implementation changed; audited migration is required")
    parent = Path(data["parent_root"])
    if main.file_hash(parent / "experiment.json") != data["parent_manifest_sha256"]:
        raise ValueError("Original experiment manifest changed")
    original = main.context(parent)
    if _read(output / "parent_experiment.json") != original:
        raise ValueError("Parent snapshot changed")
    if data["source_sha256"] != original["source_sha256"] or data["conditions"] != json.loads(json.dumps(CONDITIONS)):
        raise ValueError("Additional source/condition registry differs")
    for name, expected in data["bank_hashes"].items():
        if main.file_hash(output / "banks" / f"{name}.json") != expected:
            raise ValueError(f"Suite bank changed: {name}")
    for method in NEW_METHODS:
        child = output / "experiments" / method
        if main.context(child) != _child_data(original, method, data["target"]):
            raise ValueError(f"Additional child changed: {method}")
    return data


def train_additional(output, *, methods=NEW_METHODS, progress_factory):
    output = Path(output).resolve()
    suite = context(output)
    if any(method not in NEW_METHODS for method in methods):
        raise ValueError("Only new F and B50 may be trained")
    target = suite["target"]
    finished = []
    with progress_factory(total=1, bar_format="{desc} {bar}{postfix}") as bar:
        for seed in suite["training_seeds"]:
            for method in methods:
                child = output / "experiments" / method
                raw = "F" if method == "F" else "B"
                manifest = main.prepare(child / "settings.yaml", child)
                run = child / "training" / raw / f"seed_{seed:03d}"
                evaluation = notebook._development_path(child, raw, seed, target)
                report = notebook._progress_reporter(bar, f"{method}/{seed}")

                def complete():
                    return notebook._run_complete(run, evaluation, manifest, raw, seed, target)

                # Compare fresh initial Actor/Encoder with the original A
                # warm-up audit when that evidence is available.
                class AuditedTrainer(IncrementalNotebookTrainer):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        fresh = main.frozen_hashes(self)
                        audit = Path(suite["parent_root"]) / "training" / "A" / f"seed_{seed:03d}" / "warmup/audit.json"
                        reference = _read(audit).get("frozen_before", {}).get("actor") if audit.is_file() else None
                        if reference is not None and fresh["actor"] != reference:
                            self.env.close()
                            raise ValueError("Initial Actor/Encoder differs from original A")
                        main.write(run / "initialization.json", {"seed": seed, "actor_sha256": fresh["actor"],
                            "original_A_actor_sha256": reference, "matched_original_audit": reference is not None})

                if not complete():
                    if not (run / "latest.pt").exists() and any(run.glob("stage_*.json")):
                        raise RuntimeError(f"{method}/{seed}: 再開用latest.ptがありません")
                    with patch.object(notebook, "NotebookTrainer", AuditedTrainer):
                        try:
                            notebook._run_with_progress(child, raw, seed, target, manifest, report, complete)
                        finally:
                            torch = require_torch()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                finished.append((method, seed))
                report("学習・development完了", len(manifest["development_ids"]), len(manifest["development_ids"]), "件")
    return finished


def _source_root(output, suite, method):
    source = CONDITIONS[method][0]
    return Path(suite["parent_root"]) if source == "parent" else output / "experiments" / source


def _baseline(output, suite, method):
    parent = Path(suite["parent_root"])
    existing = parent / "baselines" / f"{method}.pt"
    if existing.is_file():
        return existing
    checkpoint = output / "baselines" / f"{method}.pt"
    if not checkpoint.exists():
        rng = main.capture_rng()
        data = _read(output / "parent_experiment.json")
        cfg = main.config_for(data, method, 0, None)
        cfg.device = data["settings"]["evaluation_device"]
        cfg.monitoring.enabled = False
        trainer = None
        try:
            trainer = SafeRLTrainer(cfg, build_env(cfg), output / "baselines" / method)
            trainer.save_checkpoint(checkpoint, include_replay=False)
        finally:
            if trainer is not None:
                trainer.env.close()
            main.restore_rng(rng)
    return checkpoint


def _checkpoint(output, suite, method, seed):
    _, raw, _, _ = CONDITIONS[method]
    if method in ("E", "R"):
        return _baseline(output, suite, method)
    return _source_root(output, suite, method) / "training" / raw / f"seed_{seed:03d}" / f"model_{suite['target']}.pt"


def _validate_model(checkpoint, suite, method, seed):
    if not checkpoint.is_file():
        raise FileNotFoundError(f"評価用モデルがありません: {checkpoint}")
    payload = require_torch().load(checkpoint, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    if payload["experiment_identity"]["source_sha256"] != suite["source_sha256"]:
        raise ValueError(f"Checkpoint source differs: {method}/{seed}")
    if cfg["seed"] != seed:
        raise ValueError("Checkpoint training seed differs")
    if method in ("E", "R"):
        if (cfg["control"]["actor_mode"] != "reference_only" or cfg["communication"]["information_mode"] != "none"
                or cfg["safety"]["enabled"] != (method == "E")):
            raise ValueError("Invalid Reference baseline")
    else:
        if (not cfg["safety"]["enabled"] or cfg["control"]["actor_mode"] != "residual"
                or payload["trainer_progress"]["total_environment_steps"] < suite["target"]):
            raise ValueError("Checkpoint is not the ON-trained target model")
        prior = method in ("A", "C", "HBA", "HBA_OFF", "F", "F_OFF")
        expected_mode = "prior_share" if prior else "reactive_share"
        cost_enabled = method not in ("HBA", "HBA_OFF", "F", "F_OFF")
        budget = 1.5 if method in ("F", "F_OFF") else 6.0
        radius = 50.0 if method in ("B50", "D50") else 75.0
        if (cfg["communication"]["information_mode"] != expected_mode
                or (not prior and cfg["communication"].get("reactive_radius") != radius)
                or cfg["objective"]["intervention_cost_enabled"] != cost_enabled
                or cfg["learning"]["d_cost"] != budget):
            raise ValueError(f"Checkpoint method/budget/radius differs: {method}")
    original = _read(Path(suite["parent_root"]) / "experiment.json")
    origin, raw, _, _ = CONDITIONS[method]
    data = original if origin == "parent" else _child_data(original, origin, suite["target"])
    expected = main.config_for(data, raw, seed, None if method in ("E", "R") else suite["target"]).to_dict()
    for key in ("safety", "graph", "policy", "control", "environment", "scenario", "communication", "objective", "learning", "training"):
        if cfg[key] != expected[key]:
            raise ValueError(f"Checkpoint {key} differs from original main conditions")
    return payload


def _bank(output, suite, split, stochastic):
    if split not in ("development", "decision"):
        raise ValueError("Only development/decision are enabled; confirmation test stays closed")
    settings = _read(output / "parent_experiment.json")["settings"]
    bank = load_scenario_bank(output / "banks" / f"{split}.json")
    if split == "development":
        by_id = {s["scenario_id"]: s for s in bank}
        if len(set(suite["development_ids"])) != len(suite["development_ids"]):
            raise ValueError("Duplicate development ID")
        bank = [by_id[i] for i in suite["development_ids"]]
    if stochastic:
        bank = main.subset(bank, settings["calibration_per_n"])
    return bank


def _legacy_case(output, suite, method, seed, split, stochastic, scenario, repeat, identity):
    source = _source_root(output, suite, method)
    raw = CONDITIONS[method][3]
    mode = "stochastic" if stochastic else "mean"
    caseid = main.digest([scenario["scenario_id"], repeat])[:24]
    path = source / "evaluation" / split / f"step_{suite['target']}" / mode / raw / f"seed_{seed:03d}" / "cases" / caseid
    if not (path / "completed.json").is_file():
        return None
    case = _read(path / "completed.json")
    expected = {key: identity[key] for key in ("runtime_hocbf", "stochastic", "checkpoint_sha256", "source_sha256", "bank_sha256", "step_label")}
    expected.update(method=raw, training_method=CONDITIONS[method][1], training_seed=seed)
    if case.get("identity") != expected:
        return None
    # Never label an existing result as a newly executed Windows case. Its
    # execution provenance is retained explicitly when it can be adopted.
    hardware = _read(source / "experiment.json").get("hardware", {})
    if hardware.get("system") != identity["execution_system"]:
        return None
    if not (path / "rollouts/rollout_0000.npz").is_file():
        return None
    return path, case


def _validate_rollout(case, scenario):
    """Exact initial conditions, capabilities and candidate graph on every reuse."""
    with np.load(case / "rollouts/rollout_0000.npz", allow_pickle=False) as z:
        n = int(scenario["actual_agent_count"])
        if int(z["actual_agent_count"]) != n:
            raise ValueError("Rollout agent count differs from the common bank")
        for field, key in (("initial_positions", "positions"), ("initial_velocities", "velocities"),
                ("goals", "goals"), ("maximum_accelerations", "maximum_accelerations"),
                ("maximum_velocities", "maximum_velocities"), ("reference_route_lengths", "reference_route_lengths")):
            if not np.array_equal(z[field][:n], np.asarray(scenario[key])):
                raise ValueError(f"Rollout initial condition differs: {field}")
        candidate = np.zeros((n, n), dtype=bool)
        for i, j in scenario["predicted_los_edges"]:
            candidate[i, j] = candidate[j, i] = True
        if not np.array_equal(z["candidate_adjacency"][:n, :n], candidate):
            raise ValueError("Rollout candidate graph differs from the common bank")


def evaluate_condition(output, method, seed=0, *, split="development", stochastic=False, report=None):
    output = Path(output).resolve()
    suite = context(output)
    if method not in CONDITIONS:
        raise ValueError(f"Unknown additional condition: {method}")
    if stochastic and (method not in ON_METHODS or split != "decision"):
        raise ValueError("Calibration requires an ON-trained method and decision split")
    if method in ("E", "R"):
        seed = 0
    elif seed not in suite["training_seeds"]:
        raise ValueError("Unknown training seed")
    settings = _read(output / "parent_experiment.json")["settings"]
    bank = _bank(output, suite, split, stochastic)
    repeats = settings["calibration_repeats"] if stochastic else 1
    checkpoint = _checkpoint(output, suite, method, seed)
    payload = _validate_model(checkpoint, suite, method, seed)
    identity = {"method": method, "training_method": {"C":"A", "D75":"B75", "D50":"B50", "HBA_OFF":"HBA", "F_OFF":"F"}.get(method, method),
        "training_seed": seed, "runtime_hocbf": CONDITIONS[method][2], "stochastic": stochastic,
        "checkpoint_sha256": main.file_hash(checkpoint), "source_sha256": suite["source_sha256"],
        "bank_sha256": suite["bank_hashes"][split], "step_label": suite["target"],
        "learning_budget": payload["config"]["learning"]["d_cost"],
        "learning_intervention_cost": payload["config"]["objective"]["intervention_cost_enabled"],
        "communication_mode": payload["config"]["communication"]["information_mode"],
        "reactive_radius": payload["config"]["communication"].get("reactive_radius"),
        "execution_system": runtime_manifest()["system"],
        "runner_sha256": suite["implementation_sha256"]}
    del payload
    mode = "stochastic" if stochastic else "mean"
    destination = output / "evaluation" / split / f"step_{suite['target']}" / mode / method / f"seed_{seed:03d}"
    expected_count = len(bank) * repeats
    reporter = report or (lambda *args, **kwargs: None)
    reporter(f"{split} {mode}", 0, expected_count, "件")
    rows = []
    evaluator = None
    rng = main.capture_rng()
    started = perf_counter()
    try:
        for scenario in bank:
            for repeat in range(repeats):
                caseid = main.digest([scenario["scenario_id"], repeat])[:24]
                case = destination / "cases" / caseid
                completed = case / "completed.json"
                if completed.exists():
                    cached = _read(completed)
                    if cached["identity"] != identity:
                        raise ValueError(f"Cached evaluation identity changed: {method}/{seed}/{caseid}")
                    if not (case / "rollouts/rollout_0000.npz").is_file():
                        raise ValueError("Completed evaluation is missing its raw rollout")
                    row = cached["metrics"]
                else:
                    legacy = _legacy_case(output, suite, method, seed, split, stochastic, scenario, repeat, identity)
                    if legacy is not None:
                        old_path, old_case = legacy
                        shutil.copytree(old_path, case, dirs_exist_ok=True, ignore=shutil.ignore_patterns("completed.json", "*.tmp"))
                        row = dict(old_case["metrics"])
                        row.update(method=method, training_seed=seed, runtime_hocbf=identity["runtime_hocbf"])
                        _validate_rollout(case, scenario)
                        main.write(completed, {"identity": identity, "metrics": row,
                            "provenance": {"kind": "adopted", "original_case": str(old_path),
                                           "original_identity": old_case["identity"],
                                           "execution_system": identity["execution_system"]}})
                    else:
                        if evaluator is None:
                            evaluator = load_trainer_checkpoint(checkpoint, destination / "runtime",
                                device=settings["evaluation_device"],
                                config_overrides={"safety": {"enabled": identity["runtime_hocbf"]}})
                        key = main.seed_rollout(seed, scenario["scenario_id"], repeat)
                        result = evaluate(evaluator, reset_options={"scenario": scenario},
                            policy_execution_mode=mode, pair_initial_action_with_cost=stochastic)
                        row = result["episode_metrics"][0]
                        row.update(method=method, training_seed=seed, repeat=repeat, rollout_rng_key=key,
                            runtime_hocbf=identity["runtime_hocbf"], learning_budget=evaluator.config.learning.d_cost,
                            learning_intervention_cost=evaluator.config.objective.intervention_cost_enabled,
                            **main.rollout_diagnostics(result["rollouts"][0], evaluator.config))
                        if stochastic and not (row["paired_initial_action_matches_rollout"] and row["paired_initial_nominal_matches_rollout"]):
                            raise RuntimeError("Calibration initial action mismatch")
                        save_evaluation(combine_evaluation_results([result], evaluator.config.environment.dt_base), case)
                        _validate_rollout(case, scenario)
                        main.write(completed, {"identity": identity, "metrics": row,
                            "provenance": {"kind": "executed", "runtime": runtime_manifest()}})
                _validate_rollout(case, scenario)
                if (row["scenario_id"] != scenario["scenario_id"] or row["method"] != method
                        or row["learning_budget"] != identity["learning_budget"]
                        or row["learning_intervention_cost"] != identity["learning_intervention_cost"]):
                    raise ValueError("Case method, scenario or learning Cost differs")
                rows.append(row)
                reporter(f"{split} {mode}", len(rows), expected_count, "件")
        main.write(destination / "completed.json", {"identity": identity, "episodes": len(rows),
            "elapsed_seconds": perf_counter() - started})
    finally:
        if evaluator is not None:
            evaluator.env.close()
        main.restore_rng(rng)
    return rows


def evaluate_grid(output, *, methods=ALL_METHODS, split="development", stochastic=False, progress_factory):
    output = Path(output).resolve()
    suite = context(output)
    if any(m not in CONDITIONS for m in methods):
        raise ValueError("Unknown evaluation method")
    if stochastic and any(m not in ON_METHODS for m in methods):
        raise ValueError("Only ON conditions have formal calibration")
    tasks = [(m, s) for m in methods for s in ([0] if m in ("E", "R") else suite["training_seeds"])]
    missing = [str(_checkpoint(output, suite, m, s)) for m, s in tasks
               if m not in ("E", "R") and not _checkpoint(output, suite, m, s).is_file()]
    if missing:
        raise FileNotFoundError("評価モデルが不足しています。先に学習を完了するかPARENTを確認してください:\n" + "\n".join(sorted(set(missing))))
    result = []
    with progress_factory(total=1, bar_format="{desc} {bar}{postfix}") as bar:
        for method, seed in tasks:
            report = notebook._progress_reporter(bar, f"{method}/{seed}")
            rows = evaluate_condition(output, method, seed, split=split, stochastic=stochastic, report=report)
            result.append({"method": method, "seed": seed, "episodes": len(rows)})
    return result


def status(output):
    """Read partial results without requiring Windows model paths to exist."""
    output = Path(output)
    suite = _read(output / "suite.json")
    groups = {}
    for path in sorted((output / "evaluation").glob("*/*/*/*/seed_*/cases/*/completed.json")):
        case = _read(path)
        identity = case["identity"]
        key = (path.parents[6].name, "stochastic" if identity["stochastic"] else "mean", identity["method"], identity["training_seed"])
        groups.setdefault(key, []).append(case["metrics"])
    rows = []
    settings = _read(output / "parent_experiment.json")["settings"]
    for (split, mode, method, seed), rr in groups.items():
        expected = len(suite["development_ids"]) if split == "development" else sum(1 for _ in load_scenario_bank(output / "banks" / "decision.json"))
        if mode == "stochastic":
            expected = settings["calibration_per_n"] * len(settings["agent_counts"]) * settings["calibration_repeats"]
        rows.append({"split": split, "mode": mode, "method": method, "seed": seed,
            "episodes": len(rr), "expected": expected, "complete": len(rr) == expected,
            "successes": sum(r["episode_success"] for r in rr),
            "violation_episodes": sum(r["separation_violation_pair_ticks"] > 0 for r in rr)})
    dev = {(r["method"], r["seed"]): r for r in rows if r["split"] == "development" and r["mode"] == "mean"}
    expected = [(m, s) for m in ALL_METHODS for s in ([0] if m in ("E", "R") else suite["training_seeds"])]
    missing = [{"method": m, "seed": s} for m, s in expected if not dev.get((m, s), {}).get("complete", False)]
    main.write(output / "result_status.json", {"groups": rows, "missing_development": missing,
        "expected_development_cases": len(expected) * len(suite["development_ids"]),
        "completed_development_cases": sum(r["episodes"] for r in dev.values()),
        "note": "Missing groups have not been evaluated; E/R are fixed controls."})
    return rows


def export_results(output, destination=None, *, include_models=True):
    """Portable results archive; excludes Replay stores and resume checkpoints."""
    output = Path(output).resolve()
    suite = context(output)
    status(output)
    destination = Path(destination) if destination else output.with_name(output.name + "_results.zip")
    destination = destination.resolve()
    if destination.is_relative_to(output):
        raise ValueError("Export archive must be outside the live experiment directory")
    files = {}
    for name in ("suite.json", "parent_experiment.json", "result_status.json"):
        files[name] = output / name
    for folder in ("banks", "evaluation"):
        for path in (output / folder).rglob("*"):
            if path.is_file() and path.suffix in (".json", ".jsonl", ".npz") and "runtime" not in path.parts:
                files[str(path.relative_to(output))] = path
    for method in NEW_METHODS:
        child = output / "experiments" / method
        for path in child.rglob("*"):
            if (path.is_file() and path.suffix in (".json", ".jsonl", ".yaml")
                    and not any(part.endswith("_replay") for part in path.parts)):
                files[str(path.relative_to(output))] = path
    for method in ("A", "B75", "HBA"):
        source = _source_root(output, suite, method) / "training" / CONDITIONS[method][1]
        for path in source.rglob("*"):
            if path.is_file() and path.suffix in (".json", ".jsonl"):
                files[f"legacy_training/{method}/{path.relative_to(source)}"] = path
    if include_models:
        for method in ON_METHODS:
            for seed in suite["training_seeds"]:
                path = _checkpoint(output, suite, method, seed)
                if path.is_file():
                    _validate_model(path, suite, method, seed)
                    files[f"models/{method}/seed_{seed:03d}/model_{suite['target']}.pt"] = path
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    checksums = {}
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
            checksums[name] = {"sha256": main.file_hash(path), "bytes": path.stat().st_size}
        archive.writestr("export_manifest.json", json.dumps({"files": checksums,
            "conditions": CONDITIONS, "target": suite["target"],
            "note": "Models use canonical names; original HBA raw ID is F. Replay and full resume stores stay on Windows."}, ensure_ascii=False, indent=2))
    temporary.replace(destination)
    return destination
