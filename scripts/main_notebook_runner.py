"""Run the main grid in the Notebook kernel and display its live progress."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from unittest.mock import patch
from weakref import WeakMethod, proxy

import yaml

from uav_safe_marl import load_config
from uav_safe_marl.experiments import main
from uav_safe_marl.monitoring.live import LiveTrainingMonitor
from uav_safe_marl.policies.base import require_torch, resolve_device
from uav_safe_marl.runners import trainer as trainer_module


def probe_training_device(settings_path):
    """Resolve the device in the same kernel that will train."""
    settings = yaml.safe_load(Path(settings_path).read_text(encoding="utf-8"))
    requested = load_config(overrides=settings["base_config"]).device
    torch = require_torch()
    device = torch.zeros(1, device=resolve_device(requested)).device
    return {
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "requested_device": requested,
        "training_device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "evaluation_device": settings["evaluation_device"],
    }


class NotebookTrainer(trainer_module.SafeRLTrainer):
    """Attach display callbacks to the existing trainer without changing its config."""

    def __init__(self, config, env, output_dir, *, report, warmup_updates):
        super().__init__(config, env, output_dir)
        self.report = report
        self.collecting = Path(output_dir).name == "collection"
        self.warmup_updates = 0
        self.warmup_goal = warmup_updates
        self.collection_goal = None
        self.saved_collection = None
        trainer = proxy(self)
        log_episode = WeakMethod(self.logger.log)

        def track_episode(summary):
            log_episode()(summary)
            steps = summary["cumulative_environment_steps"]
            episodes = summary["completed_episodes"]
            if trainer.collecting:
                report("warm-up収集", episodes, trainer.collection_goal, "ep", f"{steps:,} step")
            else:
                report("online学習", steps, summary["total_environment_steps_budget"], "step",
                       f"ep {episodes:,}")

        self.logger.log = track_episode
        update_cost = getattr(self.backend, "update_cost_critic_only_n_step", None)
        if self.collecting and update_cost is not None:
            update_cost = WeakMethod(update_cost)
            def track_cost_update(batch):
                if trainer.warmup_updates == 0:
                    trainer.show_cost_progress()
                result = update_cost()(batch)
                trainer.warmup_updates += 1
                completed = trainer.warmup_updates
                if completed == 1 or completed % 25 == 0 or completed == warmup_updates:
                    trainer.show_cost_progress()
                return result

            self.backend.update_cost_critic_only_n_step = track_cost_update

    def show_cost_progress(self):
        self.report("Cost Critic warm-up", self.warmup_updates, self.warmup_goal, "更新")

    def load_checkpoint(self, path, **kwargs):
        self.report("保存点読込中", detail=Path(path).name)
        metadata = super().load_checkpoint(path, **kwargs)
        if self.collecting and Path(path).name == "collection.pt":
            self.saved_collection = self._collection_state(path)
        if self.collecting and "main_warmup_updates" in metadata:
            self.warmup_updates = metadata["main_warmup_updates"]
            self.show_cost_progress()
        return metadata

    def _collection_state(self, path):
        return Path(path), self.completed_episodes, self.total_environment_steps, self.scenario_bank_hash

    def save_checkpoint(self, path, **kwargs):
        collection = self.collecting and Path(path).name == "collection.pt"
        # The same collection state can already be durable after a load or periodic save.
        state = self._collection_state(path) if collection else None
        if collection and self.saved_collection == state:
            return
        self.report(detail=f"保存中 {Path(path).name}")
        super().save_checkpoint(path, **kwargs)
        if collection:
            self.saved_collection = state
        self.report(detail="")

    def train(self, *args, **kwargs):
        if self.collecting:
            self.collection_goal = len(kwargs["scenarios"])
            self.report("warm-up収集", self.completed_episodes, self.collection_goal, "ep",
                        f"{self.total_environment_steps:,} step")
        else:
            self.report("online学習", self.total_environment_steps, self.config.training.total_environment_steps_budget,
                        "step", f"ep {self.completed_episodes:,}")

        monitors = []

        class NotebookMonitor(LiveTrainingMonitor):
            def start(monitor, *start_args, **start_kwargs):
                monitors.append(monitor)
                # Use the Notebook's step bar and retain the shared TensorBoard writer.
                monitor.config = replace(monitor.config, monitoring=replace(monitor.config.monitoring, progress_bar=False))
                super().start(*start_args, **start_kwargs)

            def close(monitor):
                super().close()
                monitor.writer = None

        with patch.object(trainer_module, "LiveTrainingMonitor", NotebookMonitor):
            try:
                return super().train(*args, **kwargs)
            finally:
                for monitor in monitors:
                    monitor.close()


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _development_path(output, method, seed, target):
    return output / "evaluation" / "development" / f"step_{target}" / "mean" / method / f"seed_{seed:03d}"


def _run_complete(run, evaluation, manifest, method, seed, target):
    stage = _read_json(run / f"stage_{target}.json")
    result = _read_json(evaluation / "completed.json")
    identity = result.get("identity", {})
    model = run / f"model_{target}.pt"
    return (
        stage.get("target") == target and stage.get("actual_steps", -1) >= target
        and stage.get("settings_sha256") == manifest["settings_sha256"]
        and stage.get("source_sha256") == manifest["source_sha256"]
        and result.get("episodes") == len(manifest["development_ids"])
        and identity.get("method") == method and identity.get("training_seed") == seed
        and identity.get("step_label") == target
        and identity.get("runtime_hocbf") is True and identity.get("stochastic") is False
        and identity.get("source_sha256") == manifest["source_sha256"]
        and identity.get("bank_sha256") == manifest["bank_hashes"]["development"]
        and model.is_file() and identity.get("checkpoint_sha256") == main.file_hash(model)
    )


def _progress_reporter(bar, run_label):
    phase = "準備中"
    count, total, unit, current_detail = 0, 1, "", ""

    def report(label=None, value=0, goal=1, units="", detail=""):
        nonlocal phase, count, total, unit, current_detail
        if label is None:
            shown_detail = " · ".join(filter(None, (current_detail, detail)))
        else:
            phase, count, total, unit, current_detail = label, value, goal, units, detail
            shown_detail = detail
        if unit == "step" and count > total:
            shown_detail = " · ".join(filter(None, (shown_detail, f"実績 {count:,} step")))
        desc = f"{run_label} · {phase}"
        current = min(count, total)
        bar.set_postfix_str(shown_detail, refresh=False)
        if bar.desc != desc or bar.total != total or bar.unit != unit or current < bar.n:
            bar.set_description_str(desc, refresh=False)
            bar.unit = unit
            bar.bar_format = "{desc} {bar} {n:,}/{total:,} {unit}{postfix}" if unit else "{desc} {bar}{postfix}"
            bar.reset(total=total)
        if current != bar.n:
            bar.update(current - bar.n)
        bar.refresh()

    return report


def _run_with_progress(output, method, seed, target, settings, expected, report, complete):
    class RunTrainer(NotebookTrainer):
        def __init__(self, config, env, output_dir):
            super().__init__(config, env, output_dir, report=report,
                             warmup_updates=settings["cost_warmup_updates"])

    evaluate_condition = main.evaluate_condition
    worker_case = main._worker_case

    def evaluate(*args, **kwargs):
        step = kwargs["checkpoint_step"]
        cases = 0
        detail = f"{step:,} stepモデル"
        report("development", cases, expected, "件", detail)

        def track_case(item):
            nonlocal cases
            row = worker_case(item)
            cases += 1
            report("development", cases, expected, "件", detail)
            return row

        try:
            case_function = track_case if settings.get("evaluation_workers", 1) == 1 else worker_case
            with patch.object(main, "_worker_case", case_function):
                rows = evaluate_condition(*args, **kwargs)
        finally:
            main._EVALUATOR = None
        report("development", len(rows), expected, "件", detail)
        return rows

    with patch.object(main, "SafeRLTrainer", RunTrainer), patch.object(main, "evaluate_condition", evaluate):
        main.train_method(output, method, seed, target)
        if not complete():
            evaluate(output, method, seed, split="development", checkpoint_step=target)
            if not complete():
                raise RuntimeError(f"{method}/{seed}: 学習・development完了記録を確認できません")


def run_training_grid(settings_path, output, *, target=30000, progress_factory):
    """Train and evaluate each method/seed before starting the next."""
    if not isinstance(target, int) or target < 1:
        raise ValueError("target must be a positive integer")
    output = Path(output).resolve()
    manifest = main.prepare(Path(settings_path), output)
    settings = manifest["settings"]
    expected = len(manifest["development_ids"])
    finished = []
    with progress_factory(total=1, bar_format="{desc} {bar}{postfix}") as bar:
        for seed in settings["training_seeds"]:
            for method in main.TRAINED:
                run = output / "training" / method / f"seed_{seed:03d}"
                evaluation = _development_path(output, method, seed, target)

                def complete():
                    return _run_complete(run, evaluation, manifest, method, seed, target)

                report = _progress_reporter(bar, f"{method}/{seed}")
                if not complete():
                    if not (run / "latest.pt").exists() and any(run.glob("stage_*.json")):
                        raise RuntimeError(f"{method}/{seed}: latest.pt がありません")
                    report("準備中")
                    try:
                        _run_with_progress(output, method, seed, target, settings, expected, report, complete)
                    finally:
                        torch = require_torch()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                finished.append((method, seed))
                report("development完了", expected, expected, "件")
    return finished
