"""Run the main grid in the Notebook kernel and display its live progress."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch
from weakref import WeakMethod, proxy

import numpy as np
import yaml

from uav_safe_marl import build_env, load_config
from uav_safe_marl.evaluation import load_scenario_bank
from uav_safe_marl.graph.observations import stack_graph_observations
from uav_safe_marl.metrics.logger import ExperimentLogger
from uav_safe_marl.experiments import main
from uav_safe_marl.monitoring.live import LiveTrainingMonitor
from uav_safe_marl.policies.base import require_torch, resolve_device
from uav_safe_marl.runners import trainer as trainer_module


def _collate_cost_starts(selected, dtype):
    """Keep the original padding and fields used by SAC's n-step Cost target."""
    state = [item.state for item in selected]
    nominal = [item.nominal_action for item in selected]
    held = [item.held_residual_action for item in selected]
    if isinstance(state[0], dict):
        width = max(max(item.state["node_features"].shape[0], item.next_state["node_features"].shape[0])
                    for item in selected)
        state = stack_graph_observations(state, width)

        def pad(actions):
            batch = np.zeros((len(actions), width, 3), dtype=dtype)
            for index, action in enumerate(actions):
                batch[index, :len(action)] = action
            return batch
    else:
        state = np.stack(state)
        pad = np.stack
    return {"state": state, "nominal_action": pad(nominal), "held_residual_action": pad(held),
            "actor_phase": np.asarray([item.actor_phase for item in selected], dtype=dtype)[:, None]}


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
        self.last_online_save = None
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
        if config.policy.backend.upper() == "SAC" and config.learning.cost_target_n_step > 1:
            replay = proxy(self.buffer)
            sample_n_step = WeakMethod(self.buffer.sample_n_step)

            def sample_cost_batch(*args, **kwargs):
                with patch.object(replay, "_collate", lambda items: _collate_cost_starts(items, replay.storage_dtype)):
                    return sample_n_step()(*args, **kwargs)

            self.buffer.sample_n_step = sample_cost_batch
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
        path = Path(path)
        self.report("保存点読込中", detail=path.name)
        started = perf_counter()
        metadata = super().load_checkpoint(path, **kwargs)
        self.record_io(path, started, "load")
        if self.collecting and path.name == "collection.pt":
            self.saved_collection = self._collection_state(path)
        if self.collecting and "main_warmup_updates" in metadata:
            self.warmup_updates = metadata["main_warmup_updates"]
            self.show_cost_progress()
        return metadata

    def _collection_state(self, path):
        return Path(path), self.completed_episodes, self.total_environment_steps, self.scenario_bank_hash

    def save_checkpoint(self, path, **kwargs):
        path = Path(path)
        collection = self.collecting and path.name == "collection.pt"
        # The same collection state can already be durable after a load or periodic save.
        state = self._collection_state(path) if collection else None
        if collection and self.saved_collection == state:
            return
        online_state = (self.total_environment_steps, self.config.training.total_environment_steps_budget) if path.name == "latest.pt" else None
        if online_state is not None and self.last_online_save == online_state:
            return
        self.report(detail=f"保存中 {path.name}")
        started = perf_counter()
        super().save_checkpoint(path, **kwargs)
        self.last_online_save = online_state
        if collection:
            self.saved_collection = state
        self.record_io(path, started)
        self.report(detail="")

    def record_io(self, path, started, operation="save"):
        try:
            record = {"operation": operation, "file": path.name, "bytes": path.stat().st_size,
                      "seconds": round(perf_counter() - started, 3)}
            with (path.parent / "checkpoint_io.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
        except OSError as exc:
            print(f"チェックポイント計測ログを書き込めません: {exc}", file=sys.stderr)

    def begin_online(self, output):
        self.collecting = False
        self.total_environment_steps = self.total_gradient_updates = self.completed_episodes = 0
        self.transitions_collected_by_n = {}
        self.scenario_bank_cursor = 0
        self.scenario_bank_hash = None
        # Keep the logger object so its episode callback survives the handoff.
        self.logger.output_dir = ExperimentLogger(output, self.config).output_dir
        self.logger.truncate_after_episode(0)

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

            def update_step(monitor, episode_offset, step, max_steps, environment_steps,
                            updates, replay_size, actual_n):
                super().update_step(episode_offset, step, max_steps, environment_steps,
                                    updates, replay_size, actual_n)
                if (step + 1) % self.config.monitoring.step_refresh_interval:
                    return
                detail = f"N={actual_n} · ep内 {step + 1:,}/{max_steps:,} step"
                if self.collecting:
                    self.report("warm-up収集", episode_offset, self.collection_goal, "ep", detail)
                else:
                    self.report("online学習", environment_steps, self.config.training.total_environment_steps_budget,
                                "step", f"完了ep {episode_offset:,} · {detail}")

        with patch.object(trainer_module, "LiveTrainingMonitor", NotebookMonitor):
            try:
                return super().train(*args, **kwargs)
            finally:
                for monitor in monitors:
                    monitor.close()


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _base_hash(trainer, path):
    trainer.report(detail=f"照合中 {path.name}")
    started = perf_counter()
    result = main.file_hash(path)
    trainer.record_io(path, started, "hash")
    trainer.report(detail="")
    return result


def _warmup(trainer, output, warmup, updates):
    final = warmup / "full.pt"
    if final.is_file():
        trainer.load_checkpoint(final)
        return

    delta = warmup / "cost_resume.pt"
    legacy = warmup / "progress.pt"
    collection = warmup / "collection.pt"
    base = legacy if legacy.is_file() else collection
    completed = 0
    if delta.is_file():
        trainer.report("保存点読込中", detail=delta.name)
        started = perf_counter()
        payload = require_torch().load(delta, map_location=trainer.backend.device, weights_only=False)
        trainer.record_io(delta, started, "load")
        metadata = payload["extra_metadata"]
        if (metadata.get("format") != "cost_resume_v1"
                or payload["schema_version"] != trainer.CHECKPOINT_SCHEMA_VERSION
                or metadata["base_file"] not in {"progress.pt", "collection.pt"}):
            raise ValueError("Invalid Cost Critic base checkpoint")
        base = warmup / metadata["base_file"]
        base_digest = _base_hash(trainer, base)
        if metadata["base_sha256"] != base_digest:
            raise ValueError("Cost Critic base checkpoint hash differs")
        trainer.load_checkpoint(base)
        if (payload["experiment_identity"]["source_sha256"] != trainer.source_hash
                or payload["experiment_identity"]["config_sha256"] != main.digest(trainer.config.to_dict())
                or payload["experiment_identity"]["scenario_bank_sha256"] != trainer.scenario_bank_hash
                or payload["trainer_progress"]["completed_episodes"] != trainer.completed_episodes
                or payload["trainer_progress"]["total_environment_steps"] != trainer.total_environment_steps):
            raise ValueError("Cost Critic checkpoint identity differs from the restored collection")
        trainer.backend.load_state_dict(payload["backend"])
        trainer.buffer.load_sampling_state_dict(metadata["replay_rng"])
        initial = trainer.initial_state_buffer.state_dict()
        initial["rng_state"] = metadata["initial_rng"]
        trainer.initial_state_buffer.load_state_dict(initial)
        rng = payload["rng_state"]
        main.restore_rng((rng["python"], rng["numpy_global"], rng["torch_cpu"].cpu(),
                          [state.cpu() for state in rng["torch_cuda"]]
                          if rng["torch_cuda"] is not None and require_torch().cuda.is_available() else None))
        completed = metadata["main_warmup_updates"]
        del payload
    elif base == legacy:
        completed = trainer.load_checkpoint(legacy)["main_warmup_updates"]
        base_digest = _base_hash(trainer, base) if completed < updates else None
    else:
        bank = load_scenario_bank(output / "banks" / "warmup.json")
        if collection.is_file():
            trainer.load_checkpoint(collection)
        trainer.logger.truncate_after_episode(trainer.completed_episodes)
        trainer.train(max(0, len(bank) - trainer.completed_episodes), scenarios=bank,
                      update_mode="collect_only", live_display=False, checkpoint_path=collection)
        trainer.save_checkpoint(collection)
        base_digest = _base_hash(trainer, base) if updates else None

    if not 0 <= completed <= updates:
        raise ValueError("Invalid Cost Critic update count")
    trainer.warmup_updates = completed
    trainer.show_cost_progress()
    before = main.frozen_hashes(trainer)
    cfg = trainer.config
    for index in range(completed + 1, updates + 1):
        batch = trainer.buffer.sample_n_step(cfg.learning.batch_size, cfg.learning.cost_target_n_step,
                                             cfg.learning.gamma_cost)
        trainer.backend.update_cost_critic_only_n_step(batch)
        if index % 500 == 0 or index == updates:
            if main.frozen_hashes(trainer) != before:
                raise RuntimeError("Non-Cost parameters changed during warm-up")
            trainer.save_checkpoint(delta, include_replay=False, extra_metadata={
                "format": "cost_resume_v1", "main_warmup_updates": index,
                "base_file": base.name, "base_sha256": base_digest,
                "replay_rng": trainer.buffer.sampling_state_dict(),
                "initial_rng": trainer.initial_state_buffer.state_dict()["rng_state"],
            })
    after = main.frozen_hashes(trainer)
    if before != after:
        raise RuntimeError("Non-Cost parameters changed during warm-up")
    main.write(warmup / "audit.json", {"frozen_before": before, "frozen_after": after,
                                      "updates": updates, "collection_steps": trainer.total_environment_steps})


def _development_path(output, method, seed, target):
    return output / "evaluation" / "development" / f"step_{target}" / "mean" / method / f"seed_{seed:03d}"


def _development_complete(model, evaluation, manifest, method, seed, target):
    result = _read_json(evaluation / "completed.json")
    identity = result.get("identity", {})
    return (
        result.get("episodes") == len(manifest["development_ids"])
        and identity.get("method") == method and identity.get("training_seed") == seed
        and identity.get("step_label") == target
        and identity.get("runtime_hocbf") is True and identity.get("stochastic") is False
        and identity.get("source_sha256") == manifest["source_sha256"]
        and identity.get("bank_sha256") == manifest["bank_hashes"]["development"]
        and model.is_file() and identity.get("checkpoint_sha256") == main.file_hash(model)
    )


def _run_complete(run, evaluation, manifest, method, seed, target):
    stage = _read_json(run / f"stage_{target}.json")
    return (
        stage.get("target") == target and stage.get("actual_steps", -1) >= target
        and stage.get("settings_sha256") == manifest["settings_sha256"]
        and stage.get("source_sha256") == manifest["source_sha256"]
        and _development_complete(run / f"model_{target}.pt", evaluation, manifest, method, seed, target)
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


def _at_milestone(trainer, point):
    """Check whether the restored state is the first episode boundary past point."""
    path = trainer.logger.output_dir / "metrics.jsonl"
    if not path.is_file():
        return False
    recent = [json.loads(line)["cumulative_environment_steps"]
              for line in path.read_text(encoding="utf-8").splitlines() if line.strip()][-2:]
    return bool(recent and recent[-1] == trainer.total_environment_steps
                and (recent[-2] if len(recent) == 2 else 0) < point <= recent[-1])


def _run_with_progress(output, method, seed, target, manifest, report, complete):
    settings = manifest["settings"]
    expected = len(manifest["development_ids"])
    worker_case = main._worker_case

    def evaluate(step, snapshot):
        cases = 0
        detail = f"{step:,} stepモデル"
        report("development", cases, expected, "件", detail)

        def track_case(item):
            nonlocal cases
            row = worker_case(item)
            cases += 1
            report("development", cases, expected, "件", detail)
            return row

        rng = main.capture_rng()
        try:
            case_function = track_case if settings.get("evaluation_workers", 1) == 1 else worker_case
            with patch.object(main, "_worker_case", case_function):
                rows = main.evaluate_condition(output, method, seed, snapshot,
                                               split="development", checkpoint_step=step)
        finally:
            main._EVALUATOR = None
            main.restore_rng(rng)
        report("development", len(rows), expected, "件", detail)
        return rows

    run = output / "training" / method / f"seed_{seed:03d}"
    latest = run / "latest.pt"
    resuming = latest.is_file()
    cfg = main.config_for(manifest, method, seed, target if resuming else None)
    trainer = NotebookTrainer(cfg, build_env(cfg), run / ("online" if resuming else "warmup/collection"),
                              report=report, warmup_updates=settings["cost_warmup_updates"])
    try:
        if resuming:
            trainer.load_checkpoint(latest, allow_budget_extension=True)
            trainer.logger.truncate_after_episode(trainer.completed_episodes)
        else:
            _warmup(trainer, output, run / "warmup", settings["cost_warmup_updates"])
            trainer.begin_online(run / "online")

        bank = load_scenario_bank(output / "banks" / "training.json")
        for point in sorted({target, *(value for value in settings["checkpoint_steps"] if value <= target)}):
            snapshot = run / f"model_{point}.pt"
            if trainer.total_environment_steps < point:
                cfg.training.total_environment_steps_budget = point
                trainer.train(None, scenarios=bank, checkpoint_path=latest)
                trainer.save_checkpoint(latest)
                trainer.save_checkpoint(snapshot, include_replay=False)
            elif not snapshot.is_file():
                if trainer.total_environment_steps != point and not _at_milestone(trainer, point):
                    raise RuntimeError(f"{method}/{seed}: {point:,} stepのモデルがなく、保存点はその段階を過ぎています")
                cfg.training.total_environment_steps_budget = point
                trainer.save_checkpoint(snapshot, include_replay=False)

            evaluation = _development_path(output, method, seed, point)
            if not _development_complete(snapshot, evaluation, manifest, method, seed, point):
                evaluate(point, snapshot)
            stage = run / f"stage_{point}.json"
            if not stage.is_file():
                if not _at_milestone(trainer, point):
                    snapshot_payload = require_torch().load(snapshot, map_location="cpu", weights_only=False)
                    progress = snapshot_payload["trainer_progress"]
                    del snapshot_payload
                else:
                    progress = {"total_environment_steps": trainer.total_environment_steps,
                                "completed_episodes": trainer.completed_episodes}
                main.write(stage, {"target": point, "actual_steps": progress["total_environment_steps"],
                                   "episodes": progress["completed_episodes"], "checkpoint": str(snapshot),
                                   "source_sha256": manifest["source_sha256"],
                                   "resumable_checkpoint": str(latest), "settings_sha256": manifest["settings_sha256"]})
        if not complete():
            raise RuntimeError(f"{method}/{seed}: 学習・development完了記録を確認できません")
    finally:
        trainer.env.close()


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
                if not complete() or any(
                    not _development_complete(run / f"model_{point}.pt",
                                              _development_path(output, method, seed, point),
                                              manifest, method, seed, point)
                    for point in settings["checkpoint_steps"] if point < target
                ):
                    if not (run / "latest.pt").exists() and any(run.glob("stage_*.json")):
                        raise RuntimeError(f"{method}/{seed}: latest.pt がありません")
                    report("準備中")
                    try:
                        _run_with_progress(output, method, seed, target, manifest, report, complete)
                    finally:
                        torch = require_torch()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                finished.append((method, seed))
                report("development完了", expected, expected, "件")
    return finished
