"""Bounded subprocess launcher for the main Notebook.

Kept outside the experiment package so launcher changes do not invalidate
prepared banks or exact-resume checkpoints.
"""
from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic, sleep
from uuid import uuid4

import yaml

from uav_safe_marl import load_config
from uav_safe_marl.experiments.main import TRAINED, file_hash, prepare
from uav_safe_marl.policies.base import require_torch, resolve_device


def _training_device_info(settings_path):
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


def probe_training_device(repo, settings):
    """Check device selection in a fresh process like the training workers."""
    probe = subprocess.run(
        [sys.executable, "-c",
         "import json, sys; from scripts.main_notebook_runner import _training_device_info; "
         "print(json.dumps(_training_device_info(sys.argv[1])))",
         str(Path(settings).resolve())],
        cwd=repo, capture_output=True, text=True,
    )
    if probe.returncode:
        raise RuntimeError(f"学習デバイスを確認できませんでした。\n{probe.stderr.strip()}")
    return json.loads(probe.stdout)


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def run_training_grid(repo, settings, output, *, target=30000, workers=2, status_interval=30, progress_factory=None):
    """Train A/B/F for each configured seed with at most workers live children."""
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if not isinstance(target, int) or target < 1 or status_interval <= 0:
        raise ValueError("target and status_interval must be positive")

    repo, output = Path(repo).resolve(), Path(output).resolve()
    manifest = prepare(Path(settings), output)
    settings = manifest["settings"]
    expected = len(manifest["development_ids"])
    warmup_episodes = settings["banks"]["warmup"]["per_n"] * len(settings["agent_counts"])
    warmup_updates = settings["cost_warmup_updates"]
    milestones = sorted({target, *(point for point in settings["checkpoint_steps"] if point < target)})
    pending = deque((m, s, "train") for s in settings["training_seeds"] for m in TRAINED)
    active = {}
    finished = []
    failures = []
    live_status = {}
    bars = {}
    free_slots = deque(range(workers))

    def report(message):
        if progress_factory is None:
            print(message, flush=True)

    def paths(method, seed, point=target):
        run = output / "training" / method / f"seed_{seed:03d}"
        evaluation = (output / "evaluation" / "development" / f"step_{point}" /
                      "mean" / method / f"seed_{seed:03d}" / "completed.json")
        return run, evaluation

    def phase_progress(method, seed, phase, token):
        run, _ = paths(method, seed)
        if phase == "development完了":
            return "development完了", expected, expected, "件", ""
        status = {}
        if phase == "train":
            observed = _read_json(run / "notebook_progress.json")
            if observed.get("token") == token:
                live_status[token] = observed
            status = live_status.get(token, {})
            worker_phase = status.get("phase")
            if worker_phase == "collection":
                return "warm-up収集", status["episodes"], warmup_episodes, "ep", f"{status['steps']:,} step"
            if worker_phase == "cost":
                return "Cost Critic warm-up", status["updates"], warmup_updates, "更新", ""
            if worker_phase == "saving_collection":
                return "収集保存・Cost Critic準備中", 0, 1, "", status.get("checkpoint", "")
            if worker_phase != "online":
                return "起動・保存点読込中", 0, 1, "", status.get("checkpoint", "")
        for point in milestones:
            _, evaluation = paths(method, seed, point)
            if (phase == "evaluate" and point == target) or (
                (run / f"model_{point}.pt").exists() and not evaluation.exists()
            ):
                cases = sum(1 for _ in (evaluation.parent / "cases").glob("*/completed.json"))
                return "development", cases, expected, "件", f"{point:,} stepモデル"
        steps, episodes = status["steps"], status["episodes"]
        detail = f"ep {episodes:,}"
        if steps > target:
            detail += f" · 実績 {steps:,} step"
        return "online学習", steps, target, "step", detail

    def show_progress(method, seed, phase, slot, token):
        label, count, total, unit, detail = phase_progress(method, seed, phase, token)
        description = f"{method}/{seed} · {label}"
        if slot not in bars:
            numbers = f" {count:,}/{total:,} {unit}" if unit else ""
            report(f"進捗 {description}{numbers} {detail}".rstrip())
            return
        bar = bars[slot]
        current = min(count, total)
        if description != bar.desc or total != bar.total or unit != bar.unit or current < bar.n:
            bar.set_description_str(description, refresh=False)
            bar.unit = unit
            bar.bar_format = "{desc} {bar} {n:,}/{total:,} {unit}{postfix}" if unit else "{desc} {bar}{postfix}"
            bar.set_postfix_str(detail, refresh=False)
            bar.reset(total=total)
        if current != bar.n or detail != bar.postfix:
            bar.set_postfix_str(detail, refresh=False)
            if current != bar.n:
                bar.update(current - bar.n)
            bar.refresh()

    def done(method, seed):
        run, evaluation = paths(method, seed)
        stage = _read_json(run / f"stage_{target}.json")
        result = _read_json(evaluation)
        identity = result.get("identity", {})
        model = run / f"model_{target}.pt"
        return (
            stage.get("target") == target and stage.get("actual_steps", -1) >= target
            and stage.get("settings_sha256") == manifest["settings_sha256"]
            and stage.get("source_sha256") == manifest["source_sha256"]
            and result.get("episodes") == expected
            and identity.get("method") == method and identity.get("training_seed") == seed
            and identity.get("step_label") == target
            and identity.get("runtime_hocbf") is True and identity.get("stochastic") is False
            and identity.get("source_sha256") == manifest["source_sha256"]
            and identity.get("bank_sha256") == manifest["bank_hashes"]["development"]
            and model.is_file() and identity.get("checkpoint_sha256") == file_hash(model)
        )

    def launch(method, seed, phase):
        log = output / "launcher_logs" / f"{method}_seed{seed:03d}_{target}_{phase}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        module = "scripts.main_notebook_worker" if phase == "train" else "uav_safe_marl.experiments.main"
        args = [sys.executable, "-u", "-m", module,
                phase, "--root", str(output), "--method", method, "--seed", str(seed),
                "--target", str(target)]
        if phase == "evaluate":
            args += ["--split", "development"]
        token = uuid4().hex
        with log.open("ab") as stream:
            process = subprocess.Popen(args, cwd=repo, stdout=stream, stderr=subprocess.STDOUT,
                                       env={**os.environ, "UAV_MAIN_NOTEBOOK_PROGRESS_TOKEN": token})
        slot = free_slots.popleft()
        active[process] = (method, seed, phase, log, slot, token)
        if progress_factory is not None and slot not in bars:
            bars[slot] = progress_factory(
                total=1, desc=f"{method}/{seed}", position=slot, unit="",
                bar_format="{desc} {bar}{postfix}",
            )
        report(f"開始 {method}/{seed} {phase} → {log}")

    next_status = monotonic() + status_interval
    try:
        while pending or active:
            while pending and len(active) < workers and not failures:
                method, seed, phase = pending.popleft()
                run, _ = paths(method, seed)
                if phase == "train" and done(method, seed):
                    finished.append((method, seed))
                    report(f"完了済み {method}/{seed} を再利用")
                    continue
                # Don't silently restart an existing stage without its full state.
                if phase == "train" and not (run / "latest.pt").exists() and any(run.glob("stage_*.json")):
                    failures.append(f"{method}/{seed}: latest.pt がありません")
                    break
                launch(method, seed, phase)
            for process, (method, seed, phase, log, slot, token) in list(active.items()):
                status = process.poll()
                if status is None:
                    continue
                del active[process]
                live_status.pop(token, None)
                free_slots.append(slot)
                if status:
                    failures.append(f"{method}/{seed} {phase}: exit {status} ({log})")
                    print(f"失敗: {failures[-1]}。新規起動を止め、実行中の組の終了を待ちます。", flush=True)
                elif done(method, seed):
                    finished.append((method, seed))
                    if slot in bars:
                        show_progress(method, seed, "development完了", slot, token)
                    report(f"完了 {method}/{seed}: 学習・development保存済み")
                elif phase == "train":
                    # A resumed train can skip the milestone's evaluation.
                    pending.appendleft((method, seed, "evaluate"))
                else:
                    failures.append(f"{method}/{seed}: development完了記録を確認できません ({log})")
            if failures and not active:
                raise RuntimeError("\n".join(failures))
            due = monotonic() >= next_status
            if progress_factory is not None or due:
                for method, seed, phase, _, slot, token in active.values():
                    show_progress(method, seed, phase, slot, token)
            if due:
                next_status = monotonic() + status_interval
            if active:
                sleep(0.5)
    finally:
        # Interrupting the Notebook cell should not leave our training workers behind.
        for process in active:
            if process.poll() is None:
                process.terminate()
        for process in active:
            process.wait()
        for bar in bars.values():
            bar.close()
    return finished
