"""Bounded subprocess launcher for the main Notebook.

Kept outside the experiment package so launcher changes do not invalidate
prepared banks or exact-resume checkpoints.
"""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import subprocess
import sys
from time import monotonic, sleep

from uav_safe_marl.experiments.main import TRAINED, file_hash, prepare


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _latest_progress(path):
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            offset = max(0, stream.tell() - 65536)
            stream.seek(offset)
            lines = stream.read().splitlines()
        for line in reversed(lines[1:] if offset else lines):
            try:
                row = json.loads(line)
                return (
                    f"{row['cumulative_environment_steps']:,} step, "
                    f"N={row['actual_agent_count']}, "
                    f"Cost={row['discounted_episode_cost']:.3g}"
                )
            except (ValueError, KeyError):
                continue
    except FileNotFoundError:
        pass
    return "warm-up / 準備中"


def run_training_grid(repo, settings, output, *, target=30000, workers=2, status_interval=30):
    """Train A/B/F for each configured seed with at most workers live children."""
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if not isinstance(target, int) or target < 1 or status_interval <= 0:
        raise ValueError("target and status_interval must be positive")

    repo, output = Path(repo).resolve(), Path(output).resolve()
    manifest = prepare(Path(settings), output)
    expected = len(manifest["development_ids"])
    pending = deque((m, s, "train") for s in manifest["settings"]["training_seeds"] for m in TRAINED)
    active = {}
    finished = []
    failures = []

    def paths(method, seed):
        run = output / "training" / method / f"seed_{seed:03d}"
        evaluation = (output / "evaluation" / "development" / f"step_{target}" /
                      "mean" / method / f"seed_{seed:03d}" / "completed.json")
        return run, evaluation

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
        args = [sys.executable, "-u", "-m", "uav_safe_marl.experiments.main",
                phase, "--root", str(output), "--method", method, "--seed", str(seed),
                "--target", str(target)]
        if phase == "evaluate":
            args += ["--split", "development"]
        with log.open("ab") as stream:
            process = subprocess.Popen(args, cwd=repo, stdout=stream, stderr=subprocess.STDOUT)
        active[process] = (method, seed, phase, log)
        print(f"開始 {method}/{seed} {phase} → {log}", flush=True)

    next_status = monotonic() + status_interval
    try:
        while pending or active:
            while pending and len(active) < workers and not failures:
                method, seed, phase = pending.popleft()
                run, _ = paths(method, seed)
                if phase == "train" and done(method, seed):
                    finished.append((method, seed))
                    print(f"完了済み {method}/{seed} を再利用", flush=True)
                    continue
                # Don't silently restart an existing stage without its full state.
                if phase == "train" and not (run / "latest.pt").exists() and any(run.glob("stage_*.json")):
                    failures.append(f"{method}/{seed}: latest.pt がありません")
                    break
                launch(method, seed, phase)
            for process, (method, seed, phase, log) in list(active.items()):
                status = process.poll()
                if status is None:
                    continue
                del active[process]
                if status:
                    failures.append(f"{method}/{seed} {phase}: exit {status} ({log})")
                    print(f"失敗: {failures[-1]}。新規起動を止め、実行中の組の終了を待ちます。", flush=True)
                elif done(method, seed):
                    finished.append((method, seed))
                    print(f"完了 {method}/{seed}: 学習・development保存済み", flush=True)
                elif phase == "train":
                    # A resumed train can skip the milestone's evaluation.
                    pending.appendleft((method, seed, "evaluate"))
                else:
                    failures.append(f"{method}/{seed}: development完了記録を確認できません ({log})")
            if failures and not active:
                raise RuntimeError("\n".join(failures))
            if active and monotonic() >= next_status:
                for method, seed, phase, _ in active.values():
                    run, _ = paths(method, seed)
                    progress = _latest_progress(run / "online" / "metrics.jsonl")
                    print(f"進捗 {method}/{seed} {phase}: {progress}", flush=True)
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
    return finished
