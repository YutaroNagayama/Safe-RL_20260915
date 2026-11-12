"""Measure concurrent main training on one machine without touching production runs."""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


class WindowsMemoryStatus(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_uint32),
        ("load", ctypes.c_uint32),
        ("total_physical", ctypes.c_uint64),
        ("available_physical", ctypes.c_uint64),
        ("total_page_file", ctypes.c_uint64),
        ("available_page_file", ctypes.c_uint64),
        ("total_virtual", ctypes.c_uint64),
        ("available_virtual", ctypes.c_uint64),
        ("available_extended_virtual", ctypes.c_uint64),
    ]


def sample_resources(nvidia_smi):
    sample = {}
    if sys.platform == "win32":
        memory = WindowsMemoryStatus()
        memory.length = ctypes.sizeof(memory)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            sample["ram_available_gib"] = round(memory.available_physical / 1024**3, 2)
            sample["ram_total_gib"] = round(memory.total_physical / 1024**3, 2)
    if nvidia_smi:
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True,
            )
            sample["gpus"] = [
                dict(zip(("index", "utilization_percent", "used_mib", "total_mib"),
                         (int(value.strip()) for value in line.split(",")), strict=True))
                for line in result.stdout.splitlines() if line.strip()
            ]
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return sample


def train_worker(data, collection, scenarios, run_dir, args, barrier, results):
    import traceback

    from uav_safe_marl import build_env
    from uav_safe_marl.experiments.main import config_for
    from uav_safe_marl.runners.trainer import SafeRLTrainer

    try:
        cfg = config_for(data, args.method, 0, None)
        cfg.device = "cpu" if args.cpu else "cuda"
        cfg.monitoring.enabled = False
        # Start measuring in the active-update regime, after filling Replay.
        cfg.training.learning_starts = 1
        cfg.training.dual_learning_starts = 1
        trainer = SafeRLTrainer(cfg, build_env(cfg), run_dir)
        try:
            trainer.train(len(collection), scenarios=collection, update_mode="collect_only")
            if len(trainer.buffer) < cfg.learning.batch_size:
                raise RuntimeError("Replay is below batch size. Increase --collect-per-n.")
            initial_replay = len(trainer.buffer)
            trainer.total_environment_steps = trainer.completed_episodes = 0
            trainer.transitions_collected_by_n.clear()
            trainer.scenario_bank_cursor = 0
            trainer.scenario_bank_hash = None
            cfg.training.total_environment_steps_budget = args.steps
            barrier.wait(timeout=3600)
            started = time.monotonic()
            trainer.train(None, scenarios=scenarios, checkpoint_path=run_dir / "latest.pt")
            results.put({
                "started": started, "finished": time.monotonic(),
                "steps": trainer.total_environment_steps, "updates": trainer.total_gradient_updates,
                "initial_replay": initial_replay, "final_replay": len(trainer.buffer),
            })
        finally:
            trainer.env.close()
    except Exception:
        results.put({"error": traceback.format_exc()})
        raise


def run_level(level, args, data, collection, scenarios, root, nvidia_smi):
    from queue import Empty

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(level)
    results = ctx.Queue()
    processes = []
    samples = []
    rows = []
    try:
        for index in range(level):
            run_dir = root / f"workers_{level}" / f"run_{index}"
            process = ctx.Process(
                target=train_worker,
                args=(data, collection, scenarios, run_dir, args, barrier, results),
            )
            process.start()
            processes.append(process)
        while len(rows) < level:
            samples.append({"time": time.monotonic(), **sample_resources(nvidia_smi)})
            try:
                row = results.get(timeout=5)
            except Empty:
                if any(p.exitcode not in (0, None) for p in processes):
                    raise RuntimeError("A worker exited unsuccessfully. See its traceback above.")
                if all(p.exitcode == 0 for p in processes):
                    raise RuntimeError("Workers exited before all results were received.")
                continue
            if "error" in row:
                raise RuntimeError(row["error"])
            rows.append(row)
        for process in processes:
            process.join()
            if process.exitcode:
                raise RuntimeError(f"Worker exited with code {process.exitcode}")
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join()
        results.close()

    first_start = min(row["started"] for row in rows)
    last_finish = max(row["finished"] for row in rows)
    seconds = last_finish - first_start
    online_samples = [sample for sample in samples if first_start <= sample["time"] <= last_finish]
    return {
        "workers": level, "online_seconds": seconds,
        "total_steps_per_second": sum(row["steps"] for row in rows) / seconds,
        "total_updates_per_second": sum(row["updates"] for row in rows) / seconds,
        "runs": rows, "online_resource_samples": online_samples, "all_resource_samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--steps", type=int, default=2000, help="Online step target per worker")
    parser.add_argument("--collect-per-n", type=int, default=2, help="Replay prefill episodes per agent count")
    parser.add_argument("--method", choices=("A", "B", "F"), default="A")
    parser.add_argument("--settings", type=Path, default=ROOT / "notebooks/main/settings.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/parallel_benchmark")
    parser.add_argument("--cpu", action="store_true", help="Use CPU for a smoke test")
    args = parser.parse_args()
    if min(*args.levels, args.steps, args.collect_per_n) < 1:
        parser.error("levels, steps, and collect-per-n must be positive")

    sys.path.insert(0, str(ROOT))
    import torch

    from uav_safe_marl.evaluation import load_scenario_bank
    from uav_safe_marl.experiments.main import prepare, write

    if not args.cpu and not torch.cuda.is_available():
        parser.error("PyTorch cannot use CUDA. Run with the RTX-enabled project environment.")
    root = args.output.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    data = prepare(args.settings, root)
    settings = data["settings"]
    if args.collect_per_n > settings["banks"]["warmup"]["per_n"]:
        parser.error("--collect-per-n exceeds the number of warm-up scenarios per agent count")
    collection = load_scenario_bank(root / "banks/warmup.json")[:args.collect_per_n * len(settings["agent_counts"])]
    scenarios = load_scenario_bank(root / "banks/training.json")
    nvidia_smi = None if args.cpu else shutil.which("nvidia-smi")
    report = {
        "method": args.method, "target_steps": args.steps, "collect_per_n": args.collect_per_n,
        "device": "cpu" if args.cpu else torch.cuda.get_device_name(0), "levels": [],
    }
    print(f"Output: {root}", flush=True)
    for level in sorted(set(args.levels)):
        print(f"Testing {level} concurrent worker(s)", flush=True)
        try:
            row = run_level(level, args, data, collection, scenarios, root, nvidia_smi)
        except (RuntimeError, OSError) as error:
            report["failed_level"] = {"workers": level, "error": str(error)}
            write(root / "benchmark.json", report)
            raise
        report["levels"].append(row)
        write(root / "benchmark.json", report)
        print(f"{level} worker(s): {row['total_steps_per_second']:.2f} steps/s total, "
              f"{row['total_updates_per_second']:.2f} updates/s total", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
