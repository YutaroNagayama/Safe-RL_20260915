"""Manual Windows/Linux/macOS command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from uav_safe_marl import build_env, load_config
from uav_safe_marl.core.types import AgentState
from uav_safe_marl.evaluation import evaluate_checkpoint_parallel, evaluate_scenario_bank, load_scenario_bank, save_evaluation
from uav_safe_marl.experiments.loading import load_trainer_checkpoint
from uav_safe_marl.runners.trainer import SafeRLTrainer
from uav_safe_marl.runtime import runtime_manifest
from uav_safe_marl.safety.execution import SafetyExecutor
from uav_safe_marl.safety.qp_filter import HOCBFSafetyFilter
from uav_safe_marl.safety.responsibility import CapabilityResponsibilityAllocator


def _doctor(_: argparse.Namespace) -> None:
    manifest = runtime_manifest()
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        tensor = torch.ones((16, 16), device=device)
        manifest.update({"selected_device": device, "torch_smoke_sum": float((tensor @ tensor).sum().cpu()), "torch_smoke_ok": True})
    except Exception as exc:
        manifest.update({"torch_smoke_ok": False, "torch_smoke_error": str(exc)})
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _train(args: argparse.Namespace) -> None:
    overrides = {"device": args.device}
    if args.seed is not None:
        overrides["seed"] = args.seed
    config = load_config(args.config, overrides=overrides)
    trainer = SafeRLTrainer(config, build_env(config), args.output)
    scenarios = load_scenario_bank(args.scenario_bank) if args.scenario_bank else None
    trainer.train(args.episodes, scenarios=scenarios)
    checkpoint = Path(args.output) / "checkpoints" / "final.pt"
    trainer.save_checkpoint(checkpoint)
    trainer.env.close()
    print(checkpoint.resolve())


def _evaluate(args: argparse.Namespace) -> None:
    scenarios = load_scenario_bank(args.scenario_bank)
    if args.workers > 1:
        result = evaluate_checkpoint_parallel(args.checkpoint, scenarios, workers=args.workers, method=args.method, training_seed=args.training_seed, device=args.device)
    else:
        trainer = load_trainer_checkpoint(args.checkpoint, Path(args.output) / "runtime", device=args.device)
        result = evaluate_scenario_bank(trainer, scenarios, method=args.method, training_seed=args.training_seed)
        trainer.env.close()
    save_evaluation(result, args.output)
    print(json.dumps(result["primary_metrics"], ensure_ascii=False, indent=2))


def _benchmark_safety(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    rows = []
    for count in args.agents:
        states = [AgentState(rng.uniform(-22, 22, 3), rng.uniform(-4, 4, 3), np.zeros(3), np.full(3, 3.0), np.full(3, 12.0)) for _ in range(count)]
        nominal = rng.uniform(-2, 2, (count, 3))
        active = np.ones(count, dtype=bool)
        safety_filter = HOCBFSafetyFilter(10, 50, 1, 1, .1, CapabilityResponsibilityAllocator("capability", .1, speed_constraint_mode="norm"), True, "norm")
        timings = {}
        reference = None
        for backend in ("serial", "process"):
            executor = SafetyExecutor(safety_filter, backend, args.workers, 1)
            executor.solve(states, nominal, active, True)  # exclude spawn/warmup
            started = time.perf_counter()
            results = None
            for _ in range(args.repeats):
                results, _, _ = executor.solve(states, nominal, active, True)
            timings[backend] = (time.perf_counter() - started) / args.repeats
            actions = np.vstack([item.safe_action for item in results])
            if reference is None:
                reference = actions
            elif not np.allclose(reference, actions, rtol=1e-10, atol=1e-10):
                raise RuntimeError("serial/process HOCBF mismatch")
            executor.close()
        rows.append({"N": count, "serial_ms": 1000 * timings["serial"], "process_ms": 1000 * timings["process"], "speedup": timings["serial"] / timings["process"]})
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def _tensorboard(args: argparse.Namespace) -> None:
    from tensorboard import program
    server = program.TensorBoard()
    server.configure(argv=[None, "--logdir", str(Path(args.run).resolve()), "--host", args.host, "--port", str(args.port)])
    print(server.launch(), flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python -m uav_safe_marl")
    commands = root.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor"); doctor.set_defaults(func=_doctor)
    train = commands.add_parser("train")
    train.add_argument("--config", default="configs/full.yaml"); train.add_argument("--episodes", type=int, required=True)
    train.add_argument("--output", required=True); train.add_argument("--scenario-bank"); train.add_argument("--seed", type=int); train.add_argument("--device", default="auto"); train.set_defaults(func=_train)
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--checkpoint", required=True); evaluate_parser.add_argument("--scenario-bank", required=True); evaluate_parser.add_argument("--output", required=True)
    evaluate_parser.add_argument("--workers", type=int, default=1); evaluate_parser.add_argument("--device", default="cpu"); evaluate_parser.add_argument("--method", default="model"); evaluate_parser.add_argument("--training-seed", type=int, default=0); evaluate_parser.set_defaults(func=_evaluate)
    benchmark = commands.add_parser("benchmark-safety")
    benchmark.add_argument("--agents", type=int, nargs="+", default=[2, 4, 8, 16]); benchmark.add_argument("--workers", type=int, default=4); benchmark.add_argument("--repeats", type=int, default=3); benchmark.add_argument("--seed", type=int, default=7); benchmark.set_defaults(func=_benchmark_safety)
    tensorboard = commands.add_parser("tensorboard")
    tensorboard.add_argument("--run", required=True); tensorboard.add_argument("--host", default="127.0.0.1"); tensorboard.add_argument("--port", type=int, default=6006); tensorboard.set_defaults(func=_tensorboard)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
