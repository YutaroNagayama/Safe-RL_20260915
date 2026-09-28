"""Small CPU smoke for episodic dual training and matched-policy calibration."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from uav_safe_marl import build_env, load_config
from uav_safe_marl.evaluation import (
    calibrate_cost_critic,
    evaluate_mean_policy_cost_diagnostic,
    load_scenario_bank,
)
from uav_safe_marl.runners.trainer import SafeRLTrainer


def scenarios_per_agent_count(bank: list[dict], counts: tuple[int, ...], per_count: int) -> list[dict]:
    selected = []
    for count in counts:
        candidates = [item for item in bank if int(item["actual_agent_count"]) == count]
        if len(candidates) < per_count:
            raise ValueError(f"bank contains only {len(candidates)} scenarios for N={count}")
        selected.extend(candidates[:per_count])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-calibration", action="store_true")
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--scenarios-per-n", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    source = root / "runs" / "mini_pilot_nearest_dcost6_mps"
    bank = load_scenario_bank(source / "training_bank.json")
    scenarios = scenarios_per_agent_count(bank, (2, 4, 8), args.scenarios_per_n)
    run_root = root / "runs" / f"calibration_smoke_{datetime.now():%Y%m%d_%H%M%S}"
    run_root.mkdir(parents=True)

    base = json.loads((source / "resolved_config.json").read_text(encoding="utf-8"))
    output = {"run_root": str(run_root)}
    if not args.reference_only:
        smoke_overrides = deepcopy(base)
        smoke_overrides["device"] = "cpu"
        smoke_overrides["environment"]["max_steps"] = 40
        smoke_overrides["learning"].update({
            "batch_size": 32, "d_cost": 6.0,
            "constraint_semantics": "episodic_start_state",
            "cal_variant": "episodic_dual_replay_gradient",
            "initial_state_buffer_size": 128,
        })
        smoke_overrides["scenario"]["training_agent_count_distribution"] = {2: 1 / 3, 4: 1 / 3, 8: 1 / 3}
        smoke_overrides["training"].update({
            "total_environment_steps_budget": None, "learning_starts": 32,
            "dual_learning_starts": 32, "dual_update_period": 1,
            "dual_batch_size": 12, "checkpoint_every_episodes": 1,
        })
        smoke_overrides["monitoring"].update({
            "enabled": False, "progress_bar": False, "tensorboard": False,
            "auto_launch_tensorboard": False, "open_browser": False,
        })
        smoke_overrides["performance"].update({"profile": "exact", "safety_backend": "serial"})
        smoke = load_config(root / "configs" / "full.yaml", overrides=smoke_overrides)
        trainer = SafeRLTrainer(smoke, build_env(smoke), run_root / "training")
        history = trainer.train(episodes=len(scenarios), scenarios=scenarios, live_display=False)
        trainer.save_checkpoint(run_root / "checkpoints" / "final.pt")
        smoke_calibration = calibrate_cost_critic(
            trainer, scenarios, repeats=args.repeats, output_dir=run_root / "smoke_calibration",
        )
        trainer.env.close()
        output.update({
            "training": {
                "episodes": len(history), "environment_steps": trainer.total_environment_steps,
                "gradient_updates": trainer.total_gradient_updates, "final_lambda": history[-1]["lambda"],
                "dual_constraint_available": history[-1].get("dual_constraint_available"),
                "initial_state_occupancy": {
                    key: value for key, value in history[-1].items() if key.startswith("initial_state_occupancy_N")
                },
            },
            "smoke_calibration": smoke_calibration["overall"],
        })

    if args.reference_calibration or args.reference_only:
        reference_overrides = deepcopy(base)
        reference_overrides["device"] = "cpu"
        reference_overrides["monitoring"].update({
            "enabled": False,
            "progress_bar": False,
            "tensorboard": False,
            "auto_launch_tensorboard": False,
            "open_browser": False,
        })
        reference_overrides["performance"].update({"profile": "exact", "safety_backend": "serial"})
        reference = load_config(root / "configs" / "full.yaml", overrides=reference_overrides)
        reference_trainer = SafeRLTrainer(reference, build_env(reference), run_root / "reference")
        reference_trainer.load_checkpoint(source / "checkpoints" / "final.pt", resume_training=False)
        reference_calibration = calibrate_cost_critic(
            reference_trainer,
            scenarios,
            repeats=args.repeats,
            output_dir=run_root / "reference_calibration",
        )
        mean_diagnostic = evaluate_mean_policy_cost_diagnostic(
            reference_trainer,
            scenarios,
            repeats=1,
            output_dir=run_root / "mean_policy_diagnostic",
        )
        reference_trainer.env.close()
        output["reference_checkpoint"] = str(source / "checkpoints" / "final.pt")
        output["reference_calibration"] = reference_calibration["overall"]
        output["reference_calibration_by_n"] = reference_calibration["by_agent_count"]
        output["mean_policy_diagnostic"] = {
            key: value for key, value in mean_diagnostic.items() if key != "rows"
        }

    (run_root / "smoke_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
