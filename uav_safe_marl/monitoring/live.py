"""TensorBoard and tqdm adapters for an active Trainer run."""

from __future__ import annotations

import math
import os
import json
from pathlib import Path
from time import perf_counter
from typing import Any
import warnings
import webbrowser


_SERVERS: dict[str, tuple[Any, str]] = {}


def _tqdm_for_runtime():
    """Use the widget progress bar explicitly in Jupyter/VS Code notebooks."""
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell":
            from tqdm.notebook import tqdm

            return tqdm
    except (ImportError, NameError):
        pass
    from tqdm.auto import tqdm

    return tqdm


def _group(key: str) -> str:
    if key in {"episode_success", "separation_violation_rate", "path_stretch", "pure_safety_intervention_magnitude", "communication_load"}:
        return f"primary/{key}"
    if key in {"entropy_alpha", "log_alpha", "alpha_loss", "alpha_update_magnitude", "effective_target_entropy", "mean_policy_log_prob"}:
        return f"entropy/{key}"
    if "loss" in key:
        return f"loss/{key}"
    if key in {
        "lambda", "lambda_before_update", "lambda_after_update", "lambda_update_magnitude",
        "fraction_lambda_positive", "start_state_cost_ucb", "replay_state_cost_ucb",
        "policy_action_cost_ucb", "policy_ucb_cost_estimate", "cost_critic_mean",
        "cost_critic_std", "cost_ucb", "cost_ucb_batch_mean",
        "fraction_cost_ucb_above_d_cost", "fraction_cost_ucb_above_d_cost_batch",
        "configured_d_cost", "discounted_episode_cost", "undiscounted_episode_cost",
        "dual_constraint_estimate", "dual_constraint_violation", "effective_cal_multiplier",
        "dual_update_applied", "dual_update_count", "dual_constraint_available", "episodic_constraint_estimate",
        "start_cost_critic_mean", "start_cost_critic_std", "replay_cost_ucb_p50",
        "replay_cost_ucb_p75", "replay_cost_ucb_p90", "replay_cost_ucb_p95",
        "replay_cost_ucb_max", "start_cost_ucb_p50", "start_cost_ucb_p75",
        "start_cost_ucb_p90", "start_cost_ucb_p95", "start_cost_ucb_max",
    }:
        return f"cost/{key}"
    if key.startswith(("qp_", "conflict_", "responsibility_", "emergency_", "cbf_")) or key in {"minimum_pair_distance", "runtime_filter_correction"}:
        return f"safety/{key}"
    if key in {"message_count", "transmitted_bytes", "neighbour_count"}:
        return f"communication/{key}"
    if key in {"goal_reward", "progress", "time_penalty", "absolute_distance_penalty", "residual_penalty", "episode_reward"}:
        return f"reward/{key}"
    if "action_norm" in key or key == "residual_saturation_rate":
        return f"action/{key}"
    if key in {"mean_normalized_goal_distance", "final_goal_distance_mean", "final_goal_distance_max", "minimum_goal_distance_mean"}:
        return f"navigation/{key}"
    if key.endswith("_time"):
        return f"timing/{key}"
    return f"episode/{key}"


class LiveTrainingMonitor:
    """Mirror one training run to a browser dashboard and a local progress bar."""

    def __init__(self, output_dir: str | Path, config: Any, enabled: bool = True) -> None:
        self.output_dir = Path(output_dir)
        self.config = config
        disabled_by_environment = os.environ.get("UAV_SAFE_MARL_DISABLE_LIVE", "").lower() in {"1", "true", "yes"}
        self.enabled = bool(enabled and config.monitoring.enabled and not disabled_by_environment)
        self.writer = None
        self.progress = None
        self.url: str | None = None
        self.started = perf_counter()
        self._last_refresh_step = -1

    def start(self, total_episodes: int | None, initial_episode: int = 0) -> None:
        if not self.enabled:
            return
        if self.config.monitoring.progress_bar:
            try:
                tqdm = _tqdm_for_runtime()
                self.progress = tqdm(
                    total=total_episodes,
                    initial=0,
                    unit="episode",
                    dynamic_ncols=True,
                    desc="Training",
                    leave=True,
                )
            except (ImportError, ModuleNotFoundError):
                warnings.warn("tqdm is unavailable; install uav-safe-marl[learn] for a live progress bar", RuntimeWarning)
        if self.config.monitoring.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                log_dir = self.output_dir / "tensorboard"
                self.writer = SummaryWriter(log_dir=str(log_dir))
                self.writer.add_text("run/config", f"```json\n{json.dumps(self.config.to_dict(), ensure_ascii=False, indent=2)}\n```", initial_episode)
                self.writer.flush()
                if self.config.monitoring.auto_launch_tensorboard:
                    self.url = self._launch_tensorboard(log_dir)
            except (Exception, SystemExit) as exc:
                warnings.warn(f"TensorBoard monitoring is unavailable: {exc}", RuntimeWarning)

    def _launch_tensorboard(self, log_dir: Path) -> str:
        key = str(log_dir.resolve())
        created = key not in _SERVERS
        if created:
            from tensorboard import program

            server = program.TensorBoard()
            server.configure(argv=[
                None,
                "--logdir", key,
                "--host", self.config.monitoring.tensorboard_host,
                "--port", str(self.config.monitoring.tensorboard_port),
            ])
            url = server.launch()
            _SERVERS[key] = (server, url)
        self.url = _SERVERS[key][1]
        message = f"TensorBoard: {self.url}"
        if self.progress is not None:
            self.progress.write(message)
        else:
            print(message, flush=True)
        if created and self.config.monitoring.open_browser:
            webbrowser.open(self.url, new=2)
        return self.url

    def update_step(self, episode_offset: int, step: int, max_steps: int, environment_steps: int, updates: int, replay_size: int, actual_n: int) -> None:
        if not self.enabled or (step + 1) % self.config.monitoring.step_refresh_interval != 0:
            return
        if environment_steps == self._last_refresh_step:
            return
        self._last_refresh_step = environment_steps
        elapsed = max(perf_counter() - self.started, 1e-12)
        rate = environment_steps / elapsed
        if self.progress is not None:
            self.progress.set_postfix(N=actual_n, step=f"{step + 1}/{max_steps}", env_steps=environment_steps, updates=updates, replay=replay_size, steps_s=f"{rate:.1f}")
            self.progress.refresh()
        if self.writer is not None:
            self.writer.add_scalar("progress/environment_steps", environment_steps, environment_steps)
            self.writer.add_scalar("progress/gradient_updates", updates, environment_steps)
            self.writer.add_scalar("progress/replay_size", replay_size, environment_steps)
            self.writer.add_scalar("progress/steps_per_second", rate, environment_steps)
            self.writer.add_scalar("progress/current_episode_offset", episode_offset, environment_steps)
            self.writer.add_scalar("progress/current_episode_step", step + 1, environment_steps)

    def end_episode(self, summary: dict[str, Any], environment_steps: int, updates: int, replay_size: int) -> None:
        if not self.enabled:
            return
        episode = int(summary["episode"])
        elapsed = max(perf_counter() - self.started, 1e-12)
        if self.writer is not None:
            for key, value in summary.items():
                if isinstance(value, bool):
                    value = float(value)
                if isinstance(value, (int, float)) and value is not None and math.isfinite(float(value)):
                    self.writer.add_scalar(_group(key), float(value), episode)
            self.writer.add_scalar("progress/environment_steps", environment_steps, episode)
            self.writer.add_scalar("progress/gradient_updates", updates, episode)
            self.writer.add_scalar("progress/replay_size", replay_size, episode)
            self.writer.add_scalar("progress/elapsed_seconds", elapsed, episode)
            self.writer.flush()
        if self.progress is not None:
            self.progress.set_postfix(
                N=int(summary["actual_agent_count"]),
                success=f"{summary['episode_success']:.0f}",
                reward=f"{summary['episode_reward']:.2f}",
                cost=f"{summary['discounted_episode_cost']:.2f}",
                env_steps=environment_steps,
                updates=updates,
            )
            self.progress.update(1)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        if self.progress is not None:
            self.progress.close()
