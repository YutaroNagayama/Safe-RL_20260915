"""Report live phases without changing the training source hash."""
import os
from pathlib import Path
import sys
from weakref import WeakMethod, proxy

from uav_safe_marl.experiments import main
from uav_safe_marl.runners.trainer import SafeRLTrainer


class NotebookTrainer(SafeRLTrainer):
    def __init__(self, config, env, output_dir):
        super().__init__(config, env, output_dir)
        output = Path(output_dir)
        self.collecting = output.name == "collection" and output.parent.name == "warmup"
        run = output.parent.parent if self.collecting else output.parent
        self.status_path = run / "notebook_progress.json"
        self.warmup_updates = 0
        self.collection_goal = None
        trainer = proxy(self)
        log_episode = WeakMethod(self.logger.log)

        def track_episode(summary):
            log_episode()(summary)
            trainer.report_progress("collection" if trainer.collecting else "online",
                                    steps=summary["cumulative_environment_steps"],
                                    episodes=summary["completed_episodes"])

        self.logger.log = track_episode
        update_cost = getattr(self.backend, "update_cost_critic_only_n_step", None)
        if update_cost is not None and self.collecting:
            update_cost = WeakMethod(update_cost)

            def track_cost_update(batch):
                if trainer.warmup_updates == 0:
                    trainer.report_progress("cost", updates=0)
                result = update_cost()(batch)
                trainer.warmup_updates += 1
                if trainer.warmup_updates == 1 or trainer.warmup_updates % 25 == 0:
                    trainer.report_progress("cost", updates=trainer.warmup_updates)
                return result

            self.backend.update_cost_critic_only_n_step = track_cost_update

    def load_checkpoint(self, path, **kwargs):
        self.report_progress("loading", checkpoint=Path(path).name)
        metadata = super().load_checkpoint(path, **kwargs)
        if self.collecting and "main_warmup_updates" in metadata:
            self.warmup_updates = metadata["main_warmup_updates"]
            self.report_progress("cost", updates=self.warmup_updates)
        return metadata

    def train(self, *args, **kwargs):
        # The caller has already restored the checkpoint and truncated the log.
        if self.collecting:
            self.collection_goal = len(kwargs["scenarios"])
        self.report_progress("collection" if self.collecting else "online",
                             steps=self.total_environment_steps, episodes=self.completed_episodes)
        history = super().train(*args, **kwargs)
        if self.collecting:
            self.report_progress("saving_collection")
        return history

    def save_checkpoint(self, path, **kwargs):
        if (self.collection_goal is not None and self.completed_episodes >= self.collection_goal
                and Path(path).name == "collection.pt"):
            self.report_progress("saving_collection", checkpoint=Path(path).name)
        return super().save_checkpoint(path, **kwargs)

    def report_progress(self, phase, **fields):
        try:
            main.write(self.status_path, {
                "token": os.environ.get("UAV_MAIN_NOTEBOOK_PROGRESS_TOKEN"),
                "phase": phase,
                **fields,
            })
        except OSError as exc:
            print(f"Cannot write progress display: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main.SafeRLTrainer = NotebookTrainer
    main.main()
