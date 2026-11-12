"""Report Cost Critic warm-up progress without changing the training source hash."""
from pathlib import Path
import sys

from uav_safe_marl.experiments import main
from uav_safe_marl.runners.trainer import SafeRLTrainer


class NotebookTrainer(SafeRLTrainer):
    def __init__(self, config, env, output_dir):
        super().__init__(config, env, output_dir)
        output = Path(output_dir)
        self.warmup_updates = 0
        update_cost = getattr(self.backend, "update_cost_critic_only_n_step", None)
        if update_cost is not None and output.name == "collection" and output.parent.name == "warmup":
            self.warmup_status = output.parent / "notebook_updates.json"
            try:
                self.warmup_status.unlink(missing_ok=True)
            except OSError as exc:
                print(f"Cannot reset warm-up display: {exc}", file=sys.stderr)
            def track_cost_update(batch):
                result = update_cost(batch)
                self.warmup_updates += 1
                if self.warmup_updates == 1 or self.warmup_updates % 25 == 0:
                    self.report_warmup_updates()
                return result

            self.backend.update_cost_critic_only_n_step = track_cost_update

    def load_checkpoint(self, path, **kwargs):
        metadata = super().load_checkpoint(path, **kwargs)
        if "main_warmup_updates" in metadata and hasattr(self, "warmup_status"):
            self.warmup_updates = metadata["main_warmup_updates"]
            self.report_warmup_updates()
        return metadata

    def report_warmup_updates(self):
        try:
            main.write(self.warmup_status, {"updates": self.warmup_updates})
        except OSError as exc:
            print(f"Cannot write warm-up display: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main.SafeRLTrainer = NotebookTrainer
    main.main()
