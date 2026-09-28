"""Structured experiment logging with reproducibility metadata."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from uav_safe_marl.config.schemas import ExperimentConfig
from uav_safe_marl.runtime import runtime_manifest


class ExperimentLogger:
    """Write JSONL metrics and a configuration snapshot."""

    def __init__(self, output_dir: str | Path, config: ExperimentConfig) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        metadata = {"created_at": datetime.now(timezone.utc).isoformat(), "config": config.to_dict(), "runtime": runtime_manifest(config), "git_commit": self._git_commit()}
        (self.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _git_commit() -> str | None:
        try:
            return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    def log(self, metrics: dict[str, Any]) -> None:
        with (self.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    def truncate_after_episode(self, completed_episodes: int) -> None:
        """Discard metrics newer than the last durable periodic checkpoint."""
        path = self.output_dir / "metrics.jsonl"
        if not path.is_file():
            return
        retained = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("episode", -1)) < completed_episodes:
                retained.append(json.dumps(row, ensure_ascii=False))
        content = "\n".join(retained)
        path.write_text(content + ("\n" if content else ""), encoding="utf-8")
