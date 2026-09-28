"""Portable reconstruction of a trainer from a trusted local checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from uav_safe_marl import build_env
from uav_safe_marl.config.schemas import load_config
from uav_safe_marl.runners.trainer import SafeRLTrainer


def _deep_update(target: dict[str, Any], updates: Mapping[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def load_trainer_checkpoint(path: str | Path, output_dir: str | Path, *, device: str = "auto", nested_parallelism: bool = False, config_overrides: Mapping[str, Any] | None = None) -> SafeRLTrainer:
    import torch

    checkpoint = Path(path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    values = dict(payload["config"])
    if config_overrides:
        _deep_update(values, config_overrides)
    values["device"] = device
    monitoring = dict(values.get("monitoring", {}))
    monitoring.update({"enabled": False, "auto_launch_tensorboard": False, "open_browser": False})
    values["monitoring"] = monitoring
    if not nested_parallelism:
        performance = dict(values.get("performance", {}))
        performance["safety_backend"] = "serial"
        values["performance"] = performance
    config = load_config(overrides=values)
    trainer = SafeRLTrainer(config, build_env(config), output_dir)
    trainer.load_checkpoint(checkpoint, resume_training=False)
    return trainer
