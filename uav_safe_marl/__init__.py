"""Public construction API for the UAV Safe-MARL experiment platform."""

from .config.schemas import ExperimentConfig, load_config


def build_env(config: ExperimentConfig):
    """Build the configured multi-UAV environment."""
    from .envs.multi_uav_env import MultiUAVEnv

    return MultiUAVEnv(config)


def build_trainer(config: ExperimentConfig, env=None):
    """Build a trainer without exposing algorithm internals to notebooks."""
    from .runners.trainer import SafeRLTrainer

    return SafeRLTrainer(config, env or build_env(config))


__all__ = ["ExperimentConfig", "build_env", "build_trainer", "load_config"]

