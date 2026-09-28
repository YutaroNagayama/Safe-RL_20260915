"""Typed configuration and YAML loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, TypeVar


@dataclass(slots=True)
class PolicyConfig:
    backend: str = "SAC"
    execution_mode: str = "mean"
    hidden_dim: int = 128
    learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    entropy_alpha_initial: float = 0.2
    entropy_autotune: bool = True


@dataclass(slots=True)
class GraphConfig:
    """Actor/critic aggregation backends and independent graph capacities."""

    actor_aggregation_backend: str = "graph"
    critic_aggregation_backend: str = "graph"
    actor_embedding_dim: int = 32
    actor_hidden_dim: int = 64
    actor_layers: int = 1
    actor_attention_heads: int = 2
    critic_embedding_dim: int = 128
    critic_hidden_dim: int = 128
    critic_layers: int = 2
    critic_attention_heads: int = 4
    critic_edge_mode: str = "fully_connected"


@dataclass(slots=True)
class SafetyConfig:
    enabled: bool = True
    responsibility_mode: str = "capability"
    conflict_resolver_enabled: bool = True
    d_safe: float = 10.0
    d_warn: float = 30.0
    d_eng: float = 50.0
    k1: float = 1.0
    k2: float = 1.0
    speed_constraint_mode: str = "norm"


@dataclass(slots=True)
class CommunicationConfig:
    unimp_enabled: bool = True
    information_mode: str = "prior_share"
    actor_period_k: int = 5
    delay_steps: int = 1


@dataclass(slots=True)
class ControlConfig:
    """Reference-guidance and bounded residual-action settings."""

    actor_mode: str = "residual"
    reference_speed_fraction: float = 0.75
    reference_velocity_gain: float = 0.8
    residual_acceleration_fraction: float = 0.5
    reference_update_steps: int = 1


@dataclass(slots=True)
class EnvironmentConfig:
    dt_base: float = 0.1
    max_steps: int = 500
    goal_tolerance: float = 5.0
    agent_count: int = 2
    max_agent_count: int | None = None
    acceleration_limit: float = 3.0
    velocity_limit: float = 12.0
    goal_completion_mode: str = "exit_airspace"

    @property
    def capacity(self) -> int:
        """Fixed padded capacity; ``agent_count`` remains the default actual N."""
        return self.agent_count if self.max_agent_count is None else self.max_agent_count


@dataclass(slots=True)
class ScenarioConfig:
    """Preflight interaction-screening and boundary-corridor defaults."""

    spatial_scaling_mode: str = "fixed"
    longitudinal_length: float = 280.0
    transverse_size: float = 280.0
    altitude_min: float = -28.0
    altitude_max: float = 28.0
    volume_scale_min: float = 1.0
    volume_scale_max: float = 1.0
    minimum_initial_separation: float = 20.0
    interaction_threshold: float = 75.0
    prediction_horizon: float = 50.0
    initial_speed_fraction_min: float = 0.4
    initial_speed_fraction_max: float = 0.65
    require_predicted_candidate: bool = True
    # Empty means "use the empirical training-bank start distribution".
    # Main experiments should set this explicitly (for example, equal mass on
    # N=2/4/8/16) so the episodic dual estimator reproduces s_0 ~ mu.
    training_agent_count_distribution: dict[int, float] = field(default_factory=dict)


@dataclass(slots=True)
class ObjectiveConfig:
    goal_reward: float = 15.0
    w_progress: float = 1.0
    w_time: float = 0.01
    absolute_distance_enabled: bool = False
    absolute_distance_scale: float = 280.0
    w_absolute_distance: float = 0.002
    distance_cost_enabled: bool = True
    intervention_cost_enabled: bool = True
    w_intervention: float = 1.0
    progress_mode: str = "normalized_goal_delta"
    residual_penalty_enabled: bool = True
    w_residual: float = 0.01


@dataclass(slots=True)
class LearningConfig:
    gamma_reward: float = 0.99
    gamma_cost: float = 0.99
    d_cost: float = 3.0
    cal_lambda_initial: float = 0.0
    cal_lambda_lr: float = 0.01
    cal_coefficient: float = 1.0
    cost_ensemble_size: int = 2
    beta_ucb: float = 1.0
    batch_size: int = 256
    buffer_size: int = 100_000
    cal_mode: str = "pdf_variant"
    constraint_semantics: str = "episodic_start_state"
    cal_variant: str = "episodic_dual_replay_gradient"
    initial_state_buffer_size: int = 10_000
    # Reward/Actor learning remains one-step.  This horizon applies only to
    # the Cost Critic Bellman target so long-horizon safety cost can propagate
    # without changing the rest of SAC.
    cost_target_n_step: int = 1


@dataclass(slots=True)
class TrainingConfig:
    """Interaction-budget and update-schedule controls."""

    total_environment_steps_budget: int | None = None
    learning_starts: int = 256
    updates_per_environment_step: int = 1
    # Update reward/cost critics every gradient step, but update the Actor and
    # entropy temperature only once per this many steps.  A value of one is
    # standard SAC; two is the sole delayed-policy candidate used by the
    # Pre-Main Gate.
    actor_update_period: int = 1
    checkpoint_every_episodes: int = 10
    dual_learning_starts: int = 256
    dual_update_period: int = 1
    dual_batch_size: int = 256
    dual_update_mode: str = "gradient_step"
    dual_update_interval_episodes: int = 1


@dataclass(slots=True)
class MonitoringConfig:
    """Live display layered on top of canonical JSONL/NPZ artifacts."""

    enabled: bool = True
    progress_bar: bool = True
    tensorboard: bool = True
    auto_launch_tensorboard: bool = True
    open_browser: bool = True
    tensorboard_host: str = "127.0.0.1"
    tensorboard_port: int = 0
    step_refresh_interval: int = 25


@dataclass(slots=True)
class PerformanceConfig:
    """Execution-only tuning; these options must not change experiment math."""

    profile: str = "exact"
    safety_backend: str = "auto"
    safety_workers: int = 0
    safety_parallel_min_agents: int = 8
    evaluation_workers: int = 0
    training_workers: int = 1
    workers_per_gpu: int = 1
    torch_num_threads: int = 0
    torch_interop_threads: int = 1
    blas_threads_per_worker: int = 1
    cuda_non_blocking_transfer: bool = True
    mixed_precision: bool = False
    torch_compile: bool = False
    allow_tf32: bool = False
    deterministic_algorithms: bool = False


@dataclass(slots=True)
class ExperimentConfig:
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    seed: int = 0
    device: str = "auto"

    def validate(self) -> None:
        """Raise ``ValueError`` when an experiment is internally inconsistent."""
        if self.policy.backend.upper() not in {"SAC", "TD3"}:
            raise ValueError("policy.backend must be SAC or TD3")
        if self.policy.execution_mode not in {"stochastic", "mean"}:
            raise ValueError("execution_mode must be stochastic or mean")
        if self.graph.actor_aggregation_backend not in {"pooled", "graph"}:
            raise ValueError("actor_aggregation_backend must be pooled or graph")
        if self.graph.critic_aggregation_backend not in {"concat", "graph"}:
            raise ValueError("critic_aggregation_backend must be concat or graph")
        if (self.graph.actor_aggregation_backend, self.graph.critic_aggregation_backend) not in {
            ("pooled", "concat"), ("graph", "graph")
        }:
            raise ValueError("supported aggregation pairs are pooled/concat and graph/graph")
        if self.graph.critic_edge_mode not in {"fully_connected", "interaction"}:
            raise ValueError("critic_edge_mode must be fully_connected or interaction")
        if min(self.graph.actor_embedding_dim, self.graph.actor_hidden_dim, self.graph.actor_layers, self.graph.actor_attention_heads, self.graph.critic_embedding_dim, self.graph.critic_hidden_dim, self.graph.critic_layers, self.graph.critic_attention_heads) < 1:
            raise ValueError("graph dimensions, layers, and heads must be positive")
        if self.graph.actor_hidden_dim % self.graph.actor_attention_heads:
            raise ValueError("actor_hidden_dim must be divisible by actor_attention_heads")
        if self.graph.critic_hidden_dim % self.graph.critic_attention_heads:
            raise ValueError("critic_hidden_dim must be divisible by critic_attention_heads")
        if self.safety.responsibility_mode not in {"equal", "capability"}:
            raise ValueError("responsibility_mode must be equal or capability")
        if self.safety.speed_constraint_mode not in {"norm", "box"}:
            raise ValueError("speed_constraint_mode must be norm or box")
        if self.communication.information_mode not in {"prior_share", "reactive_share", "none"}:
            raise ValueError("invalid information_mode")
        if self.control.actor_mode not in {"residual", "reference_only", "full"}:
            raise ValueError("control.actor_mode must be residual, reference_only, or full")
        if not 0.0 < self.control.reference_speed_fraction <= 1.0:
            raise ValueError("reference_speed_fraction must lie in (0, 1]")
        if self.control.reference_velocity_gain <= 0.0 or not 0.0 < self.control.residual_acceleration_fraction <= 1.0:
            raise ValueError("reference gain and residual acceleration fraction must be positive")
        if self.control.reference_update_steps != 1:
            raise ValueError("the current specification requires reference_update_steps=1")
        if not 0 < self.safety.d_safe < self.safety.d_warn <= self.safety.d_eng:
            raise ValueError("require 0 < d_safe < d_warn <= d_eng")
        if self.communication.actor_period_k < 1 or self.environment.agent_count < 1:
            raise ValueError("period and agent count must be positive")
        if self.environment.capacity < self.environment.agent_count:
            raise ValueError("max_agent_count must be at least agent_count")
        if self.environment.goal_completion_mode != "exit_airspace":
            raise ValueError("only goal_completion_mode=exit_airspace is currently implemented")
        if min(self.scenario.longitudinal_length, self.scenario.transverse_size, self.scenario.minimum_initial_separation, self.scenario.prediction_horizon) <= 0.0:
            raise ValueError("scenario dimensions, separation, and horizon must be positive")
        if self.scenario.spatial_scaling_mode not in {"fixed", "density_controlled"}:
            raise ValueError("spatial_scaling_mode must be fixed or density_controlled")
        if self.scenario.altitude_max <= self.scenario.altitude_min:
            raise ValueError("scenario altitude range must be increasing")
        if not 0.0 < self.scenario.volume_scale_min <= self.scenario.volume_scale_max:
            raise ValueError("invalid scenario volume scale range")
        if not 0.0 <= self.scenario.initial_speed_fraction_min <= self.scenario.initial_speed_fraction_max <= 1.0:
            raise ValueError("initial speed fractions must lie in [0, 1]")
        if self.scenario.interaction_threshold <= self.safety.d_eng:
            raise ValueError("interaction screening threshold must exceed d_eng")
        if self.communication.delay_steps < 0:
            raise ValueError("communication delay must be non-negative")
        if self.policy.entropy_alpha_initial <= 0:
            raise ValueError("entropy_alpha_initial must be positive")
        if self.policy.alpha_learning_rate <= 0:
            raise ValueError("alpha_learning_rate must be positive")
        if self.objective.progress_mode not in {"distance_delta", "directional", "normalized_goal_delta"}:
            raise ValueError("invalid progress_mode")
        if self.objective.absolute_distance_scale <= 0.0:
            raise ValueError("absolute_distance_scale must be positive")
        if self.objective.w_absolute_distance < 0.0:
            raise ValueError("w_absolute_distance must be non-negative")
        if self.objective.w_residual < 0.0:
            raise ValueError("w_residual must be non-negative")
        if self.learning.cal_mode not in {"pdf_variant", "paper"}:
            raise ValueError("cal_mode must be pdf_variant or paper")
        if self.learning.constraint_semantics not in {"episodic_start_state", "replay_state_surrogate"}:
            raise ValueError("invalid constraint_semantics")
        if self.learning.cal_variant not in {"replay_cal_legacy", "episodic_dual_replay_gradient"}:
            raise ValueError("invalid cal_variant")
        expected_semantics = {
            "episodic_dual_replay_gradient": "episodic_start_state",
            "replay_cal_legacy": "replay_state_surrogate",
        }[self.learning.cal_variant]
        if self.learning.constraint_semantics != expected_semantics:
            raise ValueError("constraint_semantics and cal_variant must describe the same dual signal")
        if self.learning.initial_state_buffer_size < 1:
            raise ValueError("initial_state_buffer_size must be positive")
        if self.learning.cost_target_n_step < 1:
            raise ValueError("cost_target_n_step must be positive")
        if self.learning.cost_target_n_step > 1 and self.policy.backend.upper() != "SAC":
            raise ValueError("multi-step Cost targets are currently supported only by SAC")
        distribution = {
            int(count): float(probability)
            for count, probability in self.scenario.training_agent_count_distribution.items()
        }
        if any(count < 1 or probability <= 0.0 for count, probability in distribution.items()):
            raise ValueError("training_agent_count_distribution requires positive N and probability")
        if distribution and abs(sum(distribution.values()) - 1.0) > 1e-6:
            raise ValueError("training_agent_count_distribution probabilities must sum to one")
        if distribution and max(distribution) > self.environment.capacity:
            raise ValueError("training_agent_count_distribution exceeds max_agent_count")
        self.scenario.training_agent_count_distribution = distribution
        if self.training.total_environment_steps_budget is not None and self.training.total_environment_steps_budget < 1:
            raise ValueError("total_environment_steps_budget must be positive or null")
        if min(
            self.training.learning_starts,
            self.training.updates_per_environment_step,
            self.training.actor_update_period,
            self.training.checkpoint_every_episodes,
            self.training.dual_learning_starts,
            self.training.dual_update_period,
            self.training.dual_batch_size,
            self.training.dual_update_interval_episodes,
        ) < 1:
            raise ValueError("training and dual update schedule values must be positive")
        if self.training.dual_update_mode not in {"gradient_step", "episode"}:
            raise ValueError("dual_update_mode must be gradient_step or episode")
        if self.training.dual_update_mode == "episode" and (
            self.policy.backend.upper() != "SAC"
            or self.learning.cal_variant != "episodic_dual_replay_gradient"
        ):
            raise ValueError("episode dual updates require SAC episodic_dual_replay_gradient")
        if self.monitoring.tensorboard_port < 0 or self.monitoring.tensorboard_port > 65535:
            raise ValueError("tensorboard_port must be in [0, 65535]")
        if self.monitoring.step_refresh_interval < 1:
            raise ValueError("step_refresh_interval must be positive")
        if self.performance.profile not in {"exact", "fast"}:
            raise ValueError("performance.profile must be exact or fast")
        if self.performance.safety_backend not in {"serial", "process", "auto"}:
            raise ValueError("safety_backend must be serial, process, or auto")
        for name in ("safety_workers", "evaluation_workers", "torch_num_threads"):
            if getattr(self.performance, name) < 0:
                raise ValueError(f"performance.{name} must be non-negative")
        if min(self.performance.safety_parallel_min_agents, self.performance.training_workers, self.performance.workers_per_gpu, self.performance.torch_interop_threads, self.performance.blas_threads_per_worker) < 1:
            raise ValueError("performance worker thresholds and fixed thread counts must be positive")
        if self.performance.profile == "exact" and any((self.performance.mixed_precision, self.performance.torch_compile, self.performance.allow_tf32)):
            raise ValueError("exact performance profile forbids mixed precision, torch.compile, and TF32")

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-ready configuration dictionary."""
        return asdict(self)


T = TypeVar("T")


def _merge_dataclass(instance: T, values: Mapping[str, Any]) -> T:
    known = {f.name: f for f in fields(instance)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    for key, value in values.items():
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, Mapping):
            _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> ExperimentConfig:
    """Load defaults, an optional YAML file, then nested mapping overrides."""
    config = ExperimentConfig()
    if path is not None:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to load a YAML file") from exc
        with Path(path).open(encoding="utf-8") as stream:
            values = yaml.safe_load(stream) or {}
        _merge_dataclass(config, values)
    if overrides:
        _merge_dataclass(config, overrides)
    config.validate()
    return config
