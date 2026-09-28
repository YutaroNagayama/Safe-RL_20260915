# UAV P2P Safe-MARL

## Windows本番実験

Windows + GeForce RTX上の手動実験は、[Windows実験Book](notebooks/windows/README.md)を使用します。共通scenario生成とResidual Actor方式群の学習・固定bank評価が別Notebookになっており、全成果物は `results/result_番号_日付/方式名/` に保存されます。

Gymnasium-based research package for decentralized UAV execution and
centralized safe multi-agent reinforcement learning (CTDE).

The proposed configuration is `graph/graph`:

- an externally supplied preflight LoS candidate set determines which peers
  may exchange P2P messages;
- a small shared Actor Graph Encoder produces each UAV's learned message
  embedding `z_i` from its local, delayed graph;
- fixed exit-waypoint guidance produces a reference acceleration and the shared
  actor produces a bounded residual correction;
- the reference and residual are composed and clipped into the nominal action;
- capability-aware responsibility, relative-degree-two HOCBF constraints, and
  a lexicographic runtime resolver produce safe actions;
- independent, larger centralized state-action graph encoders receive the
  global state and joint **nominal** action for Twin Reward Critics and every
  member of the Cost Critic Ensemble.

The previous `pooled/concat` path remains available as a debug baseline and
ablation (`configs/pooled_concat.yaml`). It is not used by the proposed default.

## Quick start

```bash
python -m pip install -e '.[learn,plot,dev]'
python -m pytest
```

```python
from uav_safe_marl import build_env, build_trainer, load_config

config = load_config()  # graph/graph + pdf_variant
env = build_env(config)
trainer = build_trainer(config, env)
history = trainer.train(episodes=2)
```

Training automatically opens a local TensorBoard page and displays a `tqdm`
progress bar. Canonical data continues to be written to `metrics.jsonl`; the
browser copy is stored below the same run directory in `tensorboard/`.

```yaml
monitoring:
  enabled: true
  progress_bar: true
  tensorboard: true
  auto_launch_tensorboard: true
  open_browser: true
  tensorboard_host: 127.0.0.1
  tensorboard_port: 0       # choose an available local port
  step_refresh_interval: 25
```

For a headless/CI run, use `monitoring.enabled: false`, pass
`trainer.train(..., live_display=False)`, or set
`UAV_SAFE_MARL_DISABLE_LIVE=1`. TensorBoard is a display layer only: JSONL and
evaluation NPZ files remain the reproducible source of truth.

For mixed-N training, configure a fixed padded capacity and pass an exact
scenario sequence. Both graph/graph and pooled/concat use the same distribution:

```python
config = load_config(overrides={"environment": {"agent_count": 2, "max_agent_count": 16}})
trainer = build_trainer(config, build_env(config))
bank = generate_random_traffic_bank(
    agent_count=[2, 4, 8, 12, 16], scenario_count=500, seed=2026,
    initial_speed_range=(4.0, 8.0),
    acceleration_range=(2.0, 4.0),
    velocity_limit_range=(10.0, 16.0),
)
trainer.train(episodes=len(bank), scenarios=bank)
```

The environment exposes 14 direct node features: position, velocity,
goal-relative position, own maximum acceleration, own maximum velocity, and
the normalized own reference action.
All tensors are padded to `max_agent_count`; nonexistent slots have a false
actual-agent mask and are excluded from policies, graph pooling, communication,
safety, rewards, costs, and metrics. The formal distance defaults are
`d_safe/d_warn/d_eng = 10/30/50 m`.

Goal completion uses local tactical-segment exit semantics: `goal` is the
downstream waypoint rather than a landing point. Entering its configurable
capture region (5 m by default) completes the segment without requiring a stop.
The agent is retained in the audit trajectory, but leaves physical interaction,
Actor/Critic graphs, communication, HOCBF constraints, costs, and metric
denominators from the following tick onward.

To inject a mission-specific preflight LoS candidate set directly:

```python
observation, info = env.reset(
    seed=0,
    options={"predicted_los_edges": [(0, 1), (1, 2)]},
)

# The same injection is available through the training/evaluation runners.
history = trainer.train(
    episodes=2,
    reset_options={"predicted_los_edges": [(0, 1), (1, 2)]},
)
```

Actor, Critic, and CBF edges intentionally have different meanings. Actor
edges are successful P2P deliveries restricted to the preflight candidate set;
Critic edges are configurable centralized interaction relations; CBF edges are
runtime pairs satisfying `d_ij <= d_eng`.

The CAL logic is representation-independent. `learning.cal_mode: pdf_variant`
is the primary SAC + PDF-defined piecewise formulation; `paper` provides a
separate paper-oriented penalty branch. This package is CAL-based / CAL-
consistent and does **not** claim to reproduce the official CAL implementation
exactly. The default `episodic_dual_replay_gradient` variant interprets
`d_cost` as an active-agent-normalized expected discounted episode-Cost budget:
`E_{s0~mu,pi}[sum_t gamma_cost^t c_t] <= d_cost`. Its dual update and augmented
multiplier use an initial-state batch sampled according to the configured
training distribution `mu`; the Actor still obtains its Cost gradient from
Replay states. The retained `replay_cal_legacy` variant is available only with
`constraint_semantics: replay_state_surrogate` for controlled comparisons.

The graph layers are dependency-light learned attention layers inspired by
InforMARL/UniMP. They implement graph-valued replay, explicit learned messages,
permutation equivariance/invariance, and separate Actor/Critic capacities, but
are not a drop-in reproduction of PyTorch-Geometric's UniMP convolution.
See `docs/architecture.md` and `docs/reference_audit.md` for exact boundaries.

## Fixed-scenario evaluation

```python
from uav_safe_marl.evaluation import (
    evaluate_scenario_bank,
    generate_random_traffic_bank,
    load_scenario_bank,
    save_evaluation,
    save_scenario_bank,
)

save_scenario_bank(
    "scenarios/test_bank.json",
    generate_random_traffic_bank(agent_count=8, scenario_count=200, seed=2026),
)
bank = load_scenario_bank("scenarios/test_bank.json")
result = evaluate_scenario_bank(trainer, bank, method="full", training_seed=0)
save_evaluation(result, "runs/full_seed0/evaluation")
```

Cost Critic calibration is a separate frozen-checkpoint operation. It preserves
the trainer's configured Runtime HOCBF condition. For SAC it samples one
initial action, evaluates `Q_C(s0, a0)` at that action, forces the same action
into the first rollout step, and continues with the stochastic SAC policy.
Scenario-level predictions and Monte Carlo returns are retained:

```python
from uav_safe_marl.evaluation import calibrate_cost_critic

calibration = calibrate_cost_critic(
    trainer,
    bank,
    output_dir="runs/full_seed0/cost_calibration",
)
```

Use multiple `repeats` per scenario to integrate over policy action sampling.
`empirical_ucb_coverage` is diagnostic coverage on this bank; it is not a
statistical confidence guarantee. Deterministic mean-action deployment cost is
available separately through `evaluate_mean_policy_cost_diagnostic`; for SAC it
is not labeled Cost Critic calibration because the Critic was trained for a
stochastic continuation policy. HOCBF ON/OFF remains a separate contribution
experiment.

The default scenario bank models a local tactical segment rather than a hard
bounded airspace. UAVs enter on one face of an imaginary corridor and fly
toward an exit waypoint on the opposite face. Main runs use a fixed
280 x 280 x 56 m nominal envelope for every N; dynamics still impose no walls
or position clipping. Increasing N therefore increases traffic density inside
the same local sector. The retained `density_controlled` scaling mode provides
the earlier N-proportional-volume condition for supplementary experiments.
Initial speed is 40--65% of each UAV's own speed limit. Every Main scenario
contains at least one time-aligned predicted-interaction candidate, screened at
75 m versus the 50 m runtime HOCBF engagement distance. Candidate count,
degree, density, and all-to-all status are recorded but never used to reject or
label scenario difficulty. The corridor is only a generation and measurement
window: dynamics do not clip positions or impose walls.

Navigation reward combines normalized active-agent goal-distance progress, a
per-step time penalty, a one-time normalized exit bonus, and a weak normalized
residual-magnitude penalty. Absolute goal-distance and cosine shaping are off in
the Residual Pilot. Collision proximity and pure HOCBF correction remain CMDP
costs rather than reward terms.

The legacy code field `predicted_los_edges` aliases the preflight predicted-
interaction set. A selected pair remains eligible for bidirectional P2P
communication while both agents are active; CPA passage or increasing distance
does not prune it. Dynamic threat pruning is intentionally outside this work.

Evaluation returns `primary_metrics` overall, `primary_metrics_by_n`, and an
equal-N `primary_metrics_macro_by_n`. Multi-seed confidence intervals and
per-seed values are available through `aggregate_primary_by_method_and_n`.

The final table is restricted to Episode Success Rate, active-pair Separation
Violation Rate, successful-episode Path Stretch, Pure Safety Intervention
Magnitude, and Communication Load per active-agent second. Full rollout NPZ
files retain positions, velocities, nominal/safe/intervention actions, masks,
pair distances, Actor/CBF edges, information age, and cost/reward components.

## Notebook-first Model Sets

Comparable methods and training seeds can be registered without changing the
existing Trainer workflow:

```python
from uav_safe_marl.experiments import ModelSet
from uav_safe_marl.runners.trainer import SafeRLTrainer

model_set = ModelSet.create(
    "main_comparison",
    expected_methods=["full", "no_intervention", "reactive", "pooled"],
    expected_seeds=[0, 1, 2, 3, 4],
)

trainer = SafeRLTrainer(
    config,
    build_env(config),
    output_dir=model_set.training_dir("full", config.seed),
)
trainer.train(episodes=100)
model_set.save_trainer(trainer, method="full", training_seed=config.seed)

# Register every expected method/seed first, then freeze the comparison set.
model_set.attach_scenario_bank("test", "scenarios/test_bank.json")
model_set.finalize()
```

Later, the exact architecture and checkpoint can be reconstructed with
`ModelSet.open(path).load_trainer("full", 0)`. Finalization checks the complete
method/seed matrix, config consistency across seeds, and SHA-256 digests. The
display-only demo is `experiments/custom/model_set_info_demo.py`.

## Manual Windows / CUDA operation

For VSCode/Jupyter operation, start with
`notebooks/windows/00_generate_scenarios.ipynb`. The bootstrap creates `.venv`,
installs the editable package and notebook dependencies, and registers
`UAV Safe MARL (.venv)`. Select that kernel once in VSCode, then run the desired
method Book. TensorBoard and tqdm are enabled automatically; results remain as
ordinary files under `results/result_番号_日付/` for Google Drive transfer.

The package has a Codex-independent command line interface.  Run the hardware
check first, then train or evaluate from PowerShell:

```powershell
python -m uav_safe_marl doctor
python -m uav_safe_marl benchmark-safety --agents 2 4 8 16 --workers 4
python -m uav_safe_marl train --config configs/full.yaml --episodes 1000 --output runs/full_seed0 --device auto
python -m uav_safe_marl evaluate --checkpoint runs/full_seed0/checkpoints/final.pt --scenario-bank scenarios/test_bank.json --output runs/full_seed0/evaluation --workers 4 --device cpu
python -m uav_safe_marl tensorboard --run runs/full_seed0/tensorboard --port 6006
```

`device: auto` selects CUDA before MPS or CPU.  Agent safety QPs use a shared
unordered-pair constraint build and switch from serial execution to a persistent
Windows-compatible spawn process pool at the configured active-agent threshold.
Statistical evaluation parallelizes complete scenarios and disables nested QP
processes.  Every run records its OS, Python, Torch, CUDA, GPU, and performance
configuration in `metadata.json`.
