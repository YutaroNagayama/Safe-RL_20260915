# Implemented architecture and semantics

## Proposed distributed execution path

1. The scenario or caller supplies `predicted_los_edges` at reset. If omitted,
   the scenario predicts candidates from straight-line preflight trajectories.
2. `P2PCommunication(prior_share)` transmits only along those candidates. A
   receiver stores the last delivered learned embedding and its age.
3. `ActorGraphEncoder` consumes the receiver's local graph and produces
   `z_i,t`. The current actor decision uses lagged peer embeddings, so one
   communication round carries two-hop-equivalent learned context.
4. Fixed Reference Guidance maps own state and the exit waypoint to `u_ref` at
   every 0.1 s base tick. The shared Actor receives self features, normalized
   own reference, and `z_i,t`, and emits a bounded normalized residual.
5. `u_ref + delta_u_res` is acceleration-clipped into `u_nom`; the residual is
   held between the 0.5 s Actor ticks while Reference Guidance keeps updating.
6. Capability-aware responsibility assigns each pair's HOCBF demand while
   preserving `rho_pref` and reporting infeasible demand when necessary.
7. The runtime safety layer applies HOCBF constraints over CBF edges
   (`d_ij <= d_eng`), acceleration bounds, and the L2 next-speed constraint.
8. If constraints conflict, the resolver retains the maximal feasible urgency
   prefix, maximizes the first rejected barrier over `U_eff`, then minimizes
   distance to the nominal action on that lexicographic face.

## Proposed centralized training path

The global multi-agent state and joint nominal action form a centralized
state-action graph. Each reward critic and each cost-ensemble member owns a
separate, larger Critic Graph Encoder. Its invariant global pooling feeds Twin
Reward Critics or Cost-UCB. Actor and critic encoders never share parameters.

The `interaction` critic-edge option uses centralized interaction relations;
`fully_connected` is the default CTDE relation. Neither is confused with Actor
communication edges or CBF engagement edges.

## Intervention cost

Intervention cost intentionally removes `U_eff`. It reconstructs all original
HOCBF constraints, tests them again without actuator/speed limits, and invokes
the learning resolver only if the HOCBF half-spaces themselves conflict. For
the first conflicting constraint it minimizes nonnegative violation slack,
then minimizes `||u-u_nom||^2` at the optimal slack. Thus the term measures pure
safety-correction demand rather than vehicle-performance saturation.

## CAL and the cost budget

CAL is a shared module independent of pooled or graph representation. The
primary `pdf_variant` is the project's SAC + PDF-defined CAL formulation and is
not labeled an official CAL reproduction. The optional `paper` branch permits
future controlled comparison.

`d_cost` has the CMDP meaning

`J_C = E_{s0~mu,pi}[sum_t gamma_cost^t c_t] <= d_cost`.

The main `episodic_dual_replay_gradient` variant stores exact episode-start
observations in a separate Initial-state Buffer. Sampling follows the official
training start distribution `mu` (uniform over N only when the scenario bank is
defined that way). Start-state Cost-UCB determines both the projected dual
update and the scalar CAL effective multiplier. Replay-state policy Cost-UCB
remains the differentiated Actor surrogate, so hazardous intermediate states
continue to contribute off-policy safety gradients. This is an episodic
CMDP-oriented CAL-based formulation, not a claim of an unbiased on-policy
episodic policy gradient or an exact reproduction of official CAL.

`d_cost` is therefore the active-agent-normalized expected discounted
cumulative Cost budget for one episode, not an independent budget assigned to
each aircraft. `replay_cal_legacy` preserves the former replay-state dual signal
for diagnostic comparisons.

Every episode log therefore includes undiscounted and discounted episode cost,
start-state and replay-state Cost-UCB estimates, the effective multiplier,
dual-signal availability/source, and the configured `d_cost`.

Frozen-checkpoint calibration uses the same Runtime HOCBF setting as training.
Because the SAC Cost target samples its next action from the stochastic policy,
formal calibration also uses stochastic continuation. Each row samples one
initial action, evaluates `Q_C(s0,a0)`, forces that exact action into rollout,
then resumes ordinary stochastic multi-rate execution. Repeats estimate the
policy-action expectation. Its UCB coverage is explicitly an empirical
diagnostic, not a probabilistic confidence guarantee.

All-mean-action rollout is saved only as a deployment diagnostic: its
continuation policy differs from the SAC Cost Critic's training semantics and
therefore it is not used to calibrate `d_cost` or claim UCB coverage. HOCBF
ON/OFF contribution evaluation remains separate from both operations.

## Retained baseline

`pooled/concat` retains the pooled Actor input and concatenated centralized
critic input for debugging and ablation. Its width is fixed by
`max_agent_count`, while an explicit actual-agent mask excludes padded slots.
It can therefore consume the same mixed-N scenario distribution as graph/graph.
It is not the proposed method; `graph/graph` is the default.

## Mixed-N and heterogeneous capability

One run fixes only `max_agent_count`; each exact scenario supplies its actual
agent count. Direct Actor/Critic node features include per-agent maximum
acceleration and velocity. The reference/residual composition is reconstructed
identically in online execution and policy-gradient critic queries; critics
receive the resulting joint nominal action. Runtime HOCBF responsibility uses
the exact per-agent acceleration, current velocity, and velocity limit rather
than the learned P2P embedding.

## Tactical-segment completion semantics

The primary experiments use `goal_completion_mode: exit_airspace`. Here a goal
is the downstream exit waypoint of the local tactical avoidance segment, not a
landing point. Entering its configurable capture region (5 m by default)
completes the segment without requiring the UAV to stop. The final state remains
in rollout artifacts, while the agent leaves dynamics interaction, graph
pooling, P2P communication, HOCBF constraints, costs, and metric denominators on
the following tick. Dense arrays retain a zeroed slot and an explicit false
node/active mask for reproducibility.

Gymnasium termination is stored without collapsing time limits: all-agent goal
completion sets `terminated`; `max_steps` sets `truncated`. Critic targets use
`1 - terminated`, and therefore bootstrap across artificial time-limit
truncation.

## Fixed local envelope and navigation reward

Main scenarios use the same nominal 280 x 280 x 56 m generation envelope for
all N. It is not a physical box: positions are not clipped and there are no wall
collisions. The optional `density_controlled` generator retains the earlier
N-proportional cross-sectional scaling only for supplementary analysis.

Navigation reward is clipped goal-distance progress normalized by the distance
the reference cruise speed covers in one base tick, minus the existing time
penalty and a weak normalized residual-magnitude penalty (`w_res=0.01` for the
Pilot). Absolute goal-distance and cosine shaping are disabled. The normalized
one-time exit bonus defaults to 15. Collision, proximity, and pure safety-
intervention signals remain exclusively in the CMDP cost and HOCBF layers.

## Preflight interaction screening

Main scenarios sample N first and hold it fixed while regenerating geometry
until at least one time-aligned reference-trajectory CPA is within the 75 m
screening margin. No edge-count, degree, density, or all-to-all restriction is
applied. These values are ExactScenario descriptors rather than difficulty
controls. The historical `predicted_los_edges` field is an alias for this
predicted-interaction set.

For the proposed prior-sharing method, a screened pair remains eligible for
bidirectional P2P communication while both endpoints are active. Passing the
predicted CPA, separating, or leaving the 50 m HOCBF engagement region does not
delete it. Actual minimum distance, CBF engagement, and candidate misses are
method-dependent RolloutResult fields. Adaptive threat pruning is future work.

## Fixed Primary Metrics

Final claims use exactly five metrics:

1. all-agent Episode Success Rate;
2. `d_ij < d_safe` violations divided by active unordered pair-ticks;
3. mean Path Stretch over successful episodes, using the scenario-provided
   preflight reference-route length;
4. Pure Safety Intervention Magnitude, the active-agent-tick mean of
   `||u_intervention-u_nominal||^2` from the HOCBF-only learning QP;
5. transmitted payload bytes divided by active-agent seconds.

`||u_safe-u_nominal||^2` is logged separately as Runtime Filter Correction
because it may also contain acceleration and L2 speed-limit corrections. With
CBF disabled, the environment still evaluates the HOCBF-only QP
counterfactually but never applies its action.

## Model Set persistence

Notebook training remains the primary interface. A `ModelSet` adds a thin,
immutable registry around the existing Trainer checkpoint, resolved config,
training log, and scenario-bank APIs. Directories use
`YYYYMMDD_HHMMSS_name/models/method/seed_NNN`; finalization requires the full
declared method/seed matrix, equal non-seed config within each method, and valid
SHA-256 digests. Saved trainers can be reconstructed later from their registered
config and checkpoint. Custom-experiment execution is intentionally not part of
this layer; a display-only example documents how external scripts open a set.
