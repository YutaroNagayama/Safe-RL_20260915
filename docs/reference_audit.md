# Reference-to-implementation audit

Audit date: 2026-09-17. The initial-recognition PDF is the primary mathematical
specification; the implementation requirements and R1--R11 map are secondary.

| Ref. | Status | Implementation and verification |
| --- | --- | --- |
| R1 InforMARL | Adapted | The proposed path uses a learned local Actor Graph Encoder, one-round delayed learned messages, graph-valued replay, and separate centralized Critic Graph Encoders. Tests cover permutation behavior, variable agent counts, and configured message width. The attention implementation is UniMP-inspired, not an exact PyG UniMP reproduction. |
| R2 CAL | CAL-consistent | `algorithms/cal.py` is representation-independent and exposes `pdf_variant` and `paper`. The primary mode implements the PDF-defined piecewise augmented penalty, policy-action Cost-UCB, and projected additive dual update. These choices intentionally do not claim exact reproduction of the official CAL code. |
| R3 SAC | Aligned | The squashed Gaussian actor uses reparameterized sampling and tanh log-Jacobian correction. Entropy temperature can be learned. Training collection is stochastic; `execution_mode` controls evaluation/deployment behavior. |
| R4 TD3 | CAL-based variant | Twin reward critics use the target minimum, normalized target noise, and delayed actor/target updates. CAL Cost-UCB is an explicit project extension, not original TD3. |
| R5 CBF-QP | Aligned | Runtime projection minimizes nominal-action correction under engaged HOCBF half-spaces, acceleration limits, and the configured L2 next-speed constraint. `box` remains an ablation. |
| R6 HOCBF | Aligned | Relative-degree-two terms implement `2 r^T(u_i-u_j) + q >= 0`; tests cover analytic safe/unsafe cases and initial `h, psi_1` margins. |
| R7 discrete-time CBF | Reference only | No discrete-time CBF proof is claimed. Continuous-time HOCBF is reevaluated every base tick; time-step regressions are empirical checks. |
| R8 intervention reduction | Adapted | Learning intervention recomputes all original HOCBF constraints without `U_eff`. If they still conflict, a learning-only minimum-slack lexicographic resolver is used. The resulting squared physical correction measures safety demand independently of vehicle capability limits. |
| R9 dynamic responsibility | Adapted | Responsibility is asymmetric when capabilities differ and pair shares sum to one. The project uses instantaneous support over `U_eff` rather than the reference paper's complete risk-allocation formulation. |
| R10 capability-aware HCBF | Aligned | Support-function capability, demand, feasible intervals, degeneracy handling, and diagnostics are implemented. If combined capability is insufficient, the requested `rho_pref` is retained and an infeasible flag is set. |
| R11 Layered-Safe-MARL | Adapted | Runtime resolution sorts by urgency and keeps the maximal feasible priority prefix. At the first conflict it maximizes that barrier over bounded `U_eff`, then minimizes nominal-action correction; lower-priority constraints are skipped. |

## Corrections reflected by this audit

- Distance Cost now preserves the same proximity penalty curve while averaging
  each active agent's closest-peer cost. This supersedes the earlier `2/N`
  all-pair sum so graph degree alone does not inflate the learning constraint.
- Replaced equal-share fallback under insufficient capability with preserved
  `rho_pref` plus an infeasibility diagnostic.
- Split runtime and learning conflict resolution: bounded barrier maximization
  for runtime, unbounded-actuator minimum slack for intervention learning.
- Replaced hard-coded all-to-all prior communication with injected/predicted
  LoS candidates.
- Replaced the graph path's 69-to-23 slicing with the explicit learned Actor
  Graph Encoder output `z_i`; the old pooled path remains only as a baseline.
- Added graph-valued replay and independent Actor, Reward Critic, and Cost
  Critic encoders; every Cost Ensemble member owns its encoder.
- Added explicit joint nominal actions to centralized critic node features.
- Added L2 speed constraints, selectable progress definitions, shared CAL
  processing, and discounted-CMDP-scale cost logging.

## Scientific validation boundary

Passing tests establishes equations, shapes, contracts, and regression
behavior. It does not establish statistical performance. Communication savings,
intervention reduction, success/safety trade-offs, and scalability require
seeded multi-run ablations with confidence intervals. Exact replication claims
for PyG UniMP or the official CAL repository remain out of scope and must not be
made from this implementation.
