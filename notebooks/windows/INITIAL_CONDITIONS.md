# Residual Actor実験の初期条件

実値の一次設定は`experiment_settings.yaml`です。

## 制御構造

- Actor mode: residual
- Reference speed fraction: 0.75
- Reference velocity gain: 0.8
- Residual acceleration fraction: 0.5
- Reference更新: 1 tick（0.1 s）
- Actor/Graph更新: 5 tick（0.5 s）、tick間はResidualを保持
- Actor入力: self state + normalized own reference + Graph embedding
- Critic action: 合成・加速度clip後のjoint nominal action
- Runtime HOCBF OFFでも加速度・L2 next-speed制約は維持

## Reward / Cost

- Reward: normalized goal-distance progress + time penalty 0.01 + goal reward 15 − residual penalty
- Residual penalty weight: 0.01
- Absolute goal-distance penalty: OFF
- Cosine direction penalty: OFF
- Cost: Distance Cost + Pure Safety Intervention Cost
- d_cost: 3.0（Pilot暫定）

## Scenario

- 280 m × 280 m × 56 m、物理壁なし
- Main N: 2 / 4 / 8 / 16
- start/exit: 対向境界を横断
- 初期速度: 各機v_maxの40〜65%
- v_max: 10〜14 m/s、a_max: 2〜4 m/s²
- d_safe / d_warn / d_eng / d_interaction: 10 / 30 / 50 / 75 m
- predicted interaction candidateを最低1 edge含む
- exit許容半径: 5 m

## Pilot初期学習値

- SAC、learning rate 3e-4、batch 256、replay 100,000
- gamma_reward / gamma_cost: 0.99 / 0.99
- learning starts: 2,000
- updates/environment step: 1
- environment-step budget: 80,000 / seed
- Cost Critic ensemble: 2、beta_UCB: 1
- Graph Actor: embedding 32、hidden 64、1 layer、2 heads
- Graph Critic: embedding/hidden 128、2 layers、4 heads

これらはPilot初期値であり、Main確定値ではありません。
