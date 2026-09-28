# Windows Residual Actor実験Book

Windows + GeForce RTX + VS Codeで、固定scenario bankを全方式へ公平に使う実行一式です。

## 実行順

1. VS Codeでリポジトリ全体を開く。
2. `experiment_settings.yaml` のscale、seed、N、step budgetを確認する。
3. `00_generate_scenarios.ipynb`を実行する。初回のみ`.venv`が自動構成される。
4. Kernelを **UAV Safe MARL (.venv)** に切り替え、Bookを先頭から再実行する。
5. 必要な方式のBookだけを実行する。

| Book | 方式 |
|---|---|
| `01_A_full_prior.ipynb` | Full-Prior: predicted interaction P2P + Residual Graph Actor + HOCBF |
| `02_B_full_reactive.ipynb` | Full-Reactive: sensing P2P + Residual Graph Actor + HOCBF |
| `03_C_residual_prior_no_hocbf.ipynb` | Prior P2P + Residual Actor、pairwise HOCBF OFF |
| `04_D_residual_reactive_no_hocbf.ipynb` | Reactive P2P + Residual Actor、pairwise HOCBF OFF |
| `05_E_hocbf_only.ipynb` | Reference Guidance + HOCBFのみ（学習なし） |

各Bookの`RUN_THIS_METHOD`が`False`なら実行しません。A/Bは通常評価に加え、同じcheckpointを使ったHOCBF OFF診断も`evaluation_hocbf_off/`へ保存します。

## 共通設定

全Bookは`experiment_settings.yaml`と、00が生成した`resolved_experiment.json`を参照します。個別Book内に実験条件を複製しません。

- `experiment.scale`: `smoke` / `pilot` / `main`
- `profiles`: training seed、N、scenario数
- `scenario`: 固定空域、性能分布、interaction条件
- `base_config.control`: Reference/Residual設定
- `base_config`: Graph、Safety、通信、Reward、CAL、step budget
- `methods`: 方式間で異なる最小限のoverride

同じ日付・同じ`result_index`のscenario bankは上書きしません。条件を変更する場合は`result_index`を増やして00を再実行してください。

## 出力

```text
results/result_番号_日付/
  shared/
    training_bank.json
    evaluation_bank.json
    resolved_experiment.json
  A/
    seed_000/
      resolved_config.json
      training/metrics.jsonl
      training/tensorboard/events...
      checkpoints/final.pt
      evaluation/
      evaluation_hocbf_off/
      completed.json
    method_summary.json
  B/ ...
  C/ ...
  D/ ...
  E/ ...
```

学習時はNotebook内にtqdmを表示し、TensorBoardをブラウザで開きます。Macへは`result_番号_日付`フォルダーを通常のファイル群のままGoogle Driveで移してください。
