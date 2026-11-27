# Windows追加評価：A・B75・B50の航路延長ロールアウト

## 実行するもの

新規学習は行わず、既存30kのA・B75・B50各3seedだけを評価する。E/R、HBA/F、別のA条件、HOCBF OFFは追加しない。学習・制御のcoreと既存のmain/additional runnerは変更していない。

Notebook：`notebooks/main/main_route_extension.ipynb`
実行処理：`scripts/main_route_extension_runner.py`

## 航路・空間の拡張

新しい四方交通の元形状を生成し、予測CPA<10mかつ予測接近時刻>1秒のペアを含む形状を選ぶ。N=2/4/8は少なくとも1ペア、N=16は少なくとも2ペアを指定危険ペアとする。最長延長条件の指定ペアは開始時150m以上離れている。

同じ元形状に対して、開始位置を

`p_extended = p_original - v_initial * LEAD`

とし、LEAD=0/20/40秒の条件を作る。目標・航路方向・初速・能力は維持する。各指定ペアの予測CPA距離が同じで、予測CPA時刻がLEAD分だけ遅れることを生成時に検査する。全ペアの初期離隔30mも各条件で検査し、不適合形状は採用しない。面ごとの機数割当は強制しない。

これは元の交差領域と出口を残し、四方からの進入航路を後方へ延ばす試験である。出発点は共通の大きな正方形外周に揃えない。3D航路に沿って延長するため、水平・高度方向のextentも変わる。壁・高度制限を追加した実験ではない。

20/40秒は初速での予測時間の増分。Referenceは加速するので、実際の情報猶予・遭遇時刻の増分は生軌跡から測る。予測は同じ数式・75m閾値を使い、horizonだけ50/70/90秒へ延ばして元の遭遇を範囲内に残す。この評価時パラメータ変更を記録し、元mainの50秒条件と混同しない。終了上限も全bankの経路時間に合わせて延長する。

## Windowsでの起動

結果ファイルを持つ`codex/main-additional-results-20261126`を土台にした実装ブランチを使う。mainだけへ切り替えて結果JSONが見えなくなる以前の問題を避ける。

```powershell
git fetch origin
git switch codex/main-route-extension
```

いつものPython環境で`notebooks/main/main_route_extension.ipynb`を開き、上から実行する。

既定パス：

- SOURCE：`results/main_additional_20261120`
- OUTPUT：`results/main_route_extension_20261127`

SOURCEの`suite.json`に記録された親実験からA/B75、追加実験子ディレクトリからB50を読む。export zipを展開した場合は`SOURCE/models/A|B75|B50/seed_XXX/model_30000.pt`も使える。不足・方式・半径・学習予算・元rolloutのモデルhashが異なる場合は停止し、別モデルで代用しない。

## 既定の実験量と段階的な追加

| 設定 | 既定値 | 意味 |
| --- | --- | --- |
| BANK_PER_N | 100 | 最初に固定する独立元形状/N |
| COUNTS | 2/4/8/16 | 開始機数 |
| LEADS | 0/20/40秒 | 同一元形状の延長条件 |
| EVAL_PER_N | 10 | 今回評価する元形状/N |
| SEEDS | 0/1/2 | 既存学習seed |
| EVALUATION_DEVICE | cpu | 小型ActorとCPU安全QPを直列評価 |

bankは400元形状×3延長=1,200シナリオ。最初はそのうち40元形状×3延長×3方式×3seed、**1,080episode**を評価する。

`EVAL_PER_N=100`に増やせば同じbank・同じOUTPUTで**10,800episode**まで追加できる。完了ケースは再実行しない。BANK_PER_N/LEADS/COUNTS/生成条件を変える場合は別OUTPUTが必要。

最初に最小/最大N×最短/最長延長の4シナリオをseed0の3方式、計12caseで確認する。短時間確認だけにする場合は`RUN_FULL=False`。この確認結果も全量評価で再利用する。RUN_FULL=Trueが既定で、その後は今回の1,080episodeへ進む。

CPU/GPU、N、航路長によって速度が変わるため、全量の所要時間はWindowsの確認実行で判断する。Macの合成テスト時間を本番見積もりには使わない。

## 保存・進捗・中断再開

- case数の進捗と、飛行中の実step数を別のtqdmで表示。
- 1モデルずつロード、1caseずつ実行。並列workerや巨大Replay保存は行わない。
- caseごとに生NPZ・指標・出典を保存し、最後に完了markerを原子的に書く。
- 中断された1caseはやり直す。完了marker、条件identity、生NPZの初期条件が一致したcaseは再実行しない。
- モデルhash・bank・runner・実行デバイスが変わった保存先への混在を拒否。
- 途中でも`status(OUTPUT)`で集計・CSVを更新できる。
- timeoutを含むepisode時間と、成功時の完了時間・経路比を区別。失敗ケースを黙って除外しない。

主な保存物：

- `experiment.json`：固定条件、bank digest、source、最大step。
- `banks/base.json`：独立元形状。
- `banks/route_extension.json`：全延長条件の共通シナリオ。
- `models_manifest.json`：使用したA/B75/B50の実ファイル・hash。
- `evaluation/mean/<method>/seed_XXX/cases/<id>/`：生NPZ、episode指標、完了marker。
- `episodes.csv` / `result_status.json`：途中を含む全件・N/延長別集計。

## 結果をMacへ運び出す

Notebook末尾の`export_results(OUTPUT, include_models=True)`を実行する。既定では

`results/main_route_extension_20261127_results.zip`

を出力する。途中停止後にもこのセルだけ実行できる。zipには生NPZ、bank、条件・出典、CSV、**使用した評価用モデル**を含める。再開用Replay/optimizerの巨大checkpointとruntime一時ファイルは除外する。`export_manifest.json`で各ファイルのSHA256とbyte数を確認できる。

モデルだけでも各約12.5MB、全9個で約113MBになる。生軌跡が多いので実zipを直接転送する。Gitの結果コミットだけでは、gitignore対象のモデルやzipが移らないことに注意する。

## 解析上の扱い

- 比較するのはA/B75/B50の3方式のみ。全方式は同じbank、初期条件、能力、目標、候補定義を使う。
- 同じ元形状の0/20/40秒条件は対応標本で、独立形状として水増ししない。
- 指定危険ペアが接近しなかったケースも残す。接近したペアだけに絞ると予防成功を落とす。
- 未割引距離Cost、介入時間/強度、可解性/fallback、情報取得、Actor変化、到達効率を解析する。
- 大きいLEADでは割引Costが時間経過だけで小さくなるため、それだけで改善と判断しない。
- CSVの違反件数は記録点の判定。従来同様、生NPZから区間内距離まで後処理できる。
- 大きい位置・目標距離は学習範囲外にもなる。結果は通常mainの平均性能や未使用最終testとは分けた追加ストレス評価として扱う。

## 検証

新規テストは、CPA距離保持・接近時刻シフト・初期間隔、bank再現性と改変拒否、frozenモデル評価、step表示による軌跡不変、途中再開、Notebook全セルの小規模実行、partial zipとモデル同梱、モデル変更拒否を確認する。既存追加実験のテストも併走する。

実データのbank生成確認は、各N2元形状で8元形状・24条件を生成し、最小初期距離31.52m、40秒延長時の指定ペア初期距離は最小227.18m、水平extent最大約943mを確認した。学習済みmainモデルの実rolloutはMacに重みがないため未実施で、Windowsで行う。
