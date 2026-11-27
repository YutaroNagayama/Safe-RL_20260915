# main_v1.1を含むWindows一括実行

Notebookは`notebooks/main/main_route_extension.ipynb`。上から実行すると、次の3工程が直列で進む。

| 工程 | 対象 | 初回の量 |
| --- | --- | --- |
| ①通常条件の評価量拡大 | 固定main_v1.0 A/B75/B50各3seed＋E | 1,000episode |
| ②空間拡張1条件 | 固定main_v1.0 A/B75/B50各3seed | 360episode |
| ③通常航法の追加学習と比較 | A_main_v1.1各3seed、E/A_v1.0/A_v1.1 | 下記の収集・学習＋294評価episode |

v1.0とv1.1の保存先・モデル・zipを分ける。v1.0の制御・学習engine、既存評価runnerは変更しない。新しい処理は`scripts/main_v11_runner.py`と`scripts/main_v11_backend.py`に置く。

## v1.1の目的と範囲

危険な接近を予防するA_v1.0の効果を残し、単機・非接近で通常航法へ加わる不要な迂回・減速・上下動を減らせるか検証する。初回は小規模な分岐評価であり、統計的な非劣性の証明ではない。

報酬・Cost・予算・HOCBF・Actor周期は変更しない。操作平滑化、新しい急変ペナルティ、フィルタは将来課題に残す。B群/HBA/Fをv1.1として追加学習しない。

## 初回設定

- `RUN_V11=True`かつ`RUN_FULL=True`で実行する。`RUN_FULL=False`ならv1.0の試走だけにし、v1.1は開始しない。
- `V11_TARGET=5000`：各seedの追加online step。到達後のepisode終端まで実行するので実数は少し超える。
- `critic_updates=500`：Actor・λ・αを固定したCritic修正。
- `replay_capacity=4000`：単機・非接近・接近の各Replayの容量。元の巨大Replayを読み込む必要はない。
- 非接近は初回N=2/4だけ、接近は従来N=2/4/8/16。高密度で全ペアCPA≥75mを要求する生成負担を避けつつ、接近条件の全N評価を残す。
- 学習deviceはCUDAが利用できればcuda、利用できなければcpu。Notebookに実際のGPU名を表示する。評価は既定cpu。
- 保存先は`results/main_v1_1_navigation_20261127`。

3seed計15,000追加online step。以下の固定方策収集stepと評価stepは別に記録する。所要時間は実機依存なので固定の時間見積もりは置かない。

## 重み継承と経験収集

各seedのA_v1.0の30kモデルを方式・設定・hashで照合し、Actor・Critic・target・optimizer・α・λを継承する。親の実step数はモデルのprogressから記録する。

旧Replayは継承せず、固定したA_v1.0で従来シナリオを再収集する方式を採用する。したがって、v1.0をReplay/乱数まで厳密に再開したものとは扱わない。この方針はmanifestの`replay_strategy`へ記録する。

各seedの初期収集は、単機4形状をReference/Actorの両方で実行して8episode、非接近をN2/4各1形状で2episode、接近を全N各1形状で4episode、合計14episode。モデルは収集中に更新しない。Referenceは残差ゼロの実入力で収集し、架空の遷移を作らない。

単機は同じ四方交通生成器の1航路を切り出し、初速・能力・ゴール・方向を保持する。由来も保存する。非接近は全ペアの初期予測CPA≥75mを満たす形状だけを採用し、上限まで生成できなければ停止する。Actorがその後に作った接近はCostとして残す。

## 更新と固定λ

勾配更新を単機・非接近・接近・接近の順に回し、学習サンプルを25/25/50%にする。オンラインの環境収集量の比率やNの比率とは区別する。Reward/Costそれぞれのn-step窓は同じdomainのepisode-safe Replayから取得し、異なるepisodeを連結しない。

単機ではReward Criticとそのtargetだけを更新し、Cost Critic/optimizer状態/targetは維持する。Actor段階でも単機のCost項は除外する。複数機のBellman target・SAC目的・CAL式・Actor周期はv1.0と同じ。λは親の最終値で固定する。Critic段階はαも固定、Actor段階は従来のα自動調整を維持する。

CALの開始状態評価には、元のtraining bankの開始状態だけを、元のN分布で抽出して使う。単機や非接近の新しい開始状態は混ぜない。元の経験bufferを再現する方式ではなく、元のtraining bankを基準にする方式である。固定λでもCost推定に応じて実効CAL係数は変わる。

オンライン更新は既存Trainerの遷移確定境界で行い、元設定のupdates_per_environment_stepを維持する。報酬、HOCBF、通信、保持されたActor入力、次時刻位相を変更しない。

## 共通評価と初回の判断

学習・development・confirmationは形状を分離する。元実験の全bankとの形状重複も検査する。v1.1用のdevelopment/confirmationは各domain/N25形状を最初に固定する。

初回はdevelopment各2形状、confirmation各4形状で評価する。単機1区分、非接近N2/4の2区分、接近全Nの4区分、計7区分なので、14＋28＝42形状。Aの両版各3seed＋E1回で、**294episode**。先に実行するA_v1.0/Eのdevelopment評価もこの内数で、追加分ではない。

まずA_v1.0/Eの単機return差を保存してから追加学習する。これは実際の割引環境報酬の比較であり、エントロピーを含むSAC soft-Qの校正とは区別する。結果を見て自動で報酬・予算を変えない。

到着時間・経路長・実加速度・二乗加速度積分・直線逸脱、初期危険ペアの最近接、20m未満接近、区間内10m未満違反、QP解なし、介入時間/Costを保存する。成功時到着時間と、timeoutを含むepisode時間を分ける。連続区間距離は記録位置間の線形補間に基づく。

`paired_comparisons.json`でE/A_v1.0/A_v1.1の対応差を出す。Eを各seedの独立反復として水増ししない。安全指標を確認せず、通常航法の改善だけで成功としない。今回の小規模結果だけで性能維持を証明したとは扱わない。モデル選択に利用した形状は、その後は未使用testと呼ばない。

## 進捗と再開

収集episode/step、Critic update、オンラインstep/episode、評価case/stepをtqdm表示する。学習指標は`updates.jsonl`へ残し、TensorBoardが導入されていれば`training/A/seed_XXX/tensorboard`にも記録する。

- `resume.pt`：Replay・backend/optimizer・乱数・固定開始状態・進捗の完全なv1.1再開snapshot。収集4episodeごとと段階終端、online約1,000stepごとのepisode終端に保存する。
- `critic_progress.pt`：固定Replay snapshotのhashに紐づけた小さいCritic途中状態。100updateごと。巨大Replayを毎回保存しない。
- `model_latest.pt`：onlineのepisode終端ごとの途中モデル。
- `model_5000.pt`：指定追加targetに到達した時点の固定モデル。実online stepと親hashをmetadataに持つ。

同じ設定・保存先でNotebookを再実行すると完了工程とcaseを再利用する。飛行中や最後の完全snapshot以降の学習は、そのsnapshotからやり直す。中断・再開と連続実行のbackend一致は合成テストで確認する。

`V11_TARGET=10000`などに増やせば、同じrunを追加学習できる。旧targetのモデル・評価は別ラベルで維持する。評価件数も`V11_DEV_PER_N`/`V11_TEST_PER_N`を25以下で増やせる。学習仕様やデバイス変更時は別V11_OUTPUTを使う。

## 結果の持ち帰り

末尾のセルで`results/main_v1_1_navigation_20261127_results.zip`を作る。途中停止後にも集計・exportセルを実行できる。Aの親/子モデル・E、全bank、生NPZ、JSON/CSV、学習ログ、出典、全ファイルhashを同梱する。巨大Replayや再開snapshotは含めず、Windows側に保持する。

v1.0の通常条件・空間拡張・v1.1のzipは3つに分ける。v1.1の途中モデルと完了モデル、実step数とtargetラベル、親のmain_v1.0を識別できる構成とする。
