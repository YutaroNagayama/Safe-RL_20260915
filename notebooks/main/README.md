# Main実験の実行入口

研究ロジックは`notebooks/pre_main_gate.ipynb`と共通のTrainer/SAC/HOCBFを使います。main固有の処理は`uav_safe_marl.experiments.main`へ分離しました。旧Windows Bookは旧条件のまま保持しています。

設定は`settings.yaml`。初回はA/B/Fを3seed・30k、n10・episode dual・Cost-only warm-upで学習します。Reactive通信半径75mとHOCBF監視距離50mは独立です。C/DはA/Bの同じcheckpointのOFF評価です。

`main_experiment.ipynb`でプロジェクト仮想環境のカーネルを選び、上から学習セルまで実行します。既定の `TARGET = 30000` で共通bankを準備し、A/B/F各3 seedをカーネル内で1 runずつ学習・development評価します。個別コマンドを使う場合はリポジトリ直下で実行します。

学習前の「学習デバイスの確認」セルは、学習に使うカーネル上でPyTorchのデバイスを解決します。Windowsでは、`cuda:0` などの番号と目的のGeForceの製品名を確認してから学習セルへ進みます。評価用デバイスと、カーネルのPythonパス・PyTorch・CUDAビルドも表示します。このセル単体ではbank生成や学習は始まりません。

```powershell
python -m uav_safe_marl.experiments.main prepare --root results/main_four_faces_free_30m
python -m uav_safe_marl.experiments.main train --root results/main_four_faces_free_30m --method A --seed 0
```

実行中の1 runを同じtqdmバーに表示します。warm-up収集は完了episode数、Cost Critic warm-upは更新数、online学習は累積step数と完了episode数、developmentは評価ケース数で進みます。チェックポイントの読込・保存中はファイル名が出ます。旧版の並列学習から切り替える場合は、旧Notebookを中断してworkerの終了を確認してから同じ出力先で再開します。直近の完全チェックポイントから再開し、完了済みの組を再利用します。保存済みモデルのdevelopmentが欠けている場合は評価を補います。60kへ延長するときは同じ出力先で `TARGET = 60000` に変更します。decision評価・確率校正・集計はそれぞれのセルで有効にします。

各seedでA/B/Fをそろえ、初回は全9 runを30k到達後のepisode終端で止めます。固定developmentとN別学習曲線で、方式差の方向がseedをまたいで揃うか、終盤の指標がまだ動いているかを確認します。30kで判断できれば同時点のモデルをdecisionで比較し、不足なら全9 runを累積60kへ延長します。確認Testは凍結まで温存します。warm-up収集・Cost-only更新、30k（延長時のみ60k）保存と別モデルによるdevelopment評価は自動です。中断したら同じtrain呼出しで再開します。warm-upは収集終了と500更新ごと、onlineは従来のepisode間隔と段階終了でfull checkpointを保存します。

判断用評価は、各seedでA/B/C/D/F/F_OFF、固定対照E/Rを実行します。E/Rはseed0だけです。

```powershell
python -m uav_safe_marl.experiments.main evaluate --root results/main_four_faces_free_30m --method C --seed 0
python -m uav_safe_marl.experiments.main evaluate --root results/main_four_faces_free_30m --method A --seed 0 --stochastic
python -m uav_safe_marl.experiments.main analyze --root results/main_four_faces_free_30m
```

確率評価はA/B/Fだけで、各Criticと同じHOCBF ON・同じ初期sample actionを使います。rolloutは1件ずつ保存し、完了済みはidentityを確認して再利用します。未完了分だけ同じ乱数キーで再実行します。設定の`evaluation_workers`を増やすとscenario単位で並列化し、workerごとにモデルを1回だけ読み込みます。初期値1は確実な直列動作のためで、本番機で計測して固定してください。

## Windowsで学習の同時実行数を測る

リポジトリ直下で、CUDA対応のPyTorchを入れた仮想環境から実行します。

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_main_parallel.py
```

同じシナリオで、1・2・3プロセスがそれぞれ同じRTXを使う場合の合計step/秒を比較します。標準はA・seed 0の設定で、各機数2 episode分のReplayを事前に収集し、勾配更新ありの約2,000 online stepを測ります。各プロセスは独立したReplayを持ちます。計測は `results/parallel_benchmark/` 以下に保存され、`benchmark.json` で速度、Replay件数、Windowsの空きRAM、取得できた場合はGPU使用率とVRAMを確認できます。

初回の候補が決まったら、`--levels 1 2 --collect-per-n 25 --steps 6000` などでReplayを増やし、候補を絞って再計測できます。各levelの結果は終了ごとに保存します。ベンチマークではCost Criticの事前更新とdevelopment評価を省きます。本番30k終盤のメモリや、長時間の安定性は別途本番ログで確認してください。

## 継続と分岐

```powershell
python -m uav_safe_marl.experiments.main train --root results/main_four_faces_free_30m --method A --seed 0 --target 60000
python -m uav_safe_marl.experiments.main evaluate --root results/main_four_faces_free_30m --method C --seed 0 --target 60000
python -m uav_safe_marl.experiments.main analyze --root results/main_four_faces_free_30m --target 60000
```

延長は設定ファイルを書き換えず`--target`で指定します。変更を許可するのは累積step予算だけで、Replay・開始状態buffer・optimizer・乱数・scenario cursorを継承します。延長を決めた場合はA/B/Fの全9 runを同じ累積60kへそろえます。warm-upはやり直しません。

学習率を変える場合は親を残して別runへ分岐します。

```powershell
python -m uav_safe_marl.experiments.main fork --parent results/main_four_faces_free_30m --root results/main_lr_branch --method A --seed 0 --learning-rate 0.0001
python -m uav_safe_marl.experiments.main train --root results/main_lr_branch --method A --seed 0 --target 80000
```

この分岐はActor/Critic学習率だけを変更し、Adamのmoments、Replay、乱数、累積カウンタを継承します。親checkpointのhashを記録します。Reward/Cost・構造・実装hashの変更を一律に許す機能ではありません。そうした変更には別の移行設計が必要です。

確認Testは通常評価では開きません。固定後に使うときだけ`--split confirmation_test --confirm-test`を明示します。decisionを調整に使った場合、それを未知Testとは呼びません。

## pre-main相当の分析に使える保存物

- 学習：episode JSONL、Cost/Reward/lambda/entropy等の既存診断、途中モデル、完全再開checkpoint。
- 評価：全位置・速度、tick開始/終了活動mask、Reference/Residual/nominal/pure/safe入力、距離、Actor/CBF edge、情報age、送信byte、Reward/Cost内訳。
- 補強した保存：毎tick・各機の安全制約診断JSON、QP wall time、評価tick wall time。JSON文字列をNPZに格納し、pickleなしで読めます。
- 共通診断：Fでも純要求・実補正・割引共通Costを計測。早期CPA-opening、pairごとのnominal不足とsafeによる修復、近接滞在、逸脱を記録。
- 再集計：N/seed別主指標、交差paired bootstrap、Cost校正、全episode診断、主指標N依存図、Aの各Nの中央値付近/厳しい軌道図。

分析出力は`analysis/<split>/step_<target>/REPORT.md`。`summary.json`と`episodes.json`から元caseへたどれます。途中・一部の結果でも分析できますが、episode数を必ず確認してください。未実行の方式・seedは自動補完されません。独自の追加分析も保存NPZから実行でき、再飛行は不要です。

## 軽量化と限界

n-step Cost targetを使うとき、上書きされる1-step Cost targetの重複計算を省きます。既存のGraph/Replayの変更は維持し、学習則・更新頻度・ネットワーク・QPは変更しません。評価をscenarioごとに保存するため全bankの軌道をメモリへ保持しません。warm-up進捗カウンタはcheckpointと一緒に1回で保存します。

実時間の高速化率は本番GPU/CPUでは未計測です。保存データでpre-mainレベルの深掘りができますが、全ての過去の個別分析図を自動再現するわけではありません。低levelの変更を含むので、長時間学習前に小規模の再開・評価確認を行ってください。

## 四方流入と途中結果

新しいmainは `x_min/x_max/y_min/y_max` の水平4面を使います。各機の進入面と退出面が異なれば、隣接面への退出も同じ面からの複数機の進入も可能です。面ごとの機数は固定しません。進入点同士・退出点同士はそれぞれ30m以上離します。旧bankとは別に `results/main_four_faces_free_30m` を作ります。二面の結果やcheckpointはこの実験へ混ぜません。

- 学習中のepisode別指標：`training/<方式>/seed_<seed>/online/metrics.jsonl`。episode完了ごとに追記。
- 再開用：同seedの `latest.pt`。onlineは25 episodeごと、および段階終了時に完全状態を保存。直近保存後に中断した分は、再開時にその保存点からやり直します。
- 途中評価用：`model_30000.pt`、`model_60000.pt`。目標stepを越えたepisode終了時に保存し、固定developmentで自動評価します。正確な到達stepはcheckpoint側に記録します。
- 30kの結果を取り出す：`analyze --root results/main_four_faces_free_30m --split development --target 30000`。実行済みの方式・seedだけでも集計できます。未完了の組を含む結果は暫定扱いです。

decision評価の既定モデルは30kです。60kへ延長した場合は `evaluate` と `analyze` に `--target 60000` を指定します。confirmation_testは、設定・モデルを凍結してから開きます。

四方化と同時に、実行可能なnominalのQP列挙省略、Graphの不要なpad/連結削減、n-step Replayの不要な終端整形削減、開始状態抽出のlist変換削減、warm-up復帰時の重複読込解消を適用しました。固定Replayの差分保存は今回は採用せず、従来どおりfull checkpointを保持します。
