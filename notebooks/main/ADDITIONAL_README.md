# Main追加実験のWindows実行

入口は **`notebooks/main/main_additional.ipynb`**。旧 `main_experiment.ipynb` はA/B/旧Fの初回実験用として保持しています。既存の実行を終了した状態で、追加用Notebookのカーネルを起動し、上から実行してください。

## 最初に確認するもの

設定セルの既定値は次のとおりです。

```python
PARENT = ROOT / "results" / "main_four_faces_free_30m"
OUTPUT = ROOT / "results" / "main_additional_20261120"
TARGET = 30000
```

PARENTには元の `experiment.json`、全bankの実体、`training/{A,B,F}/seed_00{0,1,2}/model_30000.pt` が必要です。旧FはHBAに対応します。Gitで取得した結果だけではモデルやbankが含まれない場合があるので、学習を行ったWindows実機の元ディレクトリを指定します。

OUTPUTはPARENTと独立した新しい保存先にします。PARENTの設定・モデル・結果は書き換えません。bankは原本のファイルをコピーし、hashを照合します。準備セルでは学習しません。

デバイス確認セルで、`cuda:0` などの番号と実際のGeForce製品名、カーネルのPython、PyTorch/CUDAを確認します。追加学習は元実験のdevice設定を引き継ぎ、評価は元実験のevaluation_deviceで直列実行します。

## 通常の実行内容

上から実行すると、次の順に進みます。

1. 既存モデルのA/B75/HBA/C/D75/HBA_OFF/E/Rについて、同じdevelopment 48形状を揃えます。元のcheckpoint、bank、ON/OFF、実行条件が一致する既存ケースを取り込み、未完了分だけ飛ばします。
2. seedごとに正規F→B50を1本ずつ学習します。FはPrior・距離Costのみ・予算1.5。B50はReactive半径50m・合成Cost予算6。各3 seed、30k到達episode終端で停止します。
3. 正規F/F_OFF、B50/D50をdevelopmentで揃えます。学習直後のON評価は再利用します。
4. 正規Fの確率方策評価を、decisionの10形状/N、各3反復で行います。初期actionをCriticとrolloutで一致させ、距離Cost予算1.5の校正材料を保存します。

各方式で初期位置・速度・ゴール・機体能力・参照距離・候補graphを照合します。warm-up 100形状、training poolと供給順、development 48形状、decision部分集合は元bankを共用します。30kで停止するため、方式ごとの消化episode数までは一致しません。

現在のON条件はA/B75/B50/HBA/F、OFF条件はC/D75/D50/HBA_OFF/F_OFF、固定対照はE/Rです。E/Rは各形状1回で、3 seedに複製しません。D75は従来D、HBA_OFFは従来F_OFFに対応します。正規FとHBAは別のcheckpoint・保存先・identityです。

既存WindowsのA/B75/HBA development 432本を再利用できる場合、全方式developmentを揃える追加飛行は1,104本です。正規Fの確率評価360本を加えると、通常実行の新規評価は1,464本です。Macで補ったE/RはPARENTのWindowsケースとして取り込みません。

## 任意の評価と途中取得

設定セルのフラグで段階を選べます。

|フラグ|既定|実行内容|
|---|---|---|
|`RUN_LEGACY_ROLLOUTS`|True|既存モデルで可能な8条件のdevelopment|
|`RUN_TRAINING`|True|正規F/B50の追加学習6 run|
|`RUN_ADDITIONAL_ROLLOUTS`|True|正規F/F_OFF/B50/D50のdevelopment|
|`RUN_F_CALIBRATION`|True|正規Fの確率方策評価|
|`RUN_DECISION`|False|全12条件をdecision 120形状でmean評価|
|`RUN_ALL_CALIBRATION`|False|ON5方式の確率評価。実行済みFは再利用|
|`RUN_EXPORT`|False|結果と小型モデルの持ち帰り用ZIP|

学習が不要な再実行では`RUN_TRAINING=False`にできます。同じOUTPUTで再実行するとケース単位で再利用します。確認Testはこの入口では実行できません。TARGETや学習条件・実装を既存OUTPUT内で変更すると拒否します。60kへの自動延長はありません。

warm-up収集は完了ep数とepisode内step、Cost Criticは更新数、onlineは累積stepと完了ep数、評価は完了ケース数を表示します。読込・Replay差分保存の段階も表示します。表示は各grid内で同じtqdmバーを更新します。

`evaluation/<split>/step_30000/<mean|stochastic>/<方式>/seed_XXX/cases/`に各ケースの生NPZと`completed.json`があります。最後のセルの`status()`は途中結果だけでも読み取れます。`result_status.json`に、実行件数・期待件数・未完了development条件を保存します。

## 保存と再開

追加runの場所は次のとおりです。

```text
OUTPUT/experiments/F/training/F/seed_XXX/
OUTPUT/experiments/B50/training/B/seed_XXX/
```

小型の`latest.pt`にモデル・optimizer・乱数・cursor・Replayの参照を保存し、変更されたReplayの物理slotを不変のchunkへ追記します。収集とonlineの保存で既存データを再利用し、巨大Replay全体を毎回書き直しません。NumPy配列の保存にはpickle protocol 5を使い、精度や学習更新式は変えません。

**再開時はrunディレクトリ全体を保持してください。** `latest.pt`だけでなく、その中で参照する`warmup/collection_replay`や`latest_replay`等のchunkが必要です。chunkはhashを照合して読み込みます。小型checkpointはchunk保存後にatomicに確定するため、保存途中の中断でも直前の確定点を維持します。

既存の完全checkpointも読めます。追加runの差分形式は追加用Notebookから再開し、従来のCLIへ渡さないでください。`model_30000.pt`は従来同様、Replayを持たない独立した評価用モデルです。

学習の完全再開点は25 episodeごとと段階終端です。それ以前に中断した未保存部分は直近の確定点から再実行します。評価は1ケースずつ完了を記録するため、全ケースを飛び直す必要はありません。

## 結果を持ち帰る

`RUN_EXPORT=True`にして最後のセルを実行すると、OUTPUTの隣へ`main_additional_20261120_results.zip`を作ります。段階の途中でも、完了したケースと保存済みモデルを取り出せます。

ZIPにはbank、suiteと親実験のmanifest、方式対応、元の学習ログ、追加学習ログ、評価生NPZ、ケースidentity・来歴、ON5方式の保存済み小型モデルを含みます。モデルは`models/HBA`と`models/F`等の表示名で分けます。ファイル別SHA256とサイズは`export_manifest.json`に保存します。

巨大Replay・再開用checkpointはZIPへ入れずWindowsに保持します。ZIPをMac側へ移せば、元のWindows絶対パスが存在しなくても保存済み結果の`status()`や軌跡解析は可能です。追加学習・再飛行の再開にはWindows上の元ディレクトリが必要です。

## 実装の互換性

追加の管理処理とReplay保存は`scripts/main_additional_runner.py`と`scripts/main_incremental_checkpoint.py`に置き、`uav_safe_marl`の学習・安全制御本体は変更していません。既存のsource hashを保ち、チェックを一律に無効化する移行は行いません。

追加suiteは実行管理コードのhashも保存します。実行途中でコードを更新した場合、同じOUTPUTで無条件には再開しません。まず変更内容を確認し、互換性を監査した移行が必要です。
