# Windows追加評価：通常条件の量拡大とA/Bs空間拡張

## 実行するもの

`notebooks/main/main_route_extension.ipynb`を上から実行する。新規学習は行わず、既存30kモデルを固定する。

| 実験 | 方式 | 初回の評価量 | 保存先 |
| --- | --- | --- | --- |
| ①元条件の確認評価 | A/B75/B50各3seed＋E各形状1回 | 各N25形状、計100形状・1,000episode | results/main_confirmation_20261127 |
| ②空間拡張1条件 | A/B75/B50各3seed | 各N10形状、計40形状・360episode | results/main_spatial_extension_single_20261127 |

**合計1,360episode**。従来案の4,000＋1,080episodeから削減した。試走は本実行で再利用するため追加のepisodeではない。①を完了・zip保存した後、②へ進む。方式・seed・形状は直列実行し、並列workerは使用しない。学習・制御のcoreと既存main/additional/航路延長runnerは変更していない。

## ① 元条件の評価量拡大

実行処理は`scripts/main_confirmation_runner.py`。SOURCEの既存`banks/confirmation_test.json`を元manifestのSHA256と照合し、そのまま利用する。元の四方交通、初期離隔30m、機体性能、予測horizon50秒、最大stepを維持する。低CPAや結果で形状を選別しない。

bank自体は各N100形状で固定し、初回は事前に決めた各N25形状だけ評価する。既存分析の48形状から100形状へ増やす。必要なら`CONFIRM_EVAL_PER_N`を50、100へ増やし、同じ保存先へ未完了分だけ追加できる。結果を見て形状を選び直さない。

EはReference＋HOCBFで構築し、各形状1回だけ評価する。3seed分の独立標本として水増ししない。比較方式・指標・CPA層別・集計方針を`experiment.json`の`analysis_plan`に固定する。N別を主に、全N等重み集計を併記する。

GitによるLF/CRLF変換は、元hashに完全一致するバイト列へ復元できる場合だけ許容し、内容変更は拒否する。

## ② 空間拡張は1条件だけ

実行処理は従来の`scripts/main_route_extension_runner.py`。Notebookは`LEADS=(40.,)`とし、**0/20/40秒の条件比較は行わない**。全方式は同じ拡張形状・同じ能力・同じ目標を使う。E/R、HBA/F、HOCBF OFFは②に追加しない。

四方交通の元形状から予測CPA<10m、予測接近時刻>1秒のペアを含む形状を生成する。N=2/4/8は少なくとも1ペア、N=16は少なくとも2ペアを指定危険ペアとする。開始地点を

`p_extended = p_original - v_initial * 40`

で航路沿いに後方へ延ばし、交差領域・出口・ゴール・初速・能力を保持する。40秒は延長距離を決める内部的な換算量であり、時間別の実験群ではない。予測CPA距離を保ち、接近時刻が40秒後ろへずれることを検査する。全ペアの初期離隔30m、指定ペアの初期距離150m以上も検査する。面ごとの機数割当は強制しない。

3D航路に沿った延長なので水平・高度方向のextentが変わり、出発点は共通の大きな正方形外周には揃えない。壁・高度制限は追加しない。Referenceは加速するため、実際の猶予は生軌跡で確認する。

従来の拡張仕様として、予測horizonは90秒へ、終了上限も長い航路に合わせて延長する。通常条件の50秒horizonとは区別して記録する。実際の空間寸法・延長量・初期条件をすべて保存する。

`BANK_PER_N=100`で固定し、初回は`EVAL_PER_N=10`だけ実行する。必要なら同じbankで評価量だけ追加できる。旧0/20/40秒版とは保存先を分けるため、旧結果を消す必要はない。

## Notebookの設定と起動

いつものPython環境でNotebookを開く。SOURCEは`results/main_additional_20261120`。親実験のA/B75、追加実験のB50、またはSOURCE/modelsの正規名モデルを利用する。不足・方式・半径・予算・モデルhashが違う場合は停止し、別モデルで代用しない。

| 設定 | 既定値 |
| --- | --- |
| RUN_CONFIRMATION / RUN_EXTENSION | True / True |
| CONFIRM_BANK_PER_N / CONFIRM_EVAL_PER_N | 100 / 25 |
| BANK_PER_N / EVAL_PER_N | 100 / 10 |
| COUNTS / SEEDS | (2,4,8,16) / (0,1,2) |
| LEADS | (40.,) の1条件 |
| EVALUATION_DEVICE | cpu |
| RUN_PROBE / RUN_FULL / RUN_EXPORT | True / True / True |

①は最小/最大Nの2形状×4方式＝8case、②は最小/最大Nの2形状×3方式＝6caseをseed0で試走し、本実行で再利用する。`RUN_FULL=False`なら試走だけ。片方のRUNフラグをFalseにすればもう片方だけ実施できる。

## 進捗・保存・持ち帰り

- case数と飛行中の実step数を別のtqdmで表示する。
- 1モデルずつロードし、新規学習や巨大Replay保存を行わない。
- 各caseで生NPZ・指標を保存し、最後に完了markerを原子的に書く。
- 途中再開時は条件identityと生NPZの初期条件を照合し、完了caseを再実行しない。中断された1caseは再実行する。
- bank・モデル・runner・デバイスが変わった結果の混在を拒否する。
- 途中停止後も各実験の集計・取り出しセルを単独実行できる。

zipは別々に出力する。

- `results/main_confirmation_20261127_results.zip`
- `results/main_spatial_extension_single_20261127_results.zip`

生NPZ、JSON/CSV、bank、条件・出典、評価用モデル、全ファイルのhash/byte数を同梱する。Replay/optimizer再開checkpointとruntime一時ファイルは除外する。Gitの結果JSONだけでは、gitignoreされたモデルやzipは転送されないので、zip自体をMacへ運ぶ。

## 解析と検証

通常条件と低CPAを多く含む空間拡張結果を混ぜて平均しない。指定危険ペアが接近しなかった予防成功例も残す。初期CPA別の接近距離分布・接近率、介入時間/強度/割引および未割引Cost、QP可解性/fallback、到達効率、通信を解析する。延長条件の割引Costは接近時刻の遅れでも小さくなるため、未割引の負担と併記する。

成功限定の経路比/完了時間と、timeoutを含むepisode時間を分ける。CSVの違反判定は記録点の判定で、生NPZから区間内最近接も後処理できる。

合成の小規模モデルで、bank同一性・改変拒否・LF/CRLF復元、4方式の同一初期条件、進捗表示による軌跡不変、E非重複、再開・段階追加、モデル変更拒否、モデル付きzip、統合Notebook全セルを確認する。航路延長の旧複数条件も後方互換の検証として残す。本番の学習済みモデルでの実走と所要時間はWindowsで確認する。
