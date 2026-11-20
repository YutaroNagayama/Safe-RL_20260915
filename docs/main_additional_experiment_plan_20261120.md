# Main追加学習とWindows共通評価の計画

2026年11月20日。正規Fの距離予算1.5とB50を追加し、D50を含む全方式をWindowsで評価する計画。追加学習とロールアウトの実装は[追加用Notebook](../notebooks/main/main_additional.ipynb)と[Windows実行手順](../notebooks/main/ADDITIONAL_README.md)に反映した。小規模の学習・再開・全条件評価で検証し、Windows本番の追加学習・評価は未実施。既存A/B75/HBAの30k結果とモデルを保持する。

## 追加学習

|方式|通信|学習Costと予算|学習時HOCBF|初回学習量|
|---|---|---|---|---|
|正規F|Prior、候補CPA閾値75m|距離のみ、予算1.5|ON|seed 0/1/2、各30k|
|B50|現在距離50m以内|距離＋純介入、予算6|ON|seed 0/1/2、各30k|

追加は6 run、180,000 online stepを基本とする。各runは30k到達後のepisode終端まで実行し、超過分を保存する。方式ごとに固定Actorで100形状を収集し、Cost Criticのみ5,000更新する。warm-upはonline予算に含めない。

正規Fは新規初期化し、HBAの30kから予算を変えて継続しない。同じseedの初期Actor/Encoderは既存方式との対応を監査する。B50にもB75の学習済みActorやReplayを流用しない。既存warm-upの再利用は本計画の前提にしない。

ネットワーク、SAC、報酬、時間Cost、n-step、学習率、更新頻度、学習開始2,000 step、Replay容量などは現在のmainを継承する。AのCPA閾値75mとHOCBF監視距離50mは維持し、B50の通信半径だけ50mへ変える。Fの変更は介入Cost無効と距離予算1.5である。

距離予算1.5は、development上のEの距離Cost 2.122に約30%の低減を求める要求水準であり、HBAの1.804に対しても約17%の低減を求める。N混合開始分布に対する期待割引累積Costの条件とし、各episode・各Nが必ず1.5以下という条件にはしない。達成可能性や制約の活性化を事前保証せず、λ、有効CAL係数、Cost推定、stochastic実測を併せて確認する。実行中に予算を自動変更しない。

## 方式名と同じモデルのON OFF対応

|ON側の表示名|OFF側の表示名|モデルの由来|追加学習|
|---|---|---|---|
|A|C|既存A、各seedの30k|なし|
|B75|D75|既存B、各seedの30k|なし|
|B50|D50|新規B50、各seedの30k|B50だけ|
|HBA|HBA_OFF|旧F、各seedの30k|なし|
|F|F_OFF|新規正規F、各seedの30k|Fだけ|
|E Reference＋HOCBF|R Referenceのみ|固定Reference制御器|なし|

ON/OFFでActor、Encoder、通信方式、seed、checkpointを共通化し、実適用するpairwise HOCBFだけを切り替える。OFFでも加速度・速度の物理filterを維持し、Pure Interventionの診断を保存する。

旧実装のB/Dは表示上B75/D75に対応する。旧F/F_OFFは表示上HBA/HBA_OFFに対応する。旧ファイルやmetadataのraw IDを一括で書き換えない。正規Fと旧Fは別の保存先・実験identityで管理し、source checkpointのpathとhash、学習Cost・予算・通信半径を明示する。

比較対象は12条件。学習済み5方式のON/OFF計10条件を各3 seedで評価し、E/Rは各形状1回だけ評価する。同じE/Rを3個の独立学習結果として複製しない。

## シナリオの共通化

Windowsの既存main実験にあるbankの実体を共通の原本とし、同じ設定値から作り直す方法は使わない。追加実験用manifestから原本を参照するか、実体をコピーしてhashを照合する。

|用途|使用するbankと範囲|
|---|---|
|warm-up|既存warmup、25形状/N、計100形状、同じ順序|
|online|既存training、500形状/N、計2,000形状のpool、同じ供給順|
|development|既存の選定済み12形状/N、計48形状|
|追加のmean評価|既存decision、30形状/N、計120形状|
|stochastic校正|既存decisionの固定10形状/N、各3反復|
|最終確認|confirmation_testは今回開かず温存|

四方着発、面ごとの割当なし、開始点同士・ゴール同士30m以上、異種性能、N=2/4/8/16、共通timeoutを維持する。位置・速度・ゴール・能力・参照距離・候補graphを照合する。

onlineは同じbankと供給順を使うが、30k stepで止めるため、方式によってepisode長と消化形状数は異なる。形状数まで強制的に揃えるための学習量変更は行わず、実際のstep、episode、消化形状を記録する。評価では全方式を同じ形状集合で完了させる。

## Windowsでの実行順

1. 既存bankとA/B75/HBAの評価用モデル各3 seedを確認し、hashと方式identityを固定する。既存source/checkpointの整合性検査を一律に無効化せず、互換性を検証する。
2. 追加学習F/B50を、seedごとに1 runずつ直列実行する。Notebookで学習デバイスを確認し、warm-up収集ep、Cost更新回数、online累積stepと完了epをtqdmで表示する。
3. 各新規runの30k到達時に、既存と同じdevelopment 48形状を自動評価する。追加学習の途中でも完了したrunを解析できるよう保存する。
4. 全方式についてdevelopmentの不足条件を補う。C/D75/D50/HBA_OFF/F_OFFとWindows版E/Rを含める。実装したNotebookでは既存モデルで可能な評価を追加学習より先に実行し、追加モデルの評価を学習後に補う。
5. 全方式を共通decision 120形状でmean評価する。developmentより多い形状で、差の方向、N依存、seed差、裾、解なしと分離距離を確認する。
6. A/B75/B50/HBA/FのON条件について、固定decision部分集合でstochastic校正を行う。初期sample actionをCriticと初回rolloutで一致させる。Notebookの通常実行では正規Fの校正をdevelopmentの次に優先し、全方式decisionと残りの校正はフラグで追加する。
7. 段階ごとに集計と結果exportを行い、Mac側で読込・件数・hashを検査する。confirmation testは条件とモデルを凍結した後の別工程とする。

Windows内で完了済みのA/B75/HBAのdevelopmentは、checkpoint・bank・評価条件・完了identityが一致すれば再利用する。今回Macで補ったE/Rは参考結果として保持し、共通評価にはWindowsで実行した結果を使う。

評価は各ケース完了ごとに保存し、中断後はidentityを照合して未完了ケースだけ再開する。既定は評価CPU・1 worker、学習も1 runずつとする。確認・分析用の各評価段階は独立して選択でき、一括実行も可能にする。

## 評価量

|評価段階|構成|全完了時の本数|
|---|---|---:|
|development mean|48形状×（学習済み10条件×3 seed＋E/R）|1,536|
|decision mean|120形状×同じ32実行条件|3,840|
|stochastic校正|5学習方式×3 seed×40形状×3反復|1,800|
|合計|上記の段階を全て実施|7,176|

既存WindowsのA/B75/HBA development 432本を再利用する場合、新規評価は計6,744本。Mac版E/R 96本をWindows実行済みとして数えない。学習と評価の進捗・時間は別々に表示する。

評価量は現在の取得済み432本より大きくなるため、development完了時点で一度結果を取り出せる構成にする。decisionと校正は別ステージで起動できるようにし、途中中断による再実行の浪費を防ぐ。本計画で60k延長を自動追加しない。

## 比較の狙い

|比較|主な問い|
|---|---|
|A vs B75|反応型にも早期取得の余裕がある場合、予測共有の追加価値はあるか|
|A vs B50|反応型の取得が監視開始付近になる場合、予測共有は先回りの利益を出すか|
|B75 vs B50|Reactiveの通信半径が介入・安全・効率へ与える方式全体の差|
|A vs 正規F|介入項と予算を含む学習設計全体の比較|
|HBA vs 正規F|距離制約を強めた効果と、介入・速度・経路への波及|
|A vs C、B75 vs D75、B50 vs D50|同じ学習済みActorに対するruntime安全層の寄与|
|HBA vs HBA_OFF、F vs F_OFF|旧距離予算6と新1.5のActorに残る危険と安全層の寄与|
|Actor各方式 vs E/R|通常航法と安全層に対する学習Actorの追加効果|

全比較でN別・seed別、同じ形状の対応差、未割引/割引の距離負担・介入、所要時間、経路、通信、近接分布、QP解なし・emergency・条件不足を保存・解析する。予算達成とCritic校正は各方式の学習Costに合わせる。A/B75/B50のCriticはD+I、HBA/FのCriticはDを対象とし、全方式共通のD+Iも診断として併記する。OFF条件を元Criticの正式校正へ混ぜない。

## 実装と検証の要件

- Notebookから追加学習、全方式mean評価、stochastic校正、集計、exportを起動できるようにする。D50に独立学習を作らない。
- 旧実験を保持し、新しい方式の登録・予算・通信半径・モデル対応を明示する。旧Fと正規Fの混同、既存runへの設定上書きを防ぐ。
- シナリオhash、初期Actor対応、ON/OFFの同一checkpoint、FのCost定義、B50/D50の半径、既存checkpoint互換、評価途中再開、段階別exportをテストする。
- 学習則を維持したI/O軽量化を優先する。既存ログではonline完全checkpoint保存が大きな時間を占めていたため、巨大Replayの重複保存を避ける仕組みを検証する。再開後のReplay・optimizer・乱数・cursorの復元と更新結果の同値性を確かめる。
- 学習デバイス表示、段階ごとのtqdm、読込/保存時間の表示を維持する。追加学習・評価の本番起動前に小規模で保存・中断・再開・集計まで通す。

## 結果をMacへ持ち帰る単位

manifest、実行条件、ソースとcheckpointのhash、bank実体、方式名対応表、学習ログ、評価用小型モデル、生NPZ、ケース完了記録、集計をexport対象とする。完全再開用Replayとoptimizerを含む巨大checkpointはWindows側に保持する。

development、decision、校正を分け、方式/seed単位でも取り出せるようにする。従来のGit結果持ち帰りへ接続し、結果専用の保存先で旧結果と区別する。取り込み後には期待本数、未完了条件、初期条件とcheckpointの対応を検査する。生NPZを保持し、Mac側では再学習・再飛行なしに深掘り解析できるようにする。
