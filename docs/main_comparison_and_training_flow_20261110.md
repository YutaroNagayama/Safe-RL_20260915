# Main比較構成と学習フロー

2026年11月10日、main設定・Notebook・学習／評価コードを照合。実験条件の変更や学習の開始は行っていない。

## 比較構成

| 方式 | 通信 | Actorの由来 | 学習Cost | 評価時HOCBF |
|---|---|---|---|---|
| A Full-Prior | 予測CPA 75m以内 | 独立学習 | 距離＋純介入 | ON |
| B Full-Reactive-75 | 現在距離75m以内 | 独立学習 | 距離＋純介入 | ON |
| C | Aと同じ | 同seed・同時点のA | 追加学習なし | OFF |
| D | Bと同じ | 同seed・同時点のB | 追加学習なし | OFF |
| E HOCBF-only | Actor通信なし | Referenceのみ | 学習なし | ON |
| F No Intervention Cost | Aと同じ | 独立学習 | 距離のみ | ON |
| R / F_OFF | なし / Fと同じ | Referenceのみ / Fと同一checkpoint | 追加学習なし | OFF |

独立学習はA/B/Fのみ、いずれもHOCBF ON。HOCBF監視距離は50mで、Bの通信75mとは独立。OFF評価でも物理制約は残す。A–Bは共有方式、A–Eは通信を使うActor追加、A–Fは介入Cost込みの学習設計、A–C／B–D／F–F_OFFは同一Actorでの安全層の寄与、C–Dは安全層を外した共有方式差を測る。A/Fの予算はともに6だが制約対象が違うため、共通の距離・純介入・実補正で比較する。

## 実行順

1. **bank固定。** 水平4面のどこからでも進入し、別面へ退出。面別の割り当てなし。進入点同士・退出点同士は30m以上。N=2/4/8/16。warmup100、training pool2,000、development100、decision120、confirmation_test400。予測候補を少なくとも1組含め、全方式で同じ形状を使用する。
2. **方式・seed別warm-up。** A/B/F×seed 0/1/2。各runで同じ100形状を、それぞれの通信・Cost・HOCBF条件で初期Actorを固定して収集する。Cost Criticのみn-step 10で5,000更新。Actor、Reward Critic、entropy、dualを更新しない。pre-mainの学習済みモデルを流用する工程ではない。
3. **online SAC。** Warm-upのReplay、開始状態buffer、モデル・optimizer・乱数を引き継ぎ、onlineカウンタを0から開始。training bankをNのround-robin順に供給する。学習とdualの開始閾値は2,000step。Cost target n10、batch256、episode dual・学習率0.001、Actor optimizer間隔1。時間報酬は全方式で−0.01/tick。初回予算は各30k、9run・計27万online step（episode終了超過とwarm-upは別）。
4. **途中保存と判断。** episodeログは逐次、完全再開用latestは25episodeごとと段階終了時。30k・60k到達後のepisode終端で小型モデルを保存し、各N12形状で平均行動development評価。既定TARGET=30000で全9 runを停止し、developmentと学習曲線で方式差のseed間の一貫性・終盤の変化を確認する。必要時のみ同じrunを全て累積60kへ延長する。
5. **decision比較。** 同一到達段階・共通120形状で、A/B/C/D/E/F/R/F_OFFを平均行動で評価。E/Rは各形状1回。A/B/Fの確率評価は各N10形状×3反復で、HOCBF ON・学習時のCost定義と初回sample actionを合わせる。C/D/F_OFFは正式校正には含めない。
6. **追加・確認。** N別・seed別・同一形状の対応差を確認する。通過、分離、経路長比、純介入、通信量に加え、無違反通過と到達時間の裾も見る。初回で判断できれば止め、不足に応じて予算延長、seed、評価形状、確率反復を追加する。設定とモデルを固定するまでconfirmation_testは開かない。確率Costの打切りを考慮した予算達成判定は別途必要。

[main Notebook](../notebooks/main/main_experiment.ipynb)・[設定](../notebooks/main/settings.yaml)・[実行説明](../notebooks/main/README.md)。現行NotebookはA/B/F×3 seedをカーネル内で順番に実行し、各runの学習とdevelopment評価を終えてから次のrunへ進む。共通bank準備は学習セル内で一度だけ行う。
