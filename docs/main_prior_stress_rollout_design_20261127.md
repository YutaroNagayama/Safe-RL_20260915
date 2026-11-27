# 予測共有の追加ロールアウト：方針更新

2026-11-27。実施対象はユーザー指定どおり**A・B75・B50のみ**。
学習済みモデルを固定し、同じ低CPAの交差航路に0/20/40秒の進入側延長を加えて、情報猶予・Actor応答・HOCBF介入/可解性・効率を比較する。
追加学習、E/R、HBA/F、通信開始を操作した別のA条件は実施しない。

Windowsの実施仕様・Notebook・途中再開・結果zipは[航路延長ロールアウト実行メモ](main_route_extension_windows_20261127.md)にまとめた。

旧案に先立つ大空域一様生成1,200形状の検査は、`results/main_additional_20261120/analysis/prior_stress_design_20261127/generation_screen.json`に保持する。これは生成診断であり、A/Bの評価結果ではない。低CPA頻度・50秒予測の範囲・初速予測とReference加速の違いを踏まえ、同じ交差形状の航路を延ばす設計へ整理した。
