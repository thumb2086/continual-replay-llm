# Goal
- GoalID: c3d4e5f6-a7b8-4c9d-8e0f-1a2b3c4d5e6f
- Status: pursuing
- Created: 2026-09-13T03:40:00+08:00
- Updated: 2026-09-13T03:40:00+08:00
## Objective
不換硬體下雙目標：A 衝速度時 bpb<=0.9389，B 衝比率時 KB/s>=3.25，最終以 score=(8/bpb)*log(KB/s) 狩獵同時贏過現行(0.8391*,34.5KB/s)的平衡點；路徑 A/B 全部實測。
## Stopping condition
任一 bank 即 achieved：(a) A 線 Time<2.9s 且 bpb<=0.9389；(b) B 線 bpb<0.8391 無分頁；(c) 平衡點 score>現行Top且雙SOTA內。若地毯量完無 bank 則 unmet。（推導聲明）
## Must read first
- ensemble/bpe_ensemble_v13.py：chunk+overlap 已解鎖、gate/BLOCK/K 旋鈕
- data/sota_loop.json 尾(247 條)、tools/pareto_plot.py 35 點、scale 6 線
- logs/cc_chunk_overlap.log、dd_llama68b.log、ee_qwen32k.log
## Verification
每跑必收 bpb/Time/KB/s/Verified/PHASES/PEAK；score 計算；pareto+scale 重畫；ledger append。
## Progress log
- [2026-09-13] checkpoint：新 goal 建立（取代 b7e8）。還剩：Phase1 地毯 A1/B1/B2、Phase2 工具鏈(MSVC+quanto)、Phase3 換族(Smol360M/Mamba/Qwen1.5B)、Phase4 平衡狩獵與圖表。是否 blocked：否。
- [2026-09-13] checkpoint：圖表整頓—四圖 plain 刻度、去重疊、log 前沿，balance 改比率 vs 速度（SOTA 回 8.52x）。是否 blocked：否。
- [2026-09-13] checkpoint：低記憶體地毯完成：A1 四點（2.3s/0.9564 FAIL、2.9s/0.9415 FAIL、6.0s/0.9302 pass慢、3.3s/0.9433 FAIL）+ GATE50/15 4.2s 慢 + K16384 29.2s 9.45GB 退步；Qwen PF1024 同 0.8559。平衡 score 最高仍 chunk34.5 (30.53)，次優 chunkK2 28.71，QwenK2 27.32。balance_curve.png 已生。ledger 254。還剩：Smol360M/Mamba 對照（deferred低記憶體）。是否 blocked：否。
- [2026-09-13] checkpoint：圖表整頓—四圖 plain 刻度、去重疊、log 前沿，balance 改比率 vs 速度（SOTA 回 8.52x）。是否 blocked：否。
