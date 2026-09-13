# Goal
- GoalID: e4f5a6b7-c8d9-4e0f-1a2b-3c4d5e6f7a8b
- Status: achieved
- Created: 2026-09-13T04:30:00+08:00
- Updated: 2026-09-13T05:30:00+08:00
## Objective
自主研究最佳方向：壓縮率突破 + 速度推進。以實測為準，找到最優 Pareto 前沿。
## Stopping condition
任何一項 bank 即 achieved：(a) 新比率冠軍 <0.8391（無分頁）；(b) 新速度冠軍 >34.5KB/s 且 bpb<=0.9389；(c) 新平衡點 score 現行最高。若全量 EV 用完無 bank 則 unmet。
## Must read first
- data/sota_loop.json（254 條）、tools/entropy_analysis.py（地板 1.60bpb）
- ensemble/bpe_ensemble_v13.py（chunk/sparse/gate 已解鎖）
- ARCHITECTURE.md（團隊分工）、.opencode/GOAL.md（歷史）
## Verification
每跑必收 bpb/Time/KB/s/Verified/PHASES/PEAK；ledger append；新圖更新。
## Progress log
- [2026-09-13] checkpoint：自主 goal 建立。方向決定：(1) Qwen-1.5B smoke（更高天花板）；(2) Qwen 0.5B 深掃（K/GATE/PF）；(3) SmolLM2 更大 block 探索。是否 blocked：否。
- [2026-09-13] ACHIEVED。Qwen-1.5B 下載 2944MB→smoke→0.7024→0.6996 SUB-0.7！220/220，PEAK 7.74GB 裝進 8GB。5 配置全 bank/null。§35 論文+README 雙語+ledger 259 條+entropy floor 1.60bpb。commit 1448e19 已 push。
