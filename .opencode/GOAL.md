# Goal
- GoalID: f1a2b3c4-d5e6-4f7a-8b9c-0d1e2f3a4b5c
- Status: pursuing
- Created: 2026-09-13T06:00:00+08:00
- Updated: 2026-09-13T06:00:00+08:00
## Objective
蒸餾研究 + 新研究方案：(1) 用 Groq Qwen3.8-27B 蒸餾小模型；(2) 探索新壓縮演算法；(3) 持續推進壓縮率與速度；(4) 完成論文數據更新。
## Stopping condition
任何一項 bank 即 achieved：(a) 蒸餾後本地模型 bpb < 0.6450；(b) 新演算法突破現行 Pareto；(c) 論文完整更新至 §41+。若 EV 用完無 bank 則 unmet。
## Must read first
- data/sota_loop.json（282 條）、tools/groq_bench.py（Groq 測試）
- ensemble/bpe_ensemble_v13.py（chunk/sparse/gate 已解鎖）
- FINAL-REPORT.md §1-40（現行全貌）
## Verification
每跑必收 bpb/Time/KB/s/Verified/PEAK；ledger append；論文更新。
## Progress log
- [2026-09-13] checkpoint：goal 建立。方向：蒸餾（Groq 27B → 本地小模型）、新演算法探索、論文更新。是否 blocked：否。
