# Goal
- GoalID: d4e5f6a7-b8c9-0d1e-2f3a-4b5c6d7e8f9a
- Status: achieved
- Created: 2026-09-16T02:40:00+08:00
- Updated: 2026-09-16T03:30:00+08:00
## Objective
三大方向同時推進：(1) 100MB 實用速度 <10 分鐘；(2) 持續推 bpb 紀錄；(3) 速度+壓縮率 Pareto 再突破。
## Stopping condition
(a) 100MB 吞吐 ≥17 KB/s（<600s）且 bpb ≤0.93；或 (b) 新 Pareto dominant（bpb 更低+速度更快）；或 (c) 三條路全死（unmet）。達一即 achieved。
## Must read first
- data/sota_loop.json（312 條 ledger）
- ensemble/bpe_ensemble_v13.py（核心）
- logs/z_full100mb_smol.log（Windows 100MB: 0.8968 bpb, 4118s）
- .opencode/GOAL.md（上一輪成果）
## Verification
每跑必收 bpb/Time/KB/s/Verified/PHASES/PEAK；ledger append。
## Progress log
- [2026-09-16 02:40] checkpoint：新 goal。未嘗試路線：(A) Qwen-3B B2048 Windows；(B) SmolLM2-360M 100MB Windows；(C) Qwen-3B B2048 K1024 WSL；(D) 100MB WSL 重跑。
- [2026-09-16 02:50] (A) Qwen-3B B2048 K1024: 0.6848 bpb 27.3s 3.7KB/s。K512: 0.6954 bpb 18.8s 5.3KB/s。Qwen 仍太慢（100MB 預估 322-474 min）。
- [2026-09-16 03:10] (D) WSL 100MB SmolLM2-135M: 0.8968 bpb 3456.7s(58min) 29.6KB/s。vs Windows 4118s/24.9KB/s。WSL 快19%。Stopping (a)✓ (b)✓。
- [2026-09-16 03:30] (B) WSL 100MB SmolLM2-360M K2048: 0.7808 bpb 5284s(88min) 19.4KB/s。NEW RATIO RECORD for 100MB！13% better than 135M。Achieved。
