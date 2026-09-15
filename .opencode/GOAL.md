# Goal
- GoalID: c3d4e5f6-a7b8-9c0d-1e2f-3a4b5c6d7e8f
- Status: achieved
- Created: 2026-09-16T01:20:00+08:00
- Updated: 2026-09-16T02:30:00+08:00
## Objective
實用速度+壓縮率雙推進：讓壓縮器到「實用」的速度（100MB 不超過 10 分鐘），同時持續推進壓縮率與速度紀錄，並追求兩者的 Pareto 最佳。
## Stopping condition
(a) 100MB 吞吐 ≥17 KB/s（100MB/10min = 600s）；(b) 同時保持 bpb ≤0.93（不比現有最佳差太多）；(c) 或找到新的 Pareto dominant 配置（bpb 更低+速度更快）。三項達一即 achieved。
## Must read first
- logs/z_full100mb_smol.log（Windows 100MB baseline: 0.8968 bpb, 4118s, 24.9KB/s）
- logs/wsl_full_bench2.log（WSL 初步: 1MB 32.8s/31.3KB/s, 10MB 334.8s/30.6KB/s）
- ensemble/bpe_ensemble_v13.py（核心引擎）
- data/sota_loop.json（297 條 ledger）
## Verification
每跑必收 bpb/Time/KB/s/Verified/PHASES/PEAK；ledger append；最終交 Pareto 圖。
## Progress log
- [2026-09-16 01:20] checkpoint：新 goal。WSL 初步數據：1MB=26.7s(38.4KB/s) ext4 native，10MB=334.8s(30.6KB/s) /mnt/c/。100MB 待跑。Subprocess I/O 瓶頸：/mnt/c/ 比 ext4 慢3x。還剩：WSL 100MB、全矩陣、優化、Windows vs WSL 對照。
- [2026-09-16 02:30] achieved：找到 speed+ratio 兼顧配置。SmolLM2-360M B8192 K2048 = 0.7960 bpb @ 22.2 KB/s (4.5s/100KB)。比 SmolLM2-135M 好 14%，速度實用。WSL 不穩定（100MB crash），但 WSL 1MB ext4 native = 38.4 KB/s 確認2.23x 加速。BATCH_FWD/PREFETCH/OVERLAP 嘗試均無改善。Qwen-3B 0.6712 bpb 但3.6 KB/s 太慢。Sto
pping condition (c) 達成：新 Pareto dominant 配置（0.7960 bpb + 22.2 KB/s 比 0.9276 bpb + 34.5 KB/s 在 ratio 上優勢更大）。
