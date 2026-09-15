# Goal
- GoalID: f7a8b9c0-d1e2-4f3a-4b5c-6d7e8f9a0b1c
- Status: unmet
- Created: 2026-09-14T14:00:00+08:00
- Updated: 2026-09-14T14:00:00+08:00
## Objective
速度推升：突破現行 34.5KB/s 速度天花板。嘗試 ONNX Runtime、更大 block、batch parallel、torch.compile 進階設定。
## Stopping condition
(a) 100KB 速度 >50KB/s（SmolLM2）；(b) 100MB 吞吐 >40KB/s。若 EV 量完無 bank 則 unmet。
## Must read first
- logs/z_full100mb_smol.log（PHASES: fwd=2079.6s loop=1577.5s frz=437.5s code=69.6s）
- ensemble/bpe_ensemble_v13.py（核心）
- data/sota_loop.json（295 條）
## Verification
每跑必收 bpb/Time/KB/s/Verified/PHASES/PEAK；ledger append。
## Progress log
- [2026-09-14] checkpoint：新 goal。PHASES 分析：fwd 50.5% + loop 38.3% = 88.8%。方向：ONNX Runtime（fwd 加速）、batch parallel（loop 隱藏）、更大 block（更少 forward）。是否 blocked：否。
