# Goal
- GoalID: a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d
- Status: pursuing
- Created: 2026-09-13T07:00:00+08:00
- Updated: 2026-09-13T07:00:00+08:00
## Objective
用 Groq Qwen3.8-27B 產生訓練資料 → 訓練專用小模型（又小又快又準）→ 測試 enwik8 壓縮率。
## Stopping condition
任何一項 bank 即 achieved：(a) 蒸餾後模型 bpb < 0.6617（贏現行 3B practical crown）；(b) 訓練完成且模型可用於壓縮；(c) 論文 §42 更新。若 blocked 則 unmet。
## Must read first
- tools/groq_bench.py（Groq API 連線方式）
- ensemble/bpe_ensemble_v13.py（壓縮引擎）
- data/sota_loop.json（291 條）
## Verification
訓練後模型在 enwik8 100KB 上跑 bpb + 220/220 verified。
## Progress log
- [2026-09-13] checkpoint：goal 建立。步驟：(1) Groq 產生 1MB 訓練資料；(2) tokenizer 對齊；(3) 訓練 SmolLM2-135M；(4) 測試。是否 blocked：否。
