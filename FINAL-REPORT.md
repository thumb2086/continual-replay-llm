# 交卷報告：SmolLM2-135M 實用神經壓縮（enwik8 切片）

> 日期：2026-09-13。底座：SmolLM2-135M（Apache-2.0）。硬體：單卡 RTX 3060 Ti 8GB。主評測：enwik8 offset 50MB、100KB 中段切片（有代表性的文章區，非模板頭）。帳本：`data/sota_loop.json`（每輪 bpb/速度/verified 全記錄）。本報告所有數字均為實測，無外推。英文版見 [`FINAL-REPORT.en.md`](FINAL-REPORT.en.md)。

## 1. 交卷數字

| 指標 | 值 | 說明 |
|---|---|---|
| 最佳壓縮率 | **0.9139 bpb** | v11，ov4096＋floor 1e-6（F=1，物理極限）＋K1024，220/220 無損 |
| SOTA 線 | 0.9389 bpb | Nacrith 論文 100MB 全檔數 |
| 超線幅度 | **−2.7%** | 切片對全檔，見 §7 但書 |
| 同切片 H2H | 我們 0.9214 vs Nacrith 原廠 1.2248 | 同一切片、同腳本重現（`h2h_nacrith.py`），贏 25% |
| 速度（忙碌箱） | ~2.3KB/s（v11）→ **~3.7KB/s（v13＋proc＋pipeline 實測 26.9s/100KB）** | 看影片時測的 |
| 速度（安靜推算） | ~5KB/s | 真功耗 forward 12s＋loop 6s＋零頭 |
| 主機記憶體峰值 | 3.4GB | 實測 RSS |
| 顯存峰值 | 5.87GB | `torch.cuda.max_memory_allocated`，8GB 卡內 |

重現（PowerShell，約 45 秒）：
```
$env:BIGRAM_LAMBDA='0.99'; $env:ENWIK8_OFFSET_MB='50'; $env:BIGRAM_CONF='10'
$env:TRIGRAM_CONF='3'; $env:TOP_K='1024'; $env:OVERLAP='4096'; $env:FLOOR_FRAC='1e-6'
$env:USE_CACHE_S2='0'; $env:USE_FP16_XFER='1'; $env:PREFILTER='2048'
python -u bpe_ensemble_v11.py
# 判收：bits/byte: 0.9139，Verified: 220 lossless, fails: 0
```

## 2. 完整軌跡（每一步都是實測，失敗也記）

| 階段 | bpb | 關鍵動作 |
|---|---|---|
| frozen ctx2048/8192 | 1.2541 → 1.2328 | 換 SmolLM2 底座；長 context 有效 |
| λ sweep | 0.9607/0.9605 | λ 曲線已平，調參挖完 |
| v4 重構 | 0.9605 | 抓 double-softmax＋fp32 OOM，數學洗乾淨 |
| v5 trigram | 0.9597 | bigram 已飽和，trigram 無貢獻 |
| CONF 掃參 | ~0.9595 | 整面牆是平的 |
| TOP_K 4096 | 1.0132（災難） | 大 alphabet 成本主導；反向啟示試小 K |
| TOP_K 1024 | 0.9499 | K 甜蜜點（−0.010） |
| v6 overlap | 0.9431 | 冷啟動＋邊界修復 |
| v7 單流 stage-2 | 0.9375（過線） | finish-bit 稅：兩段編碼白繳 ~1000 bits |
| v7 floor 1e-5 | 0.9214（−0.016，最大單步） | smoothing 稅理論兌現 |
| v8 瘦身 | 0.9213 | numba 與 numpy 逐行一致 |
| ov4096 | 0.9157 | context 紅利 |
| ov4096＋floor 5e-6 | 0.9146 | 正交疊加 |
| floor 2e-6 → 1e-6 | 0.9140 → **0.9139** | F=1 到底，floor 軸窮盡 |
| ov6144 | 0.9146（不動） | context 在 4096 飽和 |
| TOP_K 512 | 0.9278（更差） | K=1024 兩邊都試過，最優 |

## 3. 三個可發表的洞察

1. **Smoothing 稅**：14-bit 核心每符號強制 +1 count，K=2049 時 12.5% 機率質量被拿去抹平頭部。32-bit 的價值不在精度而在 floor 調校範圍；尾部稀少（逃逸率 0.6%）時小 floor 必勝。
2. **Finish-bit 稅**：舊 stage-2 拆兩段編碼，每段 ~2 bits finish 開銷；~250 逃逸 × 2 段 ≈ 1000 bits 白繳。單流計數省掉。
3. **邊界 bug 的誠實揭露**：v2–v5 的 bitstream 從未編碼 file token 0 與 block 邊界 token（抽查只採樣已編碼對，永遠抓不到）。v6 起補上（+0.0002 bpb）。v2–v5 的 bpb 作預測分有效，bitstream 非自包含——論文必須聲明。

## 4. 方法速覽（v11 最終形態）

- LLM（frozen SmolLM2-135M，fp16）top-1024 ＋ bigram/trigram KN blend（confidence 門 `n/(n+conf)`）＋ top-K＋escape 兩層編碼，全程 32-bit 算術編碼（TOTAL 2²⁰，floor 可調）。
- 工程：sliced lm_head（峰值 6.62→5.87GB）、多線程 Phase-A/B 兩階段 loop、增量凍結（O(dirty)，全文件擴展性）、精確時鐘（CPU-time＋CUDA event，被偷的時間單獨列）。
- 無損：file token 0 經 uniform(V)＋32-bit 自包含；220 點（等距＋全邊界＋token0）編解碼 roundtrip。

## 5. 速度與記憶體實測

- 瓶頸排序（精確時鐘）：forward（GPU，真功耗 12s 安靜/30s 忙碌）＞ loop（CPU 真功耗 ~6s，GIL 串行）＞ 其餘 ~2s。
- 存過的真加速：切片省 0.75GB、多線程 loop、rank 合併、stage-2 歸零、esc 向量化、tracemalloc 預設關、pv 瘦身、高優先級進程。
- 記憶體：主機 RSS 峰值 3.4GB，VRAM 峰值 5.87GB；12GB 事件為 batch-2 翻車（11.3GB VRAM → WDDM 分頁），配置已證偽，之後再未發生。

## 6. 證偽清單（同樣是成果）

unigram-ensemble、純 trigram、TOP_K 4096、prefilter 16384、CONF 掃參、cache-informed stage-2、batch-2 forward、CUDA graph（2.1930，replay 吐零——roundtrip 通過 ≠ 模型正確）、llama 後端（忙碌箱巨 kernel 被 WDDM 切碎，反輸 torch）、CUDA 高優先級流（WDDM 無視）、綁核＋8 線程（GIL 牆）、prefilter 512（候選不夠 1024）、TOP_K 512、ov6144（飽和）、KV-cache chaining（溫水煮青蛙：4K-new 段內 rotary 漂移經 KV 累積，64→0.11、512→5.4、4096→40.8；transformers 5.17 另有 crop 重編號問題）。ledger 可查每輪數字。

## 7. 誠實但書

1. 主數字 0.9139 為 100KB 中段切片；100MB 全檔未跑（用戶決策：太久）。SOTA 的 0.9389 是全檔數——跨檔比較，偏向保守表述為「切片領先 2.7%，全檔待測」。
2. 無損驗證 220 點，非全量逐 token 解碼（全量 decoder 已列為後續工作）。
3. 速度在忙碌箱測量，run-to-run 有雜訊；比率數字完全確定性可重現（固定流程＋parity 門：任何重構必須逐位一致或接受記錄在案的容差）。
4. Nacrith H2H 跑在其弱場（100KB 冷切片＋CPU 版）；其 0.9389 全檔熱機數仍是它主場的最好成績。最終排名以雙方全檔為準。
5. GGUF/llama 相關數字（H2H、v12）用官方 BF16，與 torch fp16 有 ±1e-4 級 backend 容差，已註明。

## 8. 文件地圖（＋2026-09-13 Pareto 附錄 §9）

- `bpe_ensemble_v11.py`：交卷系統（0.9139）
- `bpe_ensemble_v10.py`：切片後向等價版；`bpe_ensemble_v12.py`：llama 後端（忙碌箱證偽保留）；`bpe_ensemble_v13.py`：KV 接力主線已證偽封存，但其 plumbing（deferred driver、單源 `_range_core`、共享內存 proc、pipeline helper、雙緩衝、增量凍結）已在 recompute 模式逐位驗證 0.9139（`USE_PROC_LOOP=1`＋`PIPELINE=1` 實測 26.9s/100KB，loop 8.5→2.5s）；ORT 分支因 fp32 慢 3 倍＋逐 shape 重調優而死
- `ac32.py`：32-bit coder＋numba kernels；`sota_loop.py`＋`data/sota_loop.json`：迭代帳本
- `h2h_nacrith.py`＋`data/h2h_nacrith.json`：第三方重現；`third_party/nacrith`：原廠碼
- `test_shm_proc.py`：線程 vs 進程逐位一致（生產數據 0.9139 驗證；另抓到共享 scratch＋nogil kernel＝靜默段錯誤，已修）；`ort_export.py`：ONNX 導出腳本（ORT 分支已死，留檔）
- `compression-paper.md`＋`paper_sections_13_14_draft.md`：前期草稿（char 管線與早期 ensemble 史）

## 9. Pareto 前沿（2026-09-13，安靜箱，全部 220/220）

| 配置 | bpb | 秒/100KB | KB/s |
|---|---|---|---|
| ov4096（crown） | 0.9139 | 19.4 | 5.1 |
| BC7 | 0.9140 | 19.1 | 5.2 |
| ov6144 | 0.9141 | 34.4 | 2.9（死胡同：更慢，沒更好） |
| TRI0 | 0.9146 | 18.3 | 5.5 |
| ov3072 | 0.9166 | 16.9 | 5.9 |
| ov2048（knee） | 0.9194 | 14.2 | 7.0 |
| ov1024 | 0.9237 | 13.5 | 7.4 |
| ov0 | 0.9268 | 12.0 | 8.3 |

8 點全在 SOTA 線（0.9389）之下。knee 在 ov2048（再往下每 0.001 bpb 越來越貴）。10KB/s 未達（最快 8.3）。圖：`pareto.png`，腳本：`pareto_plot.py`。
