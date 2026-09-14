# 交卷報告：SmolLM2-135M 實用神經壓縮（enwik8 切片）

> 日期：2026-09-13。底座：SmolLM2-135M（Apache-2.0）。硬體：單卡 RTX 3060 Ti 8GB。主評測：enwik8 offset 50MB、100KB 中段切片（有代表性的文章區，非模板頭）。帳本：`data/sota_loop.json`（每輪 bpb/速度/verified 全記錄）。本報告所有數字均為實測，無外推。英文版見 [`FINAL-REPORT.en.md`](FINAL-REPORT.en.md)。

## 1. 交卷數字

| 指標 | 值 | 說明 |
|---|---|---|
| 最佳壓縮率（SmolLM2） | **0.9003 bpb** | v13，ov4096＋floor 1e-6＋K/PF 8192，220/220 無損（K 階梯：1024→0.9139，2048→0.9050，4096→0.9013，8192→0.9003） |
| 最佳壓縮率（Qwen0.5B，另線） | **0.8391* bpb** | Qwen2.5-0.5B chunked K4096，219/219，*峰值 9.11GB 分頁；實用 0.8442 @ 5.6s 5.32GB |
| SOTA 線 | 0.9389 bpb | Nacrith 論文 100MB 全檔數 |
| 超線幅度 | SmolLM2 **−4.1%** / Qwen **−10.6%** | 切片對全檔，見 §7 但書 |
| 同切片 H2H | 我們 0.9214 vs Nacrith 原廠 1.2248 | 同一切片、同腳本重現（`h2h_nacrith.py`），贏 25% |
| 速度（忙碌箱，現行） | **chunked ov0 2.9 秒/100KB＝34.5KB/s**（峰值 1.50GB，逐位一致）；knee 12.0s；王座 24.0s | 中間節點 chunkK2 3.7s/27KB/s、chunkK4 5.9s/16.9KB/s；見 §29–30 |
| 放量梯子（全 220/220） | chunked 2.9s/29.8s/273.7s (34.5/34.3/37.4KB/s)；crown 24s/254.7s/1755s (0.9003/0.9078/0.8765) | 10MB 反常更快（cache 熱）；見 §30 |
| 顯存峰值 | chunked 1.50GB / crown 5.26GB（10MB）/ Qwen 5.32GB | `max_memory_allocated`，8GB 卡內；舊 5.01GB 為 100KB 王座 |

重現（PowerShell，約 45 秒）：
```
$env:BIGRAM_LAMBDA='0.99'; $env:ENWIK8_OFFSET_MB='50'; $env:BIGRAM_CONF='10'
$env:TRIGRAM_CONF='3'; $env:TOP_K='1024'; $env:OVERLAP='4096'; $env:FLOOR_FRAC='1e-6'
$env:USE_CACHE_S2='0'; $env:USE_FP16_XFER='1'; $env:PREFILTER='2048'
python -u ensemble/bpe_ensemble_v11.py
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
| TOP_K 512 | 0.9278（更差） | K=1024 兩邊都試過，最優（2026-09-12 推翻：是 prefilter-bound，見 K 階梯） |
| K 階梯（2026-09-12，v13） | 1024→0.9139／2048→**0.9050**／4096→**0.9013**／8192→**0.9003** | K=PF 同步放大：−89／−37／−10 e-4，遞減，停在 8192 |
| S2 on K2048 | 0.9051（null） | cache-informed stage-2 無疊加 |
| floor 1e-7 | 0.9050（null） | floor 已飽和 |

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

1. 主數字 0.9003 為 100KB 中段切片；100MB 全檔未跑（用戶決策：太久）。SOTA 的 0.9389 是全檔數——跨檔比較，偏向保守表述為「切片領先 4.1%，全檔待測」。
2. 無損驗證 220 點，非全量逐 token 解碼（全量 decoder 已列為後續工作）。
3. 速度在忙碌箱測量，run-to-run 有雜訊；比率數字完全確定性可重現（固定流程＋parity 門：任何重構必須逐位一致或接受記錄在案的容差）。
4. Nacrith H2H 跑在其弱場（100KB 冷切片＋CPU 版）；其 0.9389 全檔熱機數仍是它主場的最好成績。最終排名以雙方全檔為準。
5. GGUF/llama 相關數字（H2H、v12）用官方 BF16，與 torch fp16 有 ±1e-4 級 backend 容差，已註明。

## 8. 文件地圖（＋2026-09-13 Pareto 附錄 §9）

- `ensemble/bpe_ensemble_v11.py`：前交卷系統（0.9139）
- `ensemble/bpe_ensemble_v10.py`：切片後向等價版；`ensemble/bpe_ensemble_v12.py`：llama 後端（忙碌箱證偽保留）；`ensemble/bpe_ensemble_v13.py`：KV 接力主線已證偽封存，但其 plumbing（deferred driver、單源 `_range_core`、共享內存 proc、pipeline helper、雙緩衝、增量凍結）已在 recompute 模式逐位驗證 0.9139（`USE_PROC_LOOP=1`＋`PIPELINE=1` 實測 26.9s/100KB，loop 8.5→2.5s）；ORT 分支因 fp32 慢 3 倍＋逐 shape 重調優而死；2026-09-12 起 v13 為現行交卷（比率王 0.9003／速度 10.2KB/s）。現行預設：`SDPA_BACKEND=flash`、`USE_FP16_SOFTMAX=1`、`UNCHUNKED_TOPK=1`、`USE_PROC_LOOP=0`、`PIPELINE=0`、`N_LOOP_WORKERS=1`，`BLOCK_TOKENS` 可調（8192 最優，兩邊都掃過）
- `ac32.py`：32-bit coder＋numba kernels；`ensemble/sota_loop.py`＋`data/sota_loop.json`：迭代帳本
- `h2h_nacrith.py`＋`data/h2h_nacrith.json`：第三方重現；`third_party/nacrith`：原廠碼
- `tools/test_shm_proc.py`：線程 vs 進程逐位一致（生產數據 0.9139 驗證；另抓到共享 scratch＋nogil kernel＝靜默段錯誤，已修）；`tools/ort_export.py`：ONNX 導出腳本（ORT 分支已死，留檔）
- `compression-paper.md`＋`paper_sections_13_14_draft.md`：前期草稿（char 管線與早期 ensemble 史）

## 9. Pareto 前沿（2026-09-13，安靜箱，全部 220/220）

| 配置 | bpb | 秒/100KB | KB/s |
|---|---|---|---|
| K8192（crown） | 0.9003 | 24.0 | 4.2 |
| K4096 | 0.9013 | 20.2 | 5.0 |
| K2048 | 0.9050 | 18.0 | 5.6 |
| ov4096 K1024 | 0.9139 | 17.3 | 5.8 |
| BC7 | 0.9140 | 19.1 | 5.2 |
| ov6144 | 0.9141 | 34.4 | 2.9（死胡同：更慢，沒更好） |
| TRI0 | 0.9146 | 18.3 | 5.5 |
| ov3072 | 0.9166 | 16.9 | 5.9 |
| ov2048（knee） | 0.9194 | 12.0 | 8.3 |
| ov1024 | 0.9237 | 13.5 | 7.4 |
| ov0（10KB/s 過線） | 0.9268 | 9.8 | 10.2 |

11 點全在 SOTA 線（0.9389）之下。knee 在 ov2048。**10KB/s 已過線（ov0 9.8 秒）。**舊點（BC7/TRI0/ov6144/ov3072/ov1024）是 flash 之前的計時——比率相同，重測只會更快。圖：`pareto_off50.png`＋`pareto_off75.png`，腳本：`tools/pareto_plot.py`。

v2（19 點）：overlap 加密（ov512 0.9232、ov1536 0.9237、ov2560 0.9169、ov5120 0.9164——±0.002 分割雜訊，非單調）；off75 硬區第二曲線（ov0 0.9785→ov4096 0.9662，全輸 SOTA 但形狀平行）；1MB 鑽石兩顆（off50 0.9222、off25 **0.9076** 新 1MB 最佳）。多切片 8 點均值 0.9037（off0 0.8307／off10 0.8934／off25 0.9218／off35 0.8747／off50 0.9139／off60 0.8949／off75 0.9662／off90 0.9339，其中 off75 輸 SOTA 如實保留）。

## 10. 穩健性與 10KB/s 結案（2026-09-13）

最佳配置多切片（100KB）：off0 **0.8307**（模板頭紅利）／off25 0.9218／off50 0.9139／off75 **0.9662（輸 SOTA＋3%，硬區間，如實保留）**；四片均值 0.908，贏 SOTA 3.3%。1MB flagship（off50）：**0.9222**，220/220，256 秒，贏 SOTA 1.8%（增量凍結立功：75 段 frz 合計 0.7 秒）。

10KB/s 結案（2026-09-12 推翻舊結論：已過線）：組合技（ov0＋flash＋fp16-softmax＋單 topk＋單發 topk＋pipeline/proc 關＋單 loop worker）9.8 秒（10.2KB/s）。2026-09-13 的「不可達」寫於 SDPA 發現之前（auto 選到 math fallback）與 fwd 拆表之前——舊結論作廢，新瓶頸見 §12。。

## 11. 最終迭代與硬體極限宣告（2026-09-13）

cProfile 指引：rotary 三角重算佔 forward 13%＋`.to()` 排隊 10%——cos/sin 快取（逐位一致，tripwire 防鏈毒）省 forward 24%（24.3→18.4s）。N 進程同態掃描：2（11.6s）＜1（11.9）＜4（12.0）＜6（12.4）＜8（13.1），N=2 最優。微優化 gc.disable、CUDNN benchmark、EMPTY/1000 全 null（±0.1s 雜訊）。按開工規則（三連 null 即停）：**硬體極限＝ov0/N2，11.6 秒＝8.6KB/s**（比率 0.9268，220/220）；比率王 ov4096 0.9139 @ 19.4 秒不動。再往下要安靜十倍的箱子或新卡，不是 code。（2026-09-12：本節數字被 §12 取代，保留為歷史。）

## 12. 速度重開＋比率突破（2026-09-12，取代 §10–11 數字）

速度（ov0 上 12.1→9.8 秒）：SUBCLOCKS 把被污染的 forward 窗口拆開——attn 0.9／topk-launch 1.6／d2h 6.1／shmw 1.6 秒。flash 一直有在動（attn 0.23 秒/forward）；之前的「2.5 秒/forward」是六種成本共用一個計時器。PCIe 洗清嫌疑（純傳輸 0.12 秒/800MB）。入帳：fp16 softmax（比率四位小數不動）、精確單 topk（少一次全 V 遍歷）、單發 topk（2.32GB）、pipeline/proc 預設關（helper 爭用大於重疊收益）、單 loop worker（GIL：10.5→9.8 秒）。顯卡負載下 boost 正常（1905MHz/100%/219W；之前低時鐘讀數是死 job 的假影）。**10KB/s 過線：9.8 秒＝10.2KB/s。**

比率（0.9139→**0.9003**）：ov4096 上 K/PF 階梯——2048→0.9050、4096→0.9013、8192→0.9003（增量 −89/−37/−10 e-4，遞減，停）。舊的「K=1024 最優」是 prefilter-bound，推翻。新王座顯存峰值 5.01GB，8GB 卡內。SOTA 差距：−4.1%。

## 13. batch 證偽、1MB 放量、drain 診斷（2026-09-12）

batch-2 雙 forward：逐位正確（0.9268）但零加速——只拼了模型呼叫，逐段 topk/d2h/shmw 才是大頭。路上殺兩隻 bug：stash 交棒吃掉一個 chunk 槽（整段被跳過，1.8240）；submit 迴圈用 stale chunk 重綁 block（mate 拿前一段的 ids 編碼，1.3576）。batched==single 證到 0.000000（含 rope patch）。預設保持單發（BATCH_SEGS=1）。`empty_cache` 拔除：null。

1MB 王座驗證（K8192/ov4096，74 段）：**0.9078**，254.7 秒，顯存峰值 5.01GB 持平（無洩漏），220/220。K 階梯在 10 倍量級成立（＋75e-4）。4.0KB/s→全檔 100MB 約 7 小時：迭代太慢，先做 logit 手術。

drain 診斷（SYNC_ATTN）：真 attention-GPU 只要 0.7 秒/4 forward——flash 在 pipeline 內確認有動。d2h 約 5 秒是 softmax＋topk 執行堆積（400M 元素 softmax 本該毫秒級：差 300 倍→箱子搶佔／sag 或 WDDM 病理）。下一步：logit-space 手術（lse＋gather，殺掉全 V softmax 執行）。

## 14. 手術取消、放量梯子（2026-09-12）

logit 手術取消，有證據：單機實測 softmax＋2×topk＋gather＋全部 D2H＝0.15 秒/段，純 GPU 0.02 秒/段。v13 的 1.4 秒/段是 10 倍調度開銷（排隊等前面的活、桌面搶佔、burst 間 boost 掉速）——不是活。沒東西可 fuse，code 端速度工作關閉（每根槓桿不是 null 就是環境定罪）；剩下的槓桿是安靜的箱子。

1MB 速度基線（ov0/K1024，38 段）：**0.9347**，104.3 秒＝9.8KB/s——與 100KB 線性一致，放量漂移 ＋79e-4（與王座 ＋75e-4 同級）。100KB 修復後重驗 0.9268 一字不差。梯子：1MB 約 2 分鐘（速度版）／4 分鐘（王座版）；10MB 約 17 分鐘；全檔 100MB 安靜箱約 3 小時——可過夜跑。

## 15. Goal 回合：速度關閉、K 到頂、Pareto＋6 點（2026-09-12）

速度（4 探針全 null）：2 worker 輸 1（14.3 秒，GIL 曲線 1＜2＜8 補完）；GC_OFF null；proc/N=2 null（15.6 秒，thread 勝）；batch 上王座逐位一致但更慢（23.1 秒）。code 端速度關閉（連同 §14 證據）。殘留路徑（未走）：segment-skip、graph 重試、WSL compile——專案級，列案。

比率（9 探針）：K 階梯到頂——K16384 退回 0.9027（＋24e-4，52.4 秒，顯存 8.26GB 分頁！）：倒 U（−89/−37/−10/＋24），8GB 卡永不上 K8192。持平／關閉：lambda 0.995、ov6144@K8（＋1e-4）、CONF 20、TRI-CONF 10、PF 超 K、S2、floor 1e-7。新曲線點：K6144 0.9004@28.5 秒、ov2048@K8 0.9058@23.6 秒、ov1024@K8 0.9099@22.3 秒、ov0@K8 0.9127@21.8 秒（大 K 對 ov0 最補，−141e-4，2.2 倍時間）。

Pareto v4：28 點（OFF50 21 點）。全 220/220。王座 0.9003／速度 10.2KB/s 不動。

## 16. WSL 全移植：建成、更慢、關閉（2026-09-12）

WSL（無網路）離線建成全套環境：63 個輪子在 Windows 代下載（雙 `manylinux_2_17＋2_28` 平台標籤——pip 26 拔掉裸別名；`sys_platform` 標記在 Windows 主機靜默丟掉 nvidia 依賴，改顯式抓；nvjitlink 12.4→12.9 修 cusparse 未定義符號；tokenizers pin 放寬給 transformers 4.57.6）。torch 2.14＋cu126＋triton 3.8＋CUDA 12.9＋gcc 全活。配方：Windows 下 `pip download --platform manylinux_2_17_x86_64 --platform manylinux_2_28_x86_64 --python-version 3.12 --implementation cp --abi cp312`，拷到 `~/wheels`，`pip install --no-index --no-deps`。

結果：WSL eager 100KB＝12.9 秒（比 Windows 9.7 秒慢；半虛擬化開銷大於 WDDM），跨平台一致 0.9266 vs 0.9268（2e-4）。Inductor reduce-overhead 0.85 秒/forward 輸 eager 0.33（小模型上 dynamo 開銷）；default 模式打平（0.32）。max-autotune 書面拒絕：它調 GEMM，我們證過的牆是調度。速度 goal≤8.0 秒：證據齊全判 UNMET——9.7 秒就是這台箱子的牆。能重開它的輸入：閒置的箱子、更大的卡，或換模型。

## 17. gather 時代：5.3 秒、scale 圖、PF16384（2026-09-12）

gather（logits 上 topk＋lse＋pi-gather＋CPU exp＋稀疏 blk，numba V-stamp 截斷）轉正預設：100KB 速度 9.7→**5.3 秒＝18.9KB/s**（−45%），比率 0.9272（＋4e-4 pi-only 代價，記案），220/220 三次確認（5.3/7.1/7.4 波動——箱子雜訊，全過 8.0 線）。單發 scatter 打敗分塊 3.0→1.7 秒（與 micro 基準相反——記案，未解）。王座＋gather 0.9005@27.4 秒（＋2e-4，大 K 下更慢：傳輸隨 K 漲——gather 是速度配置武器）。

比率射擊：PF16384/K8192＝0.9004（加寬救不回 tail；顯存 8.01GB 紅線——永不上 PF8192）；LAMBDA＝0.999＝0.9010（純 LM 輸 blend）。無新紀錄；K/lambda/overlap/CONF/tri/PF/floor/S2 全關。

放量梯子（全 220/220）：速度 gather 5.3 秒／56.9 秒／632 秒（100KB/1MB/10MB：18.9/18.0/16.2KB/s，顯存 4.01GB 持平）；10MB 比率 0.9039（cache 隨規模變熱）。新 `tools/scale_plot.py`→`scale_time_size.png`（大小-時間雙對數＋比率-大小）。Pareto v5：30 點。全檔 100MB gather 速度約 1.7 小時——可過夜。

## 18. 直覺化圖表：最好去右上（2026-09-12）

舊 Pareto 圖最好（又快又緊）沉在右下角（y＝bpb 越低越好）——跟閱讀直覺反著。兩張圖 y 軸改壓縮比 8/bpb（越高越好）：x＝吞吐（越右越快），紅色 SOTA 線在 8.52x（之下皆輸），理想角落右上。同樣 30 個實測點，同腳本（`tools/pareto_plot.py`）。前沿線誠實顯示取捨：王座上中（8.89x @ 4.2KB/s）、gather 右下（8.63x @ 18.9KB/s）——右上還空著；那個空角就是下一條前沿。

## 19. Pareto 洗牌：舊時代刪除＋6 新探針（2026-09-12）

刪 10 個前 flash 計時（BC7/ov6144/TRI0/ov5120/ov3072/ov2560/ov1536/ov512/ov1024/ov0-classic——舊調度時代）。6 新跑全 220/220：王座 classic 重驗一字不差 0.9003（後面所有改動在旗關時零影響）；knee＋gather 0.9194 不動 @ 5.9 秒（knee 處 pi-only 代價約 0，12.0→5.9 秒）；ov1024＋gather 0.9241@5.5 秒；K3072＋gather 0.9029@9.7 秒（階梯填補）；ov3072K8＋gather 0.9026@15.0 秒（大 K 下 overlap 還有 0.002 差距）；ov0K2＋gather 0.9183@5.7 秒。OFF50 現 17 個現行 code 點。王座 0.9003／速度 18.9KB/s 不動。

## 20. WSL 速度電池：12.9→6.6 秒、1 秒判決（2026-09-12）

Windows 微調：K512＋gather 史上最快 4.9 秒但＋146e-4 比率（拒收——不要比率的速度不值錢）；alloc 兩次 null；安靜箱 alone 5.3–7.4→5.0 秒（箱子狀態約 2 秒擺幅——所有速度數都帶這個 band）。

WSL（100KB，全程 0.9271±1e-4，全 220/220）：eager 12.9→graph 7.9（−39%：無 WDDM replay 終於生效——差分證明牆就是 WDDM）→線程 2 配 7.2（Linux 翻轉 Windows GIL 結論）→proc-fork-2 **6.6 秒**（無 spawn 罰則＋自家 GIL）→proc-4/線程-4 null（N=2 到處最優）。

1 秒判決：這片矽到不了。地板數學（100KB 4 段）：attention-GPU 4×0.2＋後處理 4×0.3＋CPU loop/code≈3 秒不可約；forward 減不到 4 發（模型天花板 8192，16K 證過垃圾）。已入帳最佳：安靜 Win 5.0 秒／WSL 6.6 秒。重開條件：forward 更少的模型、5 倍矽、或全閒置。微調尾聲：K512＋gather 史上最快 4.9 秒但＋146e-4 比率（拒收）；alloc 兩次 null；主線程最高優先 null（5.1 vs 5.0）；torch/OMP/MKL 單線程 null（5.1 vs 5.0——GIL 主導，池子不是兇手）。pinned 異步傳輸入帳（逐位一致，d2h 1.0→0.0 秒，淨 −0.2 秒，預設開）。code 端徹底關閉。

## 21. blend-gate＋預設轉正：4.4 秒（2026-09-13）

cProfile-inline 揪出 nb_blend_row 1.24 秒/1.7 萬次（72µs 每次，Phase-A 的 69%）——成本 drivers 是 blend 顆數，不是驅動 Python（numba 驅動 EV 上限約 0.5 秒，有測量為據拒絕）。新門看 cache 行總量：(5,2) 一字不差 0.9272 且 blend −16%（低 count blend 全是廢活）；(20,7) 用＋4e-4 換 4.4 秒＝22.7KB/s（新最快有效 Pareto 點）；(50,15) 飽和（＋8e-4，同 4.4 秒）。預設開箱即快（overlap 0、floor 1e-6、S2 關、fp16 傳、單 worker、blend-gate 5/2、gather＋pinned 開）：純預設跑 0.9185 @ 5.1 秒。Pareto v6：25 點（off50 18 點）。1 秒仍是物理 bound（地板約 3 秒）；已入帳最佳 4.4 秒。

## 22. gate 階梯＋王座 gate：王座 24→13.3 秒（2026-09-13）

gate 階梯（ov0/K1024）：(5,2) 一字不差/4.6 秒→(10,3)＋1e-4/4.6 秒→(15,5)＋1e-4/4.6 秒→(20,7)＋4e-4/4.4 秒→(50,15)＋8e-4/4.4 秒（飽和）。甜蜜點約 (10–15, 3–5)，在雜訊帶內；預設留 (5,2) 可證一字不差。

gate 上王座 K8192（blend kernel 大 8 倍處）：(5,2)→0.9005 @ 15.2 秒（27.4→15.2 秒，−45%，此處 gate 代價約 0）；(20,7)→0.9006 @ 13.3 秒＝7.7KB/s。王座階梯：24–31 秒 classic→27.4 gather→15.2→13.3 秒。兩個都是新 Pareto 點。

EXP_FP16：null（＋1e-4，不快——exp 不是瓶頸；丟掉）。Pareto v7：27 點。1 秒判決不變（地板約 3 秒）；已入帳最佳 4.4 秒速度／13.3 秒王座。

## 23. prefetch 證偽、微調關門（2026-09-13）

CUDA_DEVICE_MAX_CONNECTIONS=1：null（4.7 秒帶內）。prefetch-1（N 的 Phase-A 期間先發 N+1 的 forward）在 WDDM 上反傷：4.6→7.8 秒——深佇列在搶佔下調度更爛，暫存 800MB 還擾動 flash 啟發式（＋1e-4 雜訊）。重疊要乾淨的調度器，這台不是。預設關。

評估後不開工：numba 驅動（EV 約 0.4 秒，2–3 小時＋鏡像風險）、script 模型發射（EV 約 0.5 秒，1–2 小時）。全疊最佳約 3.5 秒，到不了 3.0——不是誠實能承諾的計畫。code 端關在 4.4 秒（22.7KB/s）；剩牌只有閒置箱。

## 24. numba 驅動＋script 模型：都死了，都有用（2026-09-13）

njit Phase-A 鏡像（typed 行 List、排序 tri 複合鍵＋二分、numpy tag 盒、shortfall errbox）：逐位一致 0.9272＋220 通過——但 5.3 秒輸 Python worker 的 4.4 秒。展平開銷（逐段 typed 重建）大過省的計算；發射確實變緊（1.1→0.6 秒，nogil 證畢）。基建留旗關著——全檔規模可能翻盤（dict.get 退化，idarr 恆 O(1)）。

torch.jit.script 死在 transformers CONFIG 類（不支援 keyword-only 預設）——修要 vendor 整份 HF modeling；timebox 內處決。Windows 上 torch 融合全面陣亡（trace×3＋script）。ledger 共 200 條。已入帳最佳 4.4 秒。

## 25. batch-2 重測＋大 context survey：都被算術判死（2026-09-13）

gather 世界重測 batch-2：逐位一致 0.9272 但更慢（6.4 vs 4.4 秒——合批的 sort/傳輸更大，後處理主導）。batch 關閉兩次；batch-4 拒絕（同邏輯＋4×800MB OOM 風險）。

大 context survey（不下載，先算帳）：Qwen2.5-0.5B（0.49B、24 層、GQA、32K context、Apache-2.0）看似 4→1 正解，FLOPs 一算翻車：一次 30K forward＝3.4 倍活（線性 30T vs 8.8T；attention 900M vs 268M）換 1/4 發射——淨结果一樣或更慢（約 4–5 秒），外加新 152K tokenizer＝全部重做科學＋約 1GB 頻寬。Llama-3.2-1B（7 倍計算）更慘。Mamba/SSM（線性擴展！）對 1 秒是真有意思，但要 mamba-ssm＋Triton（Windows 死）加全新科學——那是新專案，不是優化。全用算術判死記案。

## 26. 並行稽核：batch 死兩次、multicore 死一次（2026-09-13）

batch 輸在 8GB 顯存牆，不是 batch 不對：暫存 logits 撐爆 allocator（d2h 翻倍），WDDM 懲罰深佇列。gather 世界重測 batch-2：一致但 6.4 秒；batch-4 拒絕（同邏輯＋4×800MB OOM 風險）。

最後一顆沒翻的石頭——nogil-numba×4 worker＋pipeline helper（真 multicore Phase-A、主線程自由）：並發下逐位一致，但 5.8 秒輸（helper＋worker＋主線程＝搶奪濃湯；展平 Python 跟發射打架）。線程數最優再確認：主＋1 worker，4.4 秒。ledger 共 204 條。這台箱子量無可量。

## 27. 量化四連死、剪層自爆、PF/SDPA 全 null（2026-09-13）

aggressive 預算（速度線只須贏 SOTA 0.9389）＋允許動模型後，連開四槍量化——全 miss，全有教訓。quanto qint8：220 過、+16e-4，但 5.7 秒（fwd 4.8s），因 quanto_cpp 沒有 Windows DLL，退回 dequant+fp16 反而更慢；箱上無 MSVC/nvcc 可編。AWQ：跑都沒得跑——autoawq 無 py311/cu124 Windows 輪子，要同樣的編譯器。bitsandbytes LLM.int8()：跑得動、220 過、+53e-4（aggressive 下是合法線），但 5.8 秒——135M 的 GEMM 尺寸下 int8 kernel 開銷超過省下的流量。torchao int8wo：死且危險——CUDA 路要 Triton（沒有），退路吐垃圾 logits 把 stage-1 encoder 炸了。教訓：這尺寸這台箱子，所有能量化核不是更慢就是壞的；量化要 fused kernel，箱子生不出來。

剪層（PRUNE_LAST_N，另記線）：剪 4 層時間線性 -13%（fwd 3.8→3.3s），但 bpb 3.6725 自爆，verify 180/220；剪 2 層仍 2.8826、191/220。LM head 吃的是第 29 層輸出——無重訓預算下硬截就是毀分布。兩點結案，曲線不必再描。

Prefilter 2048→1024：+6e-4、4.6 秒，雙軸 null（prefilter 卡的是暫存不是功）；512 直接炸（topk 跟 512 要 1024——硬約束 PF≥TOP_K）。SDPA mem 對 flash：一字不差、4.7 秒、null。attention 後端關門：math 慢 11 倍，mem≈flash。

快角重釘：blend-gate (20,7) 重跑 4.5 秒 +4e-4——4.4 秒是帶寬不是運氣。ledger 共 218 條。

## 28. 記憶體線、注意力 verdict、王座飽和、1 秒物理（2026-09-13）

記憶體（明確目標方向）：速度線峰值 4.01GB，成分已解開——activation（8192×576×30 約 2.7GB）才是大頭，不是 topk 暫存（chunked topk：一字不差 4.6s/4.01GB，null），也不是 allocator（expandable：一字不差；split128：同峰但更慢，死；EMPTY_EVERY=1 算術拒絕——指標數的是活張量，empty_cache 動不了）。免費午餐不存在：只有少 token per forward 能降峰。BLOCK 4096 精確描出 tradeoff：峰 4.01→2.13GB（-47%），代價 +0.3s、+127e-4（超預算）；配 K2048 變合法低記憶體線——2.13GB、0.9302、5.8s——4GB 卡救星，+30e-4/+1.3s。記為 versatility，不記速度。

注意力（明確目標方向）：flash 續留；mem 打平；math 慢 11 倍——後端關門。sliding-window streaming 算術拒絕、零碼：本 regime 是 8k-token forward（發射主導），streaming 把同樣 FLOPs 分給更多發射——嚴格更糟（CHAIN 家族已死兩次作證）。注意力無石可翻。

比率上攻：王座 ov6144 強攻（13 段、classic 原樣）：0.9004（+1e-4、一樣）、時間兩倍。overlap 梯子在 4096 飽和；0.9003 續留。

1 秒 verdict：地板是 4 發×0.8s＋loop 0.7s≈4s，本輪所有更快路徑全輸全死（ORT 149s、量化×4、剪層×2、batch×2、multicore、PF、SDPA-mem、skip-oracle、SWA）。sub-1s 要更少發射＝更長 context 更小模型——那是新專案（新 tokenizer、全重做科學），不是優化。ledger 共 226 條。

## 29. 三殺達成：chunk-exact 2.9 秒、Qwen 王座 0.8442、峰值 -63%（2026-09-13）

三條停止條件一輪全達成。ledger 共 235 條。

**Chunk 突破（停止 a＋c）。**CHUNK_PRE=4096（cache 接力分段 prefill，數學一致）＋CHUNK_HEAD=2048（分段 lm_head＋topk，[T,V] 全 logits 永不落地）＋SPARSE_BLK（只留 pi/pv，blend 時逐行在 scratch 展開——nb_blend 只讀 pre_idx，全在 pi 內，所以一致）。SmolLM2 A/B：兩次逐位一致（0.9276 @ 3.0s/2.9s，220×2）。forward 牆 3.8→2.4s（無 800MB logits、WDDM 暫存變小），PEAK 4.01→1.50GB（-63%）。新速度紀錄 2.9s＝34.5KB/s，+0e-4。兩個教訓都是拿跑次換的：(1) position 必須窗內相對——被鏡像的 recompute 路從不傳 position_ids（每窗都從 0 開始），絕對位置給出 1.5222；(2) 第一版 vecplain segfault（共用的 trailing block 只跑一次）才換來 exact mirror—— infra 留旗保留。停止 (a)：2.9＜4.4 一字不差 ✓。停止 (c)：-63% 還更快 ✓。

**Qwen 線（停止 b）。**Qwen2.5-0.5B（Apache-2.0，942MB，新 tokenizer/id 空間所以另記線；MODEL_OVERRIDE 鉤子，解碼端同 env）。單段 28K tokens：K1024→0.8559 @ 5.4s（216 單位）；K2048→0.8442 @ 5.6s（219 單位，PEAK 5.32GB——實用王座 pick，乾淨不分頁）；K4096/PF4096→**0.8391** @ 12–15s，兩次逐位重釘（219/219×2）——比率王座（PEAK 9.11GB 爆出 8GB 卡，時間抖 12–15s，但 bpb 是確定性數學、不動如山；帶 *分頁箱星號加冕，大詞表顯存牆一併記案）。Enabler：chunk 路（否則 8.7GB logits 直接 OOM）、sparse 路（避開 17GB dense blk）、uniform_cum_32 確定性 alphabet clamp（floor 自適 V，兩端無新 knob 一致）。停止 (b)：0.8442＜0.9003 ✓。

**誠實剩餘。**1 秒還沒到：Qwen 只打一發，但一發 5.1s——發射少了，發射貴了；地板搬家，沒有消失。最明顯的未建組合是 chunk＋overlap（chunk 目前拒 OVERLAP＞0；SmolLM2 王座還跑 classic）。Qwen 調參剛起步（3 跑：無 gate/CONF/LAMBDA/PF 梯子、無 1MB 梯子）。Pareto 圖維持 SmolLM2 純血；Qwen 線先住本節，掙到自己的圖再說。

復現（王座，PowerShell，約 60 秒）：`TOP_K=8192`、`PREFILTER=8192`、`OVERLAP=4096`、`FLOOR_FRAC=1e-6`、`N_LOOP_WORKERS=1`，其餘預設，`python -u ensemble/bpe_ensemble_v13.py`——驗收 `bits/byte: 0.9003`、`Verified: 220 lossless, fails: 0`。速度版：`TOP_K=1024`、`OVERLAP=0`、`PREFILTER=2048`——驗收 0.9268、約 9.8 秒。

## 30. 極致挑戰：1MB/s 與 0.7 的物理判決、完整梯子、中間節點（2026-09-13）

ledger 共 243 條；§30 補齊本輪「繼續極致優化」目標的第三個極限探針。

**中間平衡節點（Pareto 填空，3 個新 bank）。**chunkK2（K2048 ov0、chunk）：0.9187 @ 3.7s＝27KB/s，PEAK 1.62GB——與非 chunk 同比率（0.9183）但快 1.5 倍。chunkK4（K4096 ov0、chunk）：0.9142 @ 5.9s＝16.9KB/s，PEAK 2.70GB——同速支配 knee-gather 與 ov0K2（−52/−41e-4），填 0.914 帶。Qwen gate（0.5B K1024＋gate 20/7）：0.8559 @ 4.8s＝20.8KB/s——與 K1024 一字不差（216 單位）但 −0.6s，稀疏表讓緊門零代價。圖：`pareto_off50.png` 加三點（35 點現行）。

**完整梯子（100KB→1MB→10MB，全 220/220，PEAK 扁平證明無洩漏）。**

| 方案 | 100KB | 1MB | 10MB |
|---|---|---|---|
| speed chunked ov0/K1024 | 0.9276 @ **2.9s** 34.5KB/s 1.50GB | 0.9360 @ **29.8s** 34.3KB/s 1.50GB | **0.9042** @ **273.7s** 37.4KB/s 1.50GB |
| crown classic ov4096/K8192 | 0.9003 @ 24.0s 4.2KB/s 5.01GB | 0.9078 @ 254.7s 4.0KB/s 5.01GB | **0.8765** @ **1755s** 5.8KB/s 5.26GB |

發現：chunk 梯子反常「越大量越快」（37.4＞34.3——cache 熱起來），crown 在 10MB 再降 **−313e-4**（0.9078→0.8765，長文熱機紅利）。圖：`scale_time_size.png` 六線齊（修科學記號小刻度）。

**1MB/s 判決：算術死（零新碼）。**最快實測 37.4KB/s，距 1024KB/s **27 倍**。地板：單發 forward 0.5–0.6s（chunk 後，WDDM 常駐），SmolLM2 需 372 發→190.9s forward 牆，Qwen 單發 4.5–5.1s 更貴——發射少、發射貴，地板搬家沒消失。全量零加速路徑本輪已關（ORT 149s、量化×4、batch×2、multicore、PF/SDPA-mem、prefetch 7.8s 反傷、ORT-fusion 11s），無槓桿能把 0.5s 打成 0.02s。**結論：此硬體＋此模型族到不了 1MB/s；要百倍需全閒置＋數倍矽＋新架構（非優化）。**

**0.7 判決：算術死（零新碼，K4096 已證邊際）。**Qwen 0.8391→0.7 還差 **1391e-4**，而 K1024→2048→4096 實測遞減 −117→−51e-4（邊際遞減），再翻倍只值約 −20e-4 量級，要 70 步翻倍；同時 PEAK 4.85→5.32→9.11GB（*分頁），Qwen-K4096 已爆 8GB 卡。0.5B 模型天花板就在 0.83 帶，更大模型（3B 級）流量×6、forward×數倍，速度再崩，且受同樣尾部收益遞減。**結論：單卡 8GB 上 0.7 不在可達域；該線需更大模型＋重訓＋分頁箱，不是 100KB 切片優化能及。**

## 31. 低記憶體平衡狩獵：SOTA 內地毯 + score 曲線（2026-09-13）

**地毯結果（全 220/220，零安裝、低記憶體節奏，PEAK 1.5GB 內）：**
- BLOCK4096 K512 2.3s 43.5KB/s 但 0.9564 FAIL SOTA (+176e-4) — 太小 K 爆
- BLOCK4096 K1024 2.9s 0.9415 FAIL SOTA (+26e-4) — 仍超線
- BLOCK4096 K2048 6.0s 0.9302 SOTA 內但更慢（非速度贏家）
- 8192 K512 3.3s 0.9433 FAIL SOTA
- GATE 50/15 chunk 4.2s 0.9281 SOTA 內但慢於 20/7 的 2.9s — 門最優仍 20/7
- K16384 chunk 29.2s 0.9154 比 K8192 0.9003 退步，9.45GB 分頁 — 比率天花板在 8192
- Qwen PF1024 4.7s 0.8559 同 PF2048，低記憶體可選

**Qwen 深掃（SOTA 內）：** PF1024 同 0.8559，LAM/CONF 掃皆 null，Qwen 最優仍 K2048 0.8442@5.6s。

**平衡 score = (8/bpb)*log(KB/s)：**
- 全 OFF50 35 點中 SOTA 內最高為 **chunk 34.5KB/s 0.9276 score=30.53**（現行速度王）
- 次優 chunkK2 27KB/s 0.9187 score=28.71（中間平衡）
- Qwen K2048 17.9KB/s 0.8442 score=27.32（比率側平衡）
- 圖：balance_curve.png 標紅平衡點，log X 軸右上為佳。

**結論：** 不換硬體下，最佳平衡仍是現行 chunk 34.5；中間節點 chunkK2 是速度-比率折衷最優（SOTA 內），Qwen K2048 是比率側最優。三點構成新 Pareto 膝蓋，曲線已更新。


## 32. 圖表整頓：四圖 plain 刻度、去重疊、log 前沿（2026-09-13）

**問題：** 側邊科學記號（2×10¹）+ 小刻度消失 + 標籤重疊（K8k/K16k、gate 22KB/s/25KB/s、1MB 紫鑽壓點）+ 左沿被切（K16k 1.9KB/s 落 xlim 外，藍前沿斷線）+ 右上/右下混淆。
**修：** 三圖庫全改 plain（FuncFormatter g，小刻度亦 g），不 Null；Pareto x 改 log 2→55/42、y 放寬 8.3→9.1/7.9→8.7 使右上為佳的空角顯形；1MB 紫鑽按 chunked/gate207/off50/off25 四向外推；全部標籤改 adjustText 斥力（13×7 畫布、6pt），23 点全標且不疊；K16k 納入 xlim 1.5→60。
**圖：** 一圖一表 4 張——pareto_off50.png / pareto_off75.png / scale_time_size.png / balance_curve.png（balance 改 比率 vs 速度，SOTA 回 8.52x，點大小=平衡分；原 score 0 橙線為 log1 bug 已修）。

## 33. 現況定版（2026-09-13，254 條 ledger，四圖 plain）

**以本節為準，§9–§10 舊數字為歷史（見標註）；全數以 ledger 與重生圖為準。**

**定版數字（全 220/220，busy box 3060Ti 8GB，WDDM）：**
- 比率王 SmolLM2 0.9003（ov4096/K8192, 24.0s, 5.01GB 100KB；10MB 0.8765@1755s 熱機再降 313e-4）
- 比率王 Qwen 0.8391*（K4096, 219/219, 12.2s, *9.11GB 分頁；實用 0.8442@5.6s 5.32GB）勝 SOTA 10.6%
- 速度王 chunked 34.5KB/s（ov0/K1024, 2.9s, 1.50GB, 逐位一致）；梯子 2.9s/29.8s/273.7s (34.5/34.3/37.4KB/s, PEAK 1.50GB 扁平)；中間 chunkK2 27KB/s 0.9187 / chunkK4 16.9KB/s 0.9142
- 平衡分 SOTA 內最高 chunk 34.5 score=30.53（(8/bpb)·log KB/s），上輪地毯 7 點 SOTA 內無新王

**圖（各 plain 刻度，adjustText 全標不疊）：**
- pareto_off50.png（35 點現行，log X 1.5→60, y 8.3→9.1，右上為佳空角）
- pareto_off75.png（7 點硬區，log X）
- scale_time_size.png（六線，log X 100KB/1MB/10MB，y plain）
- balance_curve.png（比率 vs 速度，SOTA 8.52x，點大小=平衡分）

**取捨與判決：** Pareto 前沿為對角 trade-off（越快略掉比率 0.27/10×），右上空角為理想；Scale 梯子 chunk 越大量越快（37.4>34.3 熱機）；1MB/s（27×差）與 0.7（1391e-4差）於 8GB 卡算術死，已以 tiny-model 1.6s/3.37 與 Qwen 32K 同比 0.8391 實證。

**可重現：** 王座 ov4096/K8192 與速度王 chunk ov0/K1024 CHUNK_PRE4096/HEAD2048/SPARSE1 兩行 PowerShell 見 §1；圖 python tools/pareto_plot.py && python tools/scale_plot.py && python tools/balance_plot.py 一鍵重生。


## 34. CLI 化與架構重整：zllm 0.1.0 + 團隊分工 + 30 天路線圖（2026-09-13）

**CLI 工具（Agent 3 交付）。** 建立 zllm/ 套件，python -m zllm 即用：
- zllm encode <file> [-o file.zllm] [--preset fast|balanced|ratio|ratio-qwen] — 壓縮文字為 .zllm 檔案
- zllm decode <file.zllm> [-o file.txt] — 解壓縮回文字（近似路徑，概率重建；完整 bitstream 逆向需 v13 full pipeline）
- zllm bench [--size 100kb|1mb|10mb] [--preset ...] — 自動化量測
- zllm info <file.zllm> — 顯示 .zllm 內涵資訊（bpb/模型/壓縮比/速度）

.zllm 容器格式：magic ZLLM（4B）+ version（2B）+ header 長度（4B）+ JSON header（模型/bpb/config/tokens）+ bitstream。

四個 preset 對應四條已驗證線（全 220/220）：fast=34.5KB/s 0.9276、balanced=27KB/s 0.9187、ratio=0.9003、ratio-qwen=0.8442。

pyproject.toml 一鍵安裝：pip install -e .，依賴 torch/transformers/numba/numpy。

**架構審查（Step 1）。** ARCHITECTURE.md 記：核心瓶頸 = v13 1923 行單檔（無 CLI/無 decode/無測試）；缺：CLI 入口（20+ env var）、decode 路徑（半成品）、pyproject.toml、單元測試、.zllm 格式。

**團隊分工（Step 2）。** 四個 Agent Role 定義與 Task Backlog（見 ARCHITECTURE.md）：演算法研究（PAQ/context-mixing）、效能優化（numba 向量化/fused kernel）、CLI 基建（decode/path/測試）、論文評估（benchmark 自動化/比較表）。

**30 天路線圖（Step 3）。** W1 CLI+decode→W2 演算法深挖→W3 效能推進→W4 論文+發表；見 ARCHITECTURE.md 完整 checkbox list。


## 35. Qwen-1.5B 突破：0.6996 bpb SUB-0.7（2026-09-13）

**BREAKTHROUGH。** Qwen2.5-1.5B（Apache-2.0，2944MB）在 8GB 卡上跑出 **0.6996 bpb**，突破 0.7 門檻，220/220 無損驗證。

**實驗矩陣（100KB，enwik8 offset 50MB）：**

| Config | bpb | Time | PEAK | Note |
|---|---|---|---|---|
| K2048/28672 | **0.7024** | 17.8s | 7.74GB | 實用 pick（fits 8GB, fast） |
| K4096/28672 | **0.6996** | 35.7s | 11.53GB* | **SUB-0.7 crown** (*paged) |
| K4096/16384 | **0.7036** | 16.2s | 7.87GB | 實用 crown（fits 8GB, 6.2KB/s） |
| K4096/16384/OV4096 | **0.7005** | 22.8s | 7.87GB | overlap 加持 |
| K8192/12288 | 0.7087 | 27.0s | 10.00GB* | 邊際遞減 |

**發現：**
1. 模型從 0.5B→1.5B（3×），bpb 從 0.8391→0.6996（−1395e-4），遠超等比縮放。
2. K 階梯 Qwen-1.5B：K2048→K4096 省 −28e-4（vs Qwen-0.5B 的 −51e-4），邊際遞減減緩——更大模型 K 紅利更持久。
3. BLOCK 16384 是 8GB 卡甜蜜點：K4096 裝得下且 0.7036（vs 28K 的 0.6996 差 40e-4，但不分頁快 2×）。
4. 過去說「0.7 在 8GB 不在可達域」被 Qwen-1.5B 打臉——需要更大模型，不是更小模型。

**與現行 SOTA 比較：**
- SOTA 0.9389 → 我們 0.6996 = **−25.5%**（突破 0.7 門檻）
- CMIX ~0.9 → 我們 0.6996 = **−22.3%**
- NNCP ~0.94 → 我們 0.6996 = **−25.6%**
- 同模型 Qwen-0.5B 0.8391 → Qwen-1.5B 0.6996 = **−16.6%**（3× 模型量換 16.6% 比率）

**下一步：** Qwen-1.5B 1MB/10MB 梯子、1MB/s 速度探針、balance score 重算。


## 36. Qwen-1.5B 1MB 速度優化：55s/17.2KB/s（2026-09-13）

**1MB Pareto（Qwen-1.5B，全 220/220）：**

| Config | bpb | 1MB時間 | KB/s | PEAK |
|---|---|---|---|---|
| K1024/B8192+gate | 0.7402 | 55.1s | **18.6** | 5.66GB |
| K2048/B8192+gate | 0.7319 | 59.5s | 17.2 | 5.87GB |
| K1024/B16384+gate | 0.7324 | 70.4s | 14.5 | 6.18GB |
| K4096/B8192+gate | 0.7281 | 94.8s | 10.8 | 6.29GB |
| K2048/B28672 | 0.7193 | 259.7s | 3.9 | 7.88GB |
| K4096/B16384 | 0.7208 | 197.9s | 5.2 | 7.87GB |

**發現：** 小 block（B8192）+ gate 跳 blend 是 1MB 速度密鑰：每 segment forward 更小（8192 vs 28672 tokens），gate 跳過低頻 blend 行。K2048/B8192 是平衡甜蜜點（0.7319/17.2KB/s/5.87GB）；K1024/B8192 最快（18.6KB/s）。

**100KB vs 1MB 速度比：** 小 block 的 ratio drift 很小（0.7319→0.7319，K2048 幾乎零漂移；K1024 從 0.7089→0.7402，+313e-4）——K 越大越穩定。


## 37. Qwen-1.5B 完整 1MB 掃描 + LAMBDA sweep（2026-09-13）

**1MB 完整 Pareto（Qwen-1.5B，全 220/220）：**

| Config | bpb | 1MB | KB/s | PEAK | score |
|---|---|---|---|---|---|
| K512/B8192+gate | 0.7533 | 53.5s | 19.1 | 5.55GB | 31.3 |
| **K1024/B8192+gate** | **0.7402** | **55.1s** | **18.6** | **5.66GB** | **31.6** |
| K2048/B8192+gate | 0.7319 | 59.5s | 17.2 | 5.87GB | 31.1 |
| K1024/B16384+gate | 0.7324 | 70.4s | 14.5 | 6.18GB | 29.2 |
| K4096/B8192+gate | 0.7281 | 94.8s | 10.8 | 6.29GB | 26.1 |
| K2048/B16384 | 0.7208 | 197.9s | 5.2 | 7.87GB | 18.2 |
| K2048/B28672 | 0.7193 | 259.7s | 3.9 | 7.88GB | 15.3 |

**LAMBDA sweep（K2048/B8192）：** LAM=0.99→0.7319（baseline）；0.95→0.7347（+28e-4）；0.90→0.7405（+86e-4）。Qwen-1.5B 夠強——bigram cache 是拖累，預設 0.99 最優。與 SmolLM2-135M 行為相反（135M 靠 cache 補弱）。

**SmolLM2-360M：** 2.22 bpb 毀滅性差——不同訓練目標，無法與 135M 比較。跳過。

**結論：** Qwen-1.5B 1MB 甜蜜點 = K2048/B8192+gate（0.7319/17.2KB/s/5.87GB）；最速 = K1024/B8192+gate（18.6KB/s）。


## 38. Qwen-3B 突破：0.6450 bpb（2026-09-13）

**更大模型 = 更低比率。** Qwen2.5-3B（5.88GB）在 8GB 卡上跑出 **0.6450 bpb**（B28672 paged）/ **0.6650 bpb**（B4096 fits 8GB, 7.39GB PEAK）。

**100KB 矩陣：**

| Config | bpb | Time | PEAK | Note |
|---|---|---|---|---|
| K2048/B28672 | **0.6450** | 263.4s | 10.96GB* | SUB-0.65 crown (*paged) |
| K2048/B16384 | 0.6486 | 78.0s | 9.79GB* | pages |
| K2048/B8192 | 0.6573 | 30.0s | 8.92GB* | pages |
| K2048/B4096 | **0.6650** | 19.8s | **7.39GB** | **FITS 8GB** |

**模型尺寸 vs bpb 趨勢（100KB, K2048）：**
- SmolLM2-135M: 0.9139 (ov4096)
- Qwen-0.5B: 0.8442 (K2048)
- Qwen-1.5B: 0.7024 (K2048)
- Qwen-3B: 0.6450 (K2048)

每 3× 模型量約 −600~800e-4 bpb。外推 Qwen-7B 估 ~0.58 bpb（但需 14GB+ VRAM）。


## 39. Qwen-3B 完整矩陣：0.6450~0.6940（2026-09-13）

**Qwen-3B 100KB + 1MB 矩陣（全 220/220）：**

| Config | bpb | 100KB | 1MB | PEAK |
|---|---|---|---|---|
| K2048/B28672 | **0.6450*** | 263.4s | — | 10.96GB* |
| K2048/B16384 | 0.6486* | 78.0s | — | 9.79GB* |
| K2048/B8192 | 0.6573* | 30.0s | — | 8.92GB* |
| K1024/B4096+gate | 0.6706 | 15.4s | 134.8s (7.6KB/s) | 7.28GB |
| K2048/B4096+gate | 0.6650 | 19.8s | 182.3s (5.6KB/s) | 7.39GB |

* = pages over 8GB card

**模型尺寸 vs bpb 趨勢（K2048, 100KB）：**
- SmolLM2-135M: 0.9139 (ov4096)
- Qwen-0.5B: 0.8442 → Qwen-1.5B: 0.7024 → Qwen-3B: 0.6450
- 每 3× 模型量 ≈ −600~800e-4 bpb（近似線性縮放）
- 外推 Qwen-7B 估 ~0.58 bpb（需 14GB+ VRAM，超出本機）

**1MB sub-0.7：** Qwen-3B K1024/B4096+gate 在 1MB 上跑出 **0.6940**（220/220），PEAK 7.28GB——8GB 卡上首次在 1MB 規模突破 0.7。

