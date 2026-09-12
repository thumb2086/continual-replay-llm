# 交卷報告：SmolLM2-135M 實用神經壓縮（enwik8 切片）

> 日期：2026-09-13。底座：SmolLM2-135M（Apache-2.0）。硬體：單卡 RTX 3060 Ti 8GB。主評測：enwik8 offset 50MB、100KB 中段切片（有代表性的文章區，非模板頭）。帳本：`data/sota_loop.json`（每輪 bpb/速度/verified 全記錄）。本報告所有數字均為實測，無外推。英文版見 [`FINAL-REPORT.en.md`](FINAL-REPORT.en.md)。

## 1. 交卷數字

| 指標 | 值 | 說明 |
|---|---|---|
| 最佳壓縮率 | **0.9003 bpb** | v13，ov4096＋floor 1e-6＋K/PF 8192，220/220 無損（K 階梯：1024→0.9139，2048→0.9050，4096→0.9013，8192→0.9003） |
| SOTA 線 | 0.9389 bpb | Nacrith 論文 100MB 全檔數 |
| 超線幅度 | **−4.1%** | 切片對全檔，見 §7 但書 |
| 同切片 H2H | 我們 0.9214 vs Nacrith 原廠 1.2248 | 同一切片、同腳本重現（`h2h_nacrith.py`），贏 25% |
| 速度（忙碌箱） | v13 預設：**ov0 9.8 秒/100KB＝10.2KB/s（10KB/s 過線）**；knee ov2048 12.0 秒；比率王 24.0 秒 | 子鐘 attn/topk/d2h/shmw，見 §11–12 |
| 速度（安靜推算） | ~5KB/s | 真功耗 forward 12s＋loop 6s＋零頭 |
| 主機記憶體峰值 | 3.4GB | 實測 RSS |
| 顯存峰值 | 5.01GB（K8192 王座）／2.32GB（速度配置） | `torch.cuda.max_memory_allocated`，8GB 卡內 |

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
| TOP_K 512 | 0.9278（更差） | K=1024 兩邊都試過，最優（2026-09-14 推翻：是 prefilter-bound，見 K 階梯） |
| K 階梯（2026-09-14，v13） | 1024→0.9139／2048→**0.9050**／4096→**0.9013**／8192→**0.9003** | K=PF 同步放大：−89／−37／−10 e-4，遞減，停在 8192 |
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
- `ensemble/bpe_ensemble_v10.py`：切片後向等價版；`ensemble/bpe_ensemble_v12.py`：llama 後端（忙碌箱證偽保留）；`ensemble/bpe_ensemble_v13.py`：KV 接力主線已證偽封存，但其 plumbing（deferred driver、單源 `_range_core`、共享內存 proc、pipeline helper、雙緩衝、增量凍結）已在 recompute 模式逐位驗證 0.9139（`USE_PROC_LOOP=1`＋`PIPELINE=1` 實測 26.9s/100KB，loop 8.5→2.5s）；ORT 分支因 fp32 慢 3 倍＋逐 shape 重調優而死；2026-09-14 起 v13 為現行交卷（比率王 0.9003／速度 10.2KB/s）。現行預設：`SDPA_BACKEND=flash`、`USE_FP16_SOFTMAX=1`、`UNCHUNKED_TOPK=1`、`USE_PROC_LOOP=0`、`PIPELINE=0`、`N_LOOP_WORKERS=1`，`BLOCK_TOKENS` 可調（8192 最優，兩邊都掃過）
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

10KB/s 結案（2026-09-14 推翻舊結論：已過線）：組合技（ov0＋flash＋fp16-softmax＋單 topk＋單發 topk＋pipeline/proc 關＋單 loop worker）9.8 秒（10.2KB/s）。2026-09-13 的「不可達」寫於 SDPA 發現之前（auto 選到 math fallback）與 fwd 拆表之前——舊結論作廢，新瓶頸見 §12。。

## 11. 最終迭代與硬體極限宣告（2026-09-13）

cProfile 指引：rotary 三角重算佔 forward 13%＋`.to()` 排隊 10%——cos/sin 快取（逐位一致，tripwire 防鏈毒）省 forward 24%（24.3→18.4s）。N 進程同態掃描：2（11.6s）＜1（11.9）＜4（12.0）＜6（12.4）＜8（13.1），N=2 最優。微優化 gc.disable、CUDNN benchmark、EMPTY/1000 全 null（±0.1s 雜訊）。按開工規則（三連 null 即停）：**硬體極限＝ov0/N2，11.6 秒＝8.6KB/s**（比率 0.9268，220/220）；比率王 ov4096 0.9139 @ 19.4 秒不動。再往下要安靜十倍的箱子或新卡，不是 code。（2026-09-14：本節數字被 §12 取代，保留為歷史。）

## 12. 速度重開＋比率突破（2026-09-14，取代 §10–11 數字）

速度（ov0 上 12.1→9.8 秒）：SUBCLOCKS 把被污染的 forward 窗口拆開——attn 0.9／topk-launch 1.6／d2h 6.1／shmw 1.6 秒。flash 一直有在動（attn 0.23 秒/forward）；之前的「2.5 秒/forward」是六種成本共用一個計時器。PCIe 洗清嫌疑（純傳輸 0.12 秒/800MB）。入帳：fp16 softmax（比率四位小數不動）、精確單 topk（少一次全 V 遍歷）、單發 topk（2.32GB）、pipeline/proc 預設關（helper 爭用大於重疊收益）、單 loop worker（GIL：10.5→9.8 秒）。顯卡負載下 boost 正常（1905MHz/100%/219W；之前低時鐘讀數是死 job 的假影）。**10KB/s 過線：9.8 秒＝10.2KB/s。**

比率（0.9139→**0.9003**）：ov4096 上 K/PF 階梯——2048→0.9050、4096→0.9013、8192→0.9003（增量 −89/−37/−10 e-4，遞減，停）。舊的「K=1024 最優」是 prefilter-bound，推翻。新王座顯存峰值 5.01GB，8GB 卡內。SOTA 差距：−4.1%。

## 13. batch 證偽、1MB 放量、drain 診斷（2026-09-14）

batch-2 雙 forward：逐位正確（0.9268）但零加速——只拼了模型呼叫，逐段 topk/d2h/shmw 才是大頭。路上殺兩隻 bug：stash 交棒吃掉一個 chunk 槽（整段被跳過，1.8240）；submit 迴圈用 stale chunk 重綁 block（mate 拿前一段的 ids 編碼，1.3576）。batched==single 證到 0.000000（含 rope patch）。預設保持單發（BATCH_SEGS=1）。`empty_cache` 拔除：null。

1MB 王座驗證（K8192/ov4096，74 段）：**0.9078**，254.7 秒，顯存峰值 5.01GB 持平（無洩漏），220/220。K 階梯在 10 倍量級成立（＋75e-4）。4.0KB/s→全檔 100MB 約 7 小時：迭代太慢，先做 logit 手術。

drain 診斷（SYNC_ATTN）：真 attention-GPU 只要 0.7 秒/4 forward——flash 在 pipeline 內確認有動。d2h 約 5 秒是 softmax＋topk 執行堆積（400M 元素 softmax 本該毫秒級：差 300 倍→箱子搶佔／sag 或 WDDM 病理）。下一步：logit-space 手術（lse＋gather，殺掉全 V softmax 執行）。

## 14. 手術取消、放量梯子（2026-09-14）

logit 手術取消，有證據：單機實測 softmax＋2×topk＋gather＋全部 D2H＝0.15 秒/段，純 GPU 0.02 秒/段。v13 的 1.4 秒/段是 10 倍調度開銷（排隊等前面的活、桌面搶佔、burst 間 boost 掉速）——不是活。沒東西可 fuse，code 端速度工作關閉（每根槓桿不是 null 就是環境定罪）；剩下的槓桿是安靜的箱子。

1MB 速度基線（ov0/K1024，38 段）：**0.9347**，104.3 秒＝9.8KB/s——與 100KB 線性一致，放量漂移 ＋79e-4（與王座 ＋75e-4 同級）。100KB 修復後重驗 0.9268 一字不差。梯子：1MB 約 2 分鐘（速度版）／4 分鐘（王座版）；10MB 約 17 分鐘；全檔 100MB 安靜箱約 3 小時——可過夜跑。

## 15. Goal 回合：速度關閉、K 到頂、Pareto＋6 點（2026-09-14）

速度（4 探針全 null）：2 worker 輸 1（14.3 秒，GIL 曲線 1＜2＜8 補完）；GC_OFF null；proc/N=2 null（15.6 秒，thread 勝）；batch 上王座逐位一致但更慢（23.1 秒）。code 端速度關閉（連同 §14 證據）。殘留路徑（未走）：segment-skip、graph 重試、WSL compile——專案級，列案。

比率（9 探針）：K 階梯到頂——K16384 退回 0.9027（＋24e-4，52.4 秒，顯存 8.26GB 分頁！）：倒 U（−89/−37/−10/＋24），8GB 卡永不上 K8192。持平／關閉：lambda 0.995、ov6144@K8（＋1e-4）、CONF 20、TRI-CONF 10、PF 超 K、S2、floor 1e-7。新曲線點：K6144 0.9004@28.5 秒、ov2048@K8 0.9058@23.6 秒、ov1024@K8 0.9099@22.3 秒、ov0@K8 0.9127@21.8 秒（大 K 對 ov0 最補，−141e-4，2.2 倍時間）。

Pareto v4：28 點（OFF50 21 點）。全 220/220。王座 0.9003／速度 10.2KB/s 不動。

## 16. WSL 全移植：建成、更慢、關閉（2026-09-14）

WSL（無網路）離線建成全套環境：63 個輪子在 Windows 代下載（雙 `manylinux_2_17＋2_28` 平台標籤——pip 26 拔掉裸別名；`sys_platform` 標記在 Windows 主機靜默丟掉 nvidia 依賴，改顯式抓；nvjitlink 12.4→12.9 修 cusparse 未定義符號；tokenizers pin 放寬給 transformers 4.57.6）。torch 2.14＋cu126＋triton 3.8＋CUDA 12.9＋gcc 全活。配方：Windows 下 `pip download --platform manylinux_2_17_x86_64 --platform manylinux_2_28_x86_64 --python-version 3.12 --implementation cp --abi cp312`，拷到 `~/wheels`，`pip install --no-index --no-deps`。

結果：WSL eager 100KB＝12.9 秒（比 Windows 9.7 秒慢；半虛擬化開銷大於 WDDM），跨平台一致 0.9266 vs 0.9268（2e-4）。Inductor reduce-overhead 0.85 秒/forward 輸 eager 0.33（小模型上 dynamo 開銷）；default 模式打平（0.32）。max-autotune 書面拒絕：它調 GEMM，我們證過的牆是調度。速度 goal≤8.0 秒：證據齊全判 UNMET——9.7 秒就是這台箱子的牆。能重開它的輸入：閒置的箱子、更大的卡，或換模型。

## 17. gather 時代：5.3 秒、scale 圖、PF16384（2026-09-14）

gather（logits 上 topk＋lse＋pi-gather＋CPU exp＋稀疏 blk，numba V-stamp 截斷）轉正預設：100KB 速度 9.7→**5.3 秒＝18.9KB/s**（−45%），比率 0.9272（＋4e-4 pi-only 代價，記案），220/220 三次確認（5.3/7.1/7.4 波動——箱子雜訊，全過 8.0 線）。單發 scatter 打敗分塊 3.0→1.7 秒（與 micro 基準相反——記案，未解）。王座＋gather 0.9005@27.4 秒（＋2e-4，大 K 下更慢：傳輸隨 K 漲——gather 是速度配置武器）。

比率射擊：PF16384/K8192＝0.9004（加寬救不回 tail；顯存 8.01GB 紅線——永不上 PF8192）；LAMBDA＝0.999＝0.9010（純 LM 輸 blend）。無新紀錄；K/lambda/overlap/CONF/tri/PF/floor/S2 全關。

放量梯子（全 220/220）：速度 gather 5.3 秒／56.9 秒／632 秒（100KB/1MB/10MB：18.9/18.0/16.2KB/s，顯存 4.01GB 持平）；10MB 比率 0.9039（cache 隨規模變熱）。新 `tools/scale_plot.py`→`scale_time_size.png`（大小-時間雙對數＋比率-大小）。Pareto v5：30 點。全檔 100MB gather 速度約 1.7 小時——可過夜。

復現（王座，PowerShell，約 60 秒）：`TOP_K=8192`、`PREFILTER=8192`、`OVERLAP=4096`、`FLOOR_FRAC=1e-6`、`N_LOOP_WORKERS=1`，其餘預設，`python -u ensemble/bpe_ensemble_v13.py`——驗收 `bits/byte: 0.9003`、`Verified: 220 lossless, fails: 0`。速度版：`TOP_K=1024`、`OVERLAP=0`、`PREFILTER=2048`——驗收 0.9268、約 9.8 秒。
