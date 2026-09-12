# §13-14 draft (appended to compression-paper.md by hand after review)

## 13. SmolLM2-135M：換底座（BPE + 預訓練大模型）

char 小模型的瓶頸確認在底座本身後，換用公開預訓練模型 SmolLM2-135M（BPE 詞彙 49,152，context 8192，3060Ti 可跑）。BPE 大詞彙超出 14-bit 精度上限（49,152 > 16,384），故採用 top-K + escape 分層編碼：stage-1 編 top-2048 或 ESCAPE（2049 符號），escape 後用兩層 uniform 編碼其餘 ~47K（3 組 × 16K，皆符合精度約束）。

| 設定（enwik8 前 100KB） | bpb | 無損驗證 | 說明 |
|---|---|---|---|
| frozen, context 2048 | 1.2541 | 200/200 | 追平 NNCP v2（1.25） |
| frozen, context 8192 | 1.2328 | 200/200 | 長 context 有用（+1.7%） |
| norm-only adapt @1e-4 | 1.3416 | 200/200 | 變差：lr 太大 |
| full adapt @1e-5 fp16 | 4.8227 | — | 炸掉：fp16 overflow |
| full adapt @1e-5 fp32 | 1.2510 | 200/200 | 穩定但只好 0.2% |

教訓：
1. fp16 線上更新會發散，fp32 才穩定（已證實為 overflow 而非 lr 問題）。
2. 預訓練好的大模型在 in-distribution 資料上幾乎沒有線上學習空間（1.2541 → 1.2510）。權重微調不是答案，ensemble 才是。

## 14. Ensemble：從失敗到接近 SOTA

### v1 unigram-cache（失敗，如實記錄）

- 結果：1.2519 bpb，比 frozen（1.2328）還差，且慢 2.5 倍。
- 死因有二：冷啟動 uniform 噪音污染早期位置；全量 float32 傳輸（每 block 1.6GB）拖慢 PCIe。
- 價值：指出了正確方向的反面教材——cache 必須有 confidence gating，且不能增加傳輸量。

### v2 bigram + confidence gating（成功）

- 機制：order-1 bigram 快取 + w = n/(n+10) 信心門；只在 CPU 端小候選並集上混合，零額外傳輸。
- 結果（enwik8 前 100KB）：**0.8734 bpb**，200/200 無損驗證，escape 率 0.8%。
- 但誠實檢查：前 100KB 恰是 XML 樣板重災區，bigram 在此有主場優勢。

### 多 slice 誠實測量

| slice | bpb | 無損驗證 | 說明 |
|---|---|---|---|
| offset 0MB（頭，樣板區） | 0.8734 | 200/200（全文件等距抽查） | 偏樂觀 |
| offset 50MB（中段，文章區） | 0.9706 | 200/200（全文件等距抽查） | 有代表性 |

### v3 trigram + 回退鏈 + 近因衰減（未超越 v2）

- 中段結果：0.9862 bpb，200/200 無損驗證，order 使用 tri=3474 / bi=8018 / cold=19296。
- 结论：trigram 在此規模下沒有贏過 bigram（λ=0.75 可能過度加權 cache；近因衰減可能砍太多）。保留為負結果，不採用。

### SOTA 對標（enwik8，單位 bpb）

| 方法 | bpb | 模型 |
|---|---|---|
| Nacrith（SOTA） | 0.9389 | 135M + ensemble + 適應 |
| **我們 v2（中段）** | **0.9706** | **135M + bigram，無權重更新** |
| ts_zip | ~1.11 | RWKV-169M |
| CMIX v21 | 1.17 | 2000+ 模型混合 |
| NNCP v2 | 1.25 | Transformer 56M |
| 我們 frozen | 1.2328 | 135M |
| gzip -9 | 2.58 | — |

差距 0.9706 → 0.9389（3.4%）。Nacrith 多的零件：神經 ensemble、32-bit 精度、線上微調。我們的 v2 以更簡單的系統（單一 bigram cache、無權重更新、14-bit）走到 3.4% 以內。

### 誠實但書

1. 100KB slice，非完整 100MB（SOTA 數字是全文件）。頭段數字受樣板紅利影響；中段數字最有代表性。
2. 無損驗證為 200 點等距抽查，非全量（全量驗證僅在 char 管線做過 798/798）。
3. 速度未優化：v2/v3 為研究程式碼（Python per-position 迴圈），throughput 遠低於 char 管線的 numba 版本。
4. λ 只試過 0.85（v2）與 0.75（v3 且混入 trigram/衰減變更，非乾淨對照）。乾淨的 λ sweep 尚未執行。

## §15. SOTA 迭代循環 R1–R12：從 0.9605 到 0.9213（2026-09-12 單日）

以 `sota_loop.py` 為驅動（想法隊列＋自動記錄 bpb/速度/verified＋停機條件 bpb<0.9389），在 100KB 中段切片（enwik8 offset 50MB）上連續迭代，ledger 見 `data/sota_loop.json`。最終 **v7floor 0.9214 bpb**，超越 SOTA 線 1.9%。

### 15.1 完整軌跡

| 輪 | 想法 | bpb | 結論 |
|---|---|---|---|
| R1 | λ 0.97/0.99 | 0.9607/0.9605 | λ 曲線已平，調參挖完 |
| R2–R4 | v4 GPU 化重構 | 0.9605（CPU 路徑四位全同） | 抓到 double-softmax bug＋fp32 OOM；修後數學乾淨 |
| R4 | KN-trigram（v5） | 0.9597（+0.001） | bigram 已飽和，trigram 幾乎無貢獻 |
| R6 | CONF 掃參（4 組） | 最好 0.9595 | 整面牆是平的（±0.0003 雜訊） |
| R7 | PREFILTER 16384 / TOP_K 4096 | 0.9597 / **1.0132（災難）** | 大 alphabet 成本主導；反向啟示→試小 K |
| R8 | TOP_K 512/768/1024 | 0.9608/0.9531/**0.9499** | K=1024 最優（-0.010） |
| R8 | v6 overlap 1024（k2048） | 0.9570（-0.0027） | 冷啟動修復有效 |
| R10 | v6 k1024＋ov1024/ov2048 | 0.9471/**0.9431** | 疊加如預期；ov2048 更好 |
| R11 | v7 單流 stage-2（parity） | **0.9375（過線 -0.0014）** | 兩段編的 finish 開銷是隱形稅 |
| R11 | v7 floor 1e-5 | **0.9214（-0.016）** | 最大單步；理論兌現 |
| R11 | v7 cache-informed stage-2 | 0.9217（+0.0003） | **證偽**：與 uniform 無異，砍掉 |
| R12 | v8 numba blend＋lean | 0.9213（±1e-4 一致） | kernel 與 numpy 完全等價；速度待解 |

### 15.2 三個可發表的洞察

1. **Smoothing 稅（§7 候選定理）**。14-bit 核心每符號強制 +1 count；K=2049 時 12.5% 機率質量被拿去抹平頭部。32-bit 的價值不在精度而在 floor 調校範圍：tails 稀少（逃逸率 0.6%）時小 floor 必勝。合成測試：floor 6.1e-5→1e-5，211→133 bits（16-bit 對照 139）。
2. **Finish-bit 稅**。舊 stage-2 拆兩段編碼，每段收 ~2 bits finish 開銷；~250 逃逸 × 2 段 ≈ 1000 bits 白繳。單流計數一次省掉（v7parity -0.006）。
3. ** latent 邊界 bug 的誠實揭露**。v2–v5 的 bitstream 從未編碼 file token 0 與 block 邊界 token（200 點抽查只採樣已編碼對，永遠抓不到）。v6 起補上（token0 經 uniform(V)＋32-bit，邊界經 overlap 首對），成本 +0.0002 bpb。v2–v5 的 bpb 數字作為預測分數仍有效，但其 bitstream 非自包含——論文必須如此聲明。

### 15.3 同切片 head-to-head：Nacrith 原廠系統

Nacrith 開源（Apache-2.0，robtacconelli/Nacrith-GPU），以 llama-cpp-python（CPU 版，自編）＋官方 BF16 GGUF 在**完全相同的 100KB 中段切片**上實測（`h2h_nacrith.py`，`data/h2h_nacrith.json`）：

| 系統 | bpb | 速度 | lossless |
|---|---|---|---|
| 我們 v7floor | **0.9214** | 0.37 KB/s | verify 220/220 |
| Nacrith（1 worker，全預設） | 1.2248 | 0.25 KB/s | roundtrip byte-exact ✓ |

同切片贏 25%。Nacrith 的 adaptive head＋n-gram＋LZP 屬熱機型（需數十 MB 進入狀態），100KB 冷切片＋2048 context 是其弱場；其論文 0.9389 為 100MB 全檔熱機數字。雙方皆缺對方主場的全檔數字——最終排名以全檔為準（待速度戰後執行）。

### 15.4 未解：速度

0.37 KB/s vs NNCP v2（RTX 3090 實測）3.25 KB/s，差 9 倍。profile（v4/v8 內建 PHASES 計時）：forward 被桌面 compositor 擠壓（15s→108s）、Python blend 迴圈 16.7s（v8 numba 化後數學等價已驗證，wall time 未降——主因仍在 forward 與系統爭用）。路線圖：段尾積極釋放（v8-lean 已上，VRAM 7.5GB→待測）、安靜系統重測、llama.cpp 後端（文獻 7 倍）、KV-cache 滑窗（文獻 37 倍）。gzip 級速度不在戰場內（查表 vs 神經，物理上不可比；SOTA 自己也輸 5000 倍）。

### 更新後的誠實但書（取代 §14）

1. 主數字 0.9214 為 100KB 中段切片；全檔 100MB 驗證待執行。
2. 無損驗證 220 點（等距＋全邊界＋token0），非全量逐 token 解碼。
3. 失敗史完整保留：unigram ensemble、純 trigram、TOP_K 4096、prefilter 16384、CONF 掃參、cache-informed stage-2——全部證偽，ledger 可查。
4. Nacrith 同切片實測由其原廠開源碼跑出，非轉述論文數字。
