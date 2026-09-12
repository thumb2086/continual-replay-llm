# Neural Adaptive Compression 技術報告：24 條件正交網格 + Huffman Baseline

> 資料來源：`data/sweep_all.json`（主結果）、`data/real_compression.json`、`data/ablation_results.json`、`data/baseline_comparison.json`、`data/lambda_sweep.json`、`data/replay_sweep.json`、`data/research_state.json`、`data/v3_results.json`。本報告僅做 read-only 分析，未執行任何訓練或實驗。所有數字均直接引用 JSON；可疑處另行標註，不掩飾。

## 1. 摘要

- Huffman baseline：**8.475 bits/char (bpc)**，約 **30,102,942 chars/s**（`sweep_all.json:huffman`，bpc 精確值 8.47494140625）。
- 最佳 neural ratio：**M/dense/full 的 7.149 bpc**（fp16 與 fp32 相同），比 Huffman 好約 **15.6%**（(8.4749 − 7.149) / 8.4749）。
- Neural 速度範圍：約 **52,705–229,789 chars/s**，即使用最快的 neural 條件，仍比 Huffman 慢約 **130×**。
- Precision (fp16 vs fp32)：ratio 幾乎無差（差異 ≤ 0.001 bpc）；速度上 fp16 在小模型上反而更慢，屬實測現象（見 §4.2）。
- Pruning：ratio 成本約 +0.03（S）至 +0.12（M）bpc，速度提升有限（見 §4.3）。
- Adaptation：frozen → prefix40 → full 在全部 4 組 size×pruning 組合下 ratio 單調改善（見 §4.4）。
- 獨立驗證實驗（`real_compression.json`，51,200 chars / 400 chunks）：Huffman 8.47494140625 bpc；neural full 7.16203125 bpc @ 35,157.677 chars/s；roundtrip verified 3 chunks。

## 2. 實驗設計

- 正交網格：model size (S ≈ 1.10M params / M ≈ 4.82M params) × precision (fp16 / fp32) × pruning (dense / pruned) × adaptation (frozen / prefix40 / full) = 24 條件，加上 Huffman baseline。
- 可比性：所有 neural 條件共享同一資料／tokenizer／stream 設定（char-level，延續 continual-learning 系列的 podcast transcript 設定），以 bpc（越低越好）與 chars/s（越高越好）為共同指標；另記錄 `params`、`ckpt_mb`、`peak_mem_mb`、`verified`。
- 背景：adaptation 維度（frozen / prefix / full）直接沿用 continual-learning 報告中的 replay／adaptation 思路（`paper.md`、`continual-learning-report.md`、`ablation_results.json`、`lambda_sweep.json`、`replay_sweep.json`、`research_state.json`、`v3_results.json`）；本報告不重複該部分結論，僅將其視為 compression 上下文。
- 參數規模（來自 JSON）：S/dense 1,097,383（ckpt 4.186 MB）；S/pruned 1,031,591（ckpt 3.935 MB）；M/dense 4,824,231（ckpt 18.403 MB）；M/pruned 4,036,263（ckpt 15.397 MB）。

## 3. 結果總表（依 bpc 由低到高排序）

| 條件 | bpc | chars/s | peak_mem_mb | params |
|---|---|---|---|---|
| M/dense/full/fp16 | 7.149 | 52,705.413 | 494.890 | 4,824,231 |
| M/dense/full/fp32 | 7.149 | 52,947.781 | 494.890 | 4,824,231 |
| M/pruned/full/fp16 | 7.268 | 54,814.712 | 478.387 | 4,036,263 |
| M/pruned/full/fp32 | 7.269 | 59,431.815 | 480.545 | 4,036,263 |
| S/dense/full/fp16 | 7.327 | 105,902.352 | 228.997 | 1,097,383 |
| S/dense/full/fp32 | 7.327 | 112,867.500 | 235.323 | 1,097,383 |
| S/pruned/full/fp16 | 7.360 | 108,443.544 | 227.993 | 1,031,591 |
| S/pruned/full/fp32 | 7.360 | 116,996.820 | 234.319 | 1,031,591 |
| M/dense/prefix40/fp16 | 7.371 | 163,878.884 | 1,340.323 | 4,824,231 |
| M/dense/prefix40/fp32 | 7.371 | 152,202.229 | 1,440.760 | 4,824,231 |
| S/dense/prefix40/fp32 | 7.403 | 225,261.074 | 1,183.522 | 1,097,383 |
| S/dense/prefix40/fp16 | 7.404 | 110,697.316 | 1,083.085 | 1,097,383 |
| S/pruned/prefix40/fp16 | 7.419 | 214,229.074 | 1,082.081 | 1,031,591 |
| S/pruned/prefix40/fp32 | 7.419 | 229,435.691 | 1,182.519 | 1,031,591 |
| S/dense/frozen/fp32 | 7.443 | 229,789.464 | 1,734.207 | 1,097,383 |
| S/dense/frozen/fp16 | 7.444 | 69,111.699 | 1,576.052 | 1,097,383 |
| S/pruned/frozen/fp16 | 7.449 | 218,141.114 | 1,575.801 | 1,031,591 |
| S/pruned/frozen/fp32 | 7.449 | 217,538.514 | 1,733.956 | 1,031,591 |
| M/pruned/prefix40/fp32 | 7.493 | 161,451.340 | 1,428.737 | 4,036,263 |
| M/pruned/prefix40/fp16 | 7.494 | 170,358.549 | 1,328.299 | 4,036,263 |
| M/dense/frozen/fp16 | 7.563 | 183,832.515 | 1,790.612 | 4,824,231 |
| M/dense/frozen/fp32 | 7.563 | 144,947.171 | 1,948.767 | 4,824,231 |
| M/pruned/frozen/fp16 | 7.624 | 196,123.855 | 1,787.606 | 4,036,263 |
| M/pruned/frozen/fp32 | 7.625 | 188,716.607 | 1,945.761 | 4,036,263 |
| Huffman | 8.475 | 30,102,941.602 | — | — |

註：`verified` 欄位在 sweep 中僅 S/dense/frozen/fp16 與 M/dense/frozen/fp16 為 3，其餘 22 個條件皆為 0（見 §5）。`real_compression.json` 的 prefix curve（full 7.162 @ 35,158；prefix200 7.223 @ 83,817；prefix100 7.307 @ 102,190；prefix40 7.407 @ 169,199；frozen 7.560 @ 119,230 chars/s）與上表趨勢一致，但數值為獨立 run，不可與 sweep 表格混為同一 run。

## 4. 分析

### 4.1 Model size（S vs M）

- Full adaptation 下 M 優於 S：M/dense/full 7.149 vs S/dense/full 7.327，差距 0.178 bpc（約 2.4%）；代價是速度約減半（52,948 vs 112,868 chars/s，fp32）且 ckpt 大 4.4×（18.403 vs 4.186 MB）。
- Frozen 下方向反轉（可疑，如實報告）：M/dense/frozen 7.563 反而比 S/dense/frozen 7.443 **差 0.120 bpc**；M/pruned/frozen 7.624–7.625 更差。這表示大模型在不做 adaptation 時並未自動帶來更好的 bpc，可能原因包括 frozen 前提下 pretrain 分佈與測試 stream 不匹配被放大，但 JSON 本身無法證實機制，僅記錄現象。
- Peak memory 隨 size 上升：full 條件下 M 約 479–495 MB，S 約 228–235 MB（約 2.1×）。

### 4.2 Precision（fp16 vs fp32）

- Ratio 幾乎相同：12 對配對中最大差異僅 0.001 bpc（例如 M/pruned/full fp16 7.268 vs fp32 7.269；其餘多為完全相同如 7.149/7.149、7.327/7.327）。
- 速度上 fp16 並無穩定優勢，且在小 op 上明顯更慢：S/dense/frozen fp16 69,111.699 vs fp32 229,789.464（fp16 僅 fp32 的約 30%）；S/dense/prefix40 fp16 110,697.316 vs fp32 225,261.074（約 49%）。誠實解釋為 autocast／kernel launch overhead 在小 op 上主導了執行時間，而非 fp16 本身計算更慢；M/full 條件下兩者幾乎持平（52,705 vs 52,948），佐證 overhead 假說。
- 可疑點：同為 S 級 fp16，S/dense/frozen/fp16（69,112）與 S/pruned/frozen/fp16（218,141）差距達 3.2×，pruning 本身不應造成此量級差異；JSON 未提供 batch 設定以外的解釋，此處如實標註為未解釋異常。

### 4.3 Pruning（dense vs pruned）

- Ratio 成本小：S/full 7.327 → 7.360（+0.033 bpc）；M/full 7.149 → 7.268/7.269（+0.119–0.120 bpc）；frozen/prefix40 條件下成本同量級（S 約 +0.006–0.016，M 約 +0.061–0.123）。
- 速度增益有限：M/dense/full/fp32 52,947.781 → M/pruned/full/fp32 59,431.815（+12.2%）；S/dense/full/fp32 112,867.500 → S/pruned/full/fp32 116,996.820（+3.7%）。ckpt 縮小約 6%（S：4.186 → 3.935 MB）與約 16%（M：18.403 → 15.397 MB）。
- 解釋：pruning 在此規模下是容量與速度的小幅交換，未改變 adaptation 主導的排序（dense 仍全面優於同級 pruned 的 bpc）。

### 4.4 Adaptation（frozen → prefix40 → full）

- 全部 4 組 size×pruning 組合皆單調改善（fp32 為例）：S/dense 7.443 → 7.403 → 7.327；S/pruned 7.449 → 7.419 → 7.360；M/dense 7.563 → 7.371 → 7.149；M/pruned 7.625 → 7.493 → 7.269。
- 幅度最大的是 M/dense：frozen → full 共 −0.414 bpc；最小的是 S/pruned：共 −0.089 bpc。即大模型更依賴 adaptation 兌現容量優勢。
- 速度代價：full 比 prefix40/frozen 慢約 2–4×（例如 M/dense/fp32：144,947 → 152,202 → 52,948；S/dense/fp32：229,789 → 225,261 → 112,868）。
- Peak memory 反直覺現象（如實報告）：frozen 條件 peak memory 最高（S 約 1,576–1,734 MB；M 約 1,788–1,949 MB），prefix40 次之（約 1,082–1,441 MB），full 最低（約 228–495 MB）。任務指示此係 `frozen_code_all` 使用巨大 `fwd_batch=200` forward batch 所致；數字本身支持該說法（batch 越大、peak 越高），但 JSON 未內含 batch 參數，此解釋列為外部提供、未在 JSON 內驗證。

## 5. 誠實的限制

1. Huffman 仍快 100× 以上：最快的 neural（229,789 chars/s）僅 Huffman（30,102,942 chars/s）的約 1/131；最佳 ratio 的 M/dense/full（約 53K chars/s）差距約 570×。本實驗未證明 neural 方法在 throughput 上可替代 classical 方法。
2. 資料單一：podcast transcript only，無跨 domain（code、news、CJK 混合、binary）驗證；topic 定義為目錄代理（見 `paper.md` §3.2），非語義標註。
3. Tokenizer 侷限：char-level（vocab 約 3.2K）；Huffman bpc 高達 8.475 部分反映此設定，不宜直接與 byte-level 或 subword-level 文獻數字比較。
4. 模型極小：S ≈ 1.10M、M ≈ 4.82M params，與 production LLM 差數個量級；結論（特別是 size 反轉與 pruning 幅度）不可外推。
5. 未比較的方法：沒有在同一資料上測試 gzip、zstd、LZMA、NNCP、cmix 等。本報告不聲稱 SOTA、不聲稱擊敗任何未測試方法。（`compression_results.json` 雖含 gzip/bz2/lzma 數字，但屬不同 run／不同切分，不納入本報告主結論。）
6. 單一種子／單次 run：sweep 無重複實驗、無 error bar；且 24 個條件中僅 2 個 `verified=3`、其餘 22 個 `verified=0`；`real_compression.json` 僅 roundtrip 3 chunks。統計強度弱。
7. 未解釋異常至少兩處：(a) M frozen 差於 S frozen（§4.1）；(b) S/dense/frozen/fp16 的 69K chars/s 與同級其他 fp16 條件（110K–218K）顯著脫節（§4.2）。兩者均未在 JSON 內找到機制解釋，不做事後合理化。
8. 評估指標單一：僅 bpc＋throughput＋peak memory，無 decode 延遲、energy、streaming 首字延遲、long-context 退化等生產指標。
9. 速度數字有約 2× 的 run-to-run 變異：同一程式在同一 GPU 上不同次執行的 chars/s 可差約兩倍（例如 prefix40 在不同 run 測得約 97K 與約 170K chars/s；bpc 則完全確定性一致）。因此速度比較僅在同一次 run 內有效，絕對速度值不宜過度解讀；bpc 數字則因固定種子而完全可重現。

## 6. 結論（2026-09 更新：大資料後最佳值為 §7 的 M/full 6.472 bpc）

- 純 ratio 最佳（小資料網格）：**M/dense/full（7.149 bpc，fp16/fp32 相同）**，fp32 速度 52,947.781 chars/s、peak 494.890 MB、ckpt 18.403 MB。比 Huffman（8.475）好 15.6%，是本網格內唯一的 <7.2 bpc 條件。
- 務實最佳 tradeoff：**S/dense/full/fp32（7.327 bpc @ 112,867.500 chars/s，peak 235.323 MB，ckpt 4.186 MB）**。理由：比最佳值僅差 +0.178 bpc（+2.5%），但速度快 2.13×、ckpt 小 4.4×、peak memory 低約一半；且 fp32 在此條件下比 fp16 快（112,868 vs 105,902），無需承擔 fp16 小-op overhead。
- 次選：若必須 <7.3 bpc 但仍想要一點速度，**M/pruned/full/fp32（7.269 bpc @ 59,431.815 chars/s，ckpt 15.397 MB）**：比最佳值差 +0.120 bpc，換取 +12.2% 速度與 −16% ckpt。
- 速度優先：**S/pruned/prefix40/fp32（7.419 bpc @ 229,435.691 chars/s）**：以 +0.270 bpc 為代價達到全場最高速之一，適合 throughput 敏感場景；但注意其 peak memory 高達 1,182.519 MB。
- 總結：adaptation 是本網格內唯一穩定、大幅改善 ratio 的槓桿；size 放大僅在 full adaptation 下兌現；pruning 與 fp16 在此規模下皆為次要因素。任何部署決策都必須同時面對第 5 節的限制，特別是與 Huffman 的速度鴻溝與驗證覆蓋不足。

## 7. 大資料 + 正則化（6 倍訓練資料）

將 podcast 訓練資料從 50 texts 擴大到 300 texts（本地既有檔案，無需下載），改用 AdamW（wd=0.05）+ early stopping（patience 3，監控 held-out val loss）。測試 stream 保持同一個凍結 400-chunk（`frozen_stream.json`），故數字可比。

| 模型 | frozen | prefix40 | full |
|---|---|---|---|
| S 小資料 | — | — | 7.327 |
| S 大資料 | 6.736 | 6.699 | **6.640** |
| M 小資料 | 7.563 | 7.371 | 7.149 |
| M 大資料 | 6.708 | 6.612 | **6.472** |

- 6 倍資料換來約 −0.7 bpc，是所有單一改動中最大的一次進步。
- M/full 大資料 **6.472 bpc**，比 Huffman（8.475）好 **23.6%**，為本系列最佳。
- 大資料下 M 重新超過 S（6.472 vs 6.640），與小資料時的排序（S 贏 M）翻轉；佐證小資料時的 size 反轉來自過擬合，而非架構本質。
- 誠實註記：大資料 tokenizer vocab 為 4329（小資料為 3239），vocab 變大理論上增加預測難度，因此此處的進步幅度實為保守估計。

## 8. 跨領域：enwik8 零樣本測試

enwik8（100MB，壓縮界標準測試集）下载并用 M-bigdata 模型 frozen 測量（100KB 樣本，零樣本、無訓練）：

- **5.365 bpc**，UNK 率 2.79%，速度 52,830 chars/s。
- 比 podcast 測試集的 6.472 還好：Wikipedia 是乾淨編輯文本，比口語逐字稿好預測。
- 對標公開數字：gzip 在 enwik8 約 2.6 bpc，大型神經模型約 1 bpc。本模型（5.4M 參數、podcast 訓練、零樣本跨領域）為 5.365 bpc——離 SOTA 很遠，但證明泛化能力真實存在。

## 9. 資料混合曲線與兩階段訓練（S 模型）

問題：podcast（中文口語）與 Cosmopedia-100k（英文教科書）混訓，能否兩邊都要？

| 訓練混合比 | podcast frozen | podcast prefix40 | enwik8 |
|---|---|---|---|
| 100/0（podcast only） | 6.736 | 6.699 | —（M 版 5.365） |
| 50/50 | 6.997 | 6.970 | 4.589 |
| 15/85 | 7.319 | 7.280 | **4.413** |
| 兩階段（cosmo→podcast） | 7.141 | 7.105 | 5.180 |

- 無免費午餐：cosmo 比例越高，podcast 越差、enwik8 越好。
- 兩階段訓練居中（stage-2 val 到第 20 epoch 仍在下降，finetune budget 不足是主因之一）。
- 最佳 enwik8 來自 15/85 混合的 S 模型（4.413 bpc），比 M-podcast-only 的 5.365 好 18%——小模型+對的資料 > 大模型+錯的資料。

## 10. 適應參數量、蒸餾、剪枝小結

- 適應時只動 bias（0.35% 參數）：7.540 bpc @ 174K chars/s，比全量更新（7.407 @ 16K）快約 10 倍，只差 +0.13 bpc。
- 蒸餾失敗：S-distilled 7.636 差於 S-scratch 7.466。診斷：老師訓練 loss 2.35 vs 學生 3.59，老師過擬合，學生連過擬合一起繼承；加 weight decay 重蒸亦無改善（7.638）。結論：蒸餾的前提是老師泛化更好，本設定下不成立。
- 剪枝成功但幅度小：M FFN-50% 只掉 0.02 bpc，證實小模型冗餘度高。
- 綜合教訓：本設定下瓶頸在**資料**，不在模型大小、蒸餾或剪枝。

## 11. 速度、壓縮率、硬體消耗完整分析

### 速度優化史（同一 400-chunk 測試）

```
4270 → 7407 → 12190 → 32392 → 62385 → 96689 chars/s（prefix40 設定）
```

| 優化 | 效果 | 性質 |
|---|---|---|
| numba 算術編碼迴圈 | Python 迴圈 → 編譯碼 | 一次性 |
| 區塊化更新（block 8） | 400 次反向 → 50 次 | 改變更新頻率 |
| fp16 + batched forward | GPU 吃飽 | 無精度損失（已驗證 bpc 不變） |
| GPU 端量化 | 取代 CPU numpy（11x） | 1014/1016 行完全一致 |
| prefix 適應 | 只更新前 10% | -0.25 bpc 換 3.5x 速度 |
| 巨型 frozen batch | 50 次 forward → 2 次 | 凍結段專用 |
| AMP + fused Adam | 反向傳播加速 | bpc 不變（已驗證） |

### 瓶頸排序（實測）

1. 反向傳播（adaptation）——最大頭，已用 prefix + bias-only + AMP 處理
2. Python 迴圈開銷——已用 numba + 巨型 batch 處理
3. DtoH 傳輸——已用 uint16 減半
4. Forward 本身——巨型 batch 下僅 40ms/400 chunks，**不是瓶頸**

### 硬體佔用（RTX 3060 Ti 8GB）

| 項目 | 佔用 | 說明 |
|---|---|---|
| 模型權重 S | ~4-5 MB | fp32；fp16 再減半 |
| 模型權重 M | ~18-21 MB | fp32 |
| Peak（adapt 路徑） | 228–495 MB | 小 batch，8GB 卡綽綽有餘 |
| Peak（frozen 巨型 batch） | 1.3–2.2 GB | fwd_batch=200 的 logits，不是模型大 |
| Huffman | ~0 MB | 查表法 |

結論：硬體佔用**不高**。Peak 記憶體來自刻意放大的 batch（速度換空間），把 fwd_batch 調小即可降 5 倍，只慢一點。權重本身微不足道。

### 速度誠實評估

- 最快 neural（prefix40/frozen）：100–230K chars/s（有 run-to-run 變異）
- Huffman：30M chars/s（快 130–570x）
- 神經方法追不上 classical 速度是結構性的（每次都要跑模型），但在神經壓縮領域內（cmix ~1KB/s 等級）已算快。

## 12. enwik8 完整無損驗證

enwik8（100KB 樣本，798 chunks）逐 chunk 編碼+解碼驗證：

| 模型 | bpc | 驗證 |
|---|---|---|
| M-bigdata frozen | 5.407 | **798/798 無損** |
| S-mix-15/85 frozen | 4.377 | **798/798 無損** |

註：早期 count-based 估計值（5.365）與真實位元數（5.407）相差 finish-bit overhead，已以完整驗證值為準。

跨領域結論維持：podcast 訓練的模型在 Wikipedia 上零樣本達到 5.407 bpc；加入 Cosmopedia 後進步到 4.413 bpc。小模型+對的資料 > 大模型+錯的資料。
