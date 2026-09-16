# 報告審閱：`FINAL-REPORT.md`（LLM + 算術編碼神經壓縮）

> 審閱日期：2026-09-16 ｜ 審閱對象：`FINAL-REPORT.md`（繁中交卷報告，678 行／§1–§48）
> 交叉核對：`data/sota_loop.json`（312 條帳本）、`ensemble/bpe_ensemble_v13.py`、`ac32.py`、`zllm/cli.py`、`ARCHITECTURE.md`、`data/*.json`、公開基準（LTCB／arXiv）
> 審閱環境限制：本沙盒無 GPU、無 torch／transformers、無外網，**無法重跑任何實驗**。因此以下結論全部來自（a）程式碼逐行閱讀、（b）帳本與 JSON 交叉比對、（c）公開基準查證，以及（d）一個不依賴 torch 的可重現小實驗（`tools/audit_zllm_stream.py`）。

---

## 0. 一句話判決

這份報告的**工程紀錄與證偽文化是真的、品質很高**；但它把一個**「模型機率下的理想碼長估算」（ideal code length estimate）**包裝成了**「世界第一的無損壓縮器」**——而整條主線程式**不產生任何位元流、沒有解碼器**，所謂的「220/220 無損」是每個檔案固定抽 220 個點、每點只做「單一符號編碼→解碼」的往返測試。兩者之間差一個 bitstream 的距離。

---

## 0.5 值得肯定的部分（不能因為下面的問題就抹掉）

1. **帳本文化**：`data/sota_loop.json` 312 條，含 `dead`／`null`／`fail`／`error` 的失敗紀錄，每條都有 id／狀態／bpb／秒數；這是很多論文等級專案都做不到的。
2. **證偽清單**（§6）貨真價實：quantization 四連敗、batch-2 兩次、prefetch 反傷、CUDA graph 吐零、剪層自爆、KV 接力證偽……這些負面結果自己寫出來，價值不低於正面結果。
3. **§7 誠實但書**主動說明「切片非全檔／驗證是抽樣／Nacrith 跑在弱場／backend 容差」——問題是後面的 §43–§48 與 README 頭條沒有繼承這些但書。
4. **工程最佳化有實測支撐**：chunked prefill+head（峰值 −63%、時間 −37%）、gather（−45%）、rope 快取（−24%）、增量凍結、SHM/process path 的逐位一致驗證，這些都是真功夫。
5. **可重現性設計**：parity gate（重構必須逐位一致）、bpb-EXACT gate、每跑存 JSON，這些習慣是對的。

---

## 1. 報告在做什麼（先講對的部分）

方法本身是合理的、甚至是聰明的：

1. 用現成 LLM（SmolLM2-135M／Qwen2.5-0.5B/1.5B/3B/7B）對文本做自迴歸預測；
2. 取 top-K 詞元，配上 bigram／trigram 快取混合（Kneser-Ney 風格、confidence 門控）與 floor 平滑；
3. 用 32-bit 算術編碼（`ac32.py`，TOTAL32 = 2²⁰）把每個詞元的機率轉成位元數；
4. `total_bits / 原文位元組數` = bpb。

這條路線是對的：它等價於 **H(text) = Σ H(tokenᵢ | context)**，而分母用 bytes，所以 3B 模型若平均 2.3 bits/詞元、約 3.6 bytes/詞元，就會得到 ~0.65 bpb。也就是說，**只看那個數字，它是自洽的**（不是亂算）。問題全部出在「它是什麼」與「它被說成什麼」。

---

## 2. 關鍵發現 A：沒有位元流、沒有解碼器

### 2.1 主線只有「計數」，沒有壓縮檔

`ensemble/bpe_ensemble_v13.py`：

| 位置 | 內容 |
|---|---|
| L1088、L1956、L2152 | `total_bits += int(nb_encode_count_32(...))` —— 只算位元數 |
| L2213–2216 | 輸出 JSON 只有 `total_bits`、`bpb`，**沒有任何檔案輸出** |
| 全檔 grep `wb` | 唯一的寫檔是 L2009 的除錯傾印 `logs/tables_seg*.pkl` |

`ac32.py` 的 `nb_encode_count_32()` 是「跑一次算術編碼的重正規化、數它會吐出幾個位元」，這是標準的碼長計算，**但它不回傳、也不保存任何位元**。

**團隊自己也知道**，`ARCHITECTURE.md` 白紙黑字：

```
Input text -> ... -> 32-bit arithmetic coded
-> Bitstream (COUNTED ONLY, no file I/O)
Decode path: ArithmeticDecoder exists but no reverse pipeline.
```

以及缺口表第一列：`No decode path (half a product) — CRITICAL`。

### 2.2 「220/220 lossless」的真正含義

`bpe_ensemble_v13.py` L1345–1460：

- 抽樣條件（L1373–1375）：`(OVERLAP>0 且是段落起始) 或 (已編碼位置 % 130 == 0)`，且 `len(stored) < 300`；
- 驗證迴圈（L1437）：`if vok + vfail >= 220: break`；
- 每個抽到的點做什麼（L1439–1471）：用**當時存下來的機率表** `v_p` 重建 CDF → 編一個符號 → `finish()` → 開一個解碼器解回來 → 比對。

所以「220/220」= **每份輸入固定最多 220 個抽樣點、每點一個符號的往返**。它證明的是「編碼器與解碼器在同一張 CDF 上互為逆運算」——這是最基本的一致性，**不能**證明：

1. 解碼端能從已解出的歷史**重建出同一張機率表**（需要 fp16 GPU 前向、快取狀態、top-K tie 順序完全逐位一致）；
2. 存在一個可被別人讀回來的檔案（**根本沒有檔案**）；
3. 100MB 全檔正確（抽樣率 = 220 / 約 27,000,000 個已編碼位置 ≈ **0.0008%**；而且 100KB 和 100MB 的抽樣上限**都是 220**）。

報告 §7.2 確實承認「無損驗證 220 點，非全量逐 token 解碼」——**但 §43–§48 與兩份 README 的頭條表格把「220/220 roundtrip」放進「Lossless check／無損檢查」欄位**，讀者只會理解成「全檔無損」。

### 2.3 反證：報告自己就報過跨平台漂移

§16 說跨平台一致 2e-4；§45 卻把同一個 1MB 設定寫成 **0.9432（Windows）vs 0.9359（WSL）＝ 73e-4**（而該列 Windows 數字本身可疑，見發現 C #3）；Qwen-3B 100KB 也有 0.6706 vs 0.6712 的 6e-4 平台差。可見比率**不是**逐位決定性的——而「解碼端要重建同一張表」正是最怕這個的東西。報告 §7.5 把這種差異輕描淡寫成「±1e-4 級 backend 容差」，與自家 73e-4 的事實不符。

對照組：這份 repo **真的做過全量無損往返**——`ensemble/verify_enwik8_full.py` + `data/enwik8_full_verify.json`（char-level 舊管線，798/798 chunks，5.407 bpc）。**團隊會做真正的驗證，只是沒把它用在頭條系統上。**

---

## 3. 關鍵發現 B：對外比較基準引錯、且性質不對等

報告與 README 的對照表寫：

| 報告說法 | 同語料的真實數字 | 判定 |
|---|---|---|
| CMIX **~0.90 bpb**（enwik8, full 100MB） | **CMIX v21 enwik8 = 14,623,723 B = 1.170 bpb**（enwik9 = 0.864） | 把 enwik9 的數字當 enwik8 用 |
| NNCP **~0.94 bpb @ 3.25 KB/s** | NNCP v2 enwik9 = 0.914 @ 3.25 KB/s；**enwik8：NNCP v2 = 1.25、v3.2 = 1.193 bpb** | 同上（連速度都是 enwik9 的） |
| 「比 CMIX 快 **25x**」（§43） | 24.9 KB/s ÷ 1.6 KB/s（cmix v21 enwik8）≈ **15.5x** | 高估 |
| PAQ8px、ts_zip、FineZip、Delétang 皆未列入 | PAQ8px ≈1.27、ts_zip ≈1.11、FineZip 1.024、Chinchilla-70B ≈1.6（enwik8） | 文獻面沒有查證 |

失誤方向對對手有利（0.90 比 1.17「更好」），所以「贏 CMIX 26%」這種說法是**保守的**——但這仍是一張**不同語料互比**的表。

更嚴重的是**性質**問題：Nacrith 的 0.9389 是**實際產生的壓縮檔**（11,737,280 B，能還原原文）；本報告的 0.6646／0.8968 是**碼長估算**（沒有檔案）。把兩者放進同一張表排名，等於拿估價單跟成交價比。

（附帶：報告 §43 說「我們達到 enwik8 全檔 100MB 的**最佳已發表 bpb**」——這是一句關於**整個文獻**的斷言，但報告沒有任何文獻調查章節，唯一的對外錨點就是 Nacrith 的 0.9389。而 0.6646 會比所有已發表結果好約 29%：這個幅度大約等於 enwik8 這個 benchmark 十幾年的進步總量。）

---

## 4. 關鍵發現 C：內部不一致清單（都可查證）

| # | 報告說法 | 出處 | 對照證據 | 判定 |
|---|---|---|---|---|
| 1 | Qwen 100KB+1MB 矩陣「**全 220/220**」 | §39 標題 | 帳本：`qwen3b-k2b4` 218/218、`qwen3b-k2b28` 219/219、`qwen3b-k1b4` **218/220（2 fail）**、`qwen3b-b2048-k512` **216/220（4 fail）** | 與帳本不符 |
| 2 | 帳本「282 條」 | §1 | 實際 `len(history) = 312`（§33 寫 254、GOAL.md 寫 308） | 過期數字 |
| 3 | 「WSL 快 **2.23 倍**」 | §45、§48 | WSL 1MB 用的是 chunked 設定（`tools/wsl_bench_1mb.py`：CHUNK_PRE=4096/K1024/gate），其 Windows 對照應為帳本 `1m-chunk` 的 **29.8s**（chunked+gate, 0.936, 34.3KB/s）→ 26.7s vs 29.8s ≈ **1.12x**。而 §45 的 Windows 列（59.5s / 17.2 KB/s）在帳本裡找不到對應的 SmolLM2 執行，卻與帳本 `qwen15b-k2b8`（Qwen-1.5B 1MB：59.5s、17.2KB/s）**完全相同** | 基準疑為他模型／他設定 |
| 4 | §47 表格標題「結果（Windows）」 | §47 | 0.6712 / 28.2s / 3.6KB/s / 7.33GB 正是帳本 `wsl-qwen3b-b4096`；Windows 是 15.4s / 6.5KB/s / 7.28GB | 標籤錯誤，結論建立在慢的那組 |
| 5 | 「每 3× 模型量 ≈ −600~800e-4」 | §38/§39 | 實測：135M→0.5B −697、0.5B→1.5B **−1418**、1.5B→3B −574、3B→7B −234 | 擬合與自家數據不符 |
| 6 | 「100MB 比 100KB 好 = 快取熱機」 | §43 | 100KB／1MB／10MB 切片都是 offset 50MB、100MB 是全檔（含 offset 0）；自家 §9 顯示 off0 的 100KB 就 0.8307、off75 是 0.9662（±8% 位置效應），8 片平均 0.9037 ≈ 全檔 0.8968 | 因果未控制（位置效應未被排除） |
| 7 | SmolLM2-360M「2.22 bpb 毀滅性差，跳過」 | §37 | 同一模型 §46 變成「新 Pareto crown 0.7960」 | 矛盾未解釋 |
| 8 | 「0.7 在 8GB 卡不可達」 | §30 | §35 立刻用 Qwen-1.5B 打臉（0.6996） | 修正正確，但 §30 沒加「已推翻」註記 |
| 9 | GOAL：100MB「≥17 KB/s（**<600s**）」 | `.opencode/GOAL.md` | 17 KB/s × 600s = 10MB；100MB 需 6024s | 自身矛盾 |
| 10 | 「比 NNCP 快 7 倍」／「比 CMIX 快 25 倍」 | README §43/§44 | vs nncp v3 (3.25KB/s) 7.7x ✓；vs cmix 15.5x ✗ | 一對一錯，且彼此矛盾 |

---

## 5. 關鍵發現 D：交付的 `zllm` CLI 不是可用的編解碼器

`zllm/cli.py`（520 行，§34 稱「Agent 3 交付」）有三個可由程式碼直接判定的缺陷；我另外寫了可重現實驗 `tools/audit_zllm_stream.py`（純 Python 重實作 16-bit coder 數學，不需 torch）：

**(1) 每個符號開一個新的編碼器，再把碎片串起來 —— 不是一條算術碼流**

```python
for t_idx in range(len(block) - 1):
    ...
    enc = ArithmeticEncoder(store=True)   # 每個符號一個全新編碼器
    enc.encode_symbol(cum, s1)
    bitstr, count = enc.finish()          # 每個碎片自帶終止碼
    for b in bitstr: all_bits.append(int(b))
```

實測（`tools/audit_zllm_stream.py`）：

```
[1] 單一連續編碼器            : decode -> [3, 17, 250, 0, 999, 41]   OK
[2] zllm 風格（每符號獨立編碼）: decode -> [3, 5, 566, 0, 918, 0]
    最長正確前綴 = 1 個符號
```

**(2) bpb 被放大 8 倍**

`finish()` 回傳的是**位元字串**（長度 = `count` 位元），CLI 把每個位元逐一放進 `all_bits`，然後：

```python
total_bits = len(all_bits) * 8      # len 已經是「位元數」，再乘 8
```

實測：`per-symbol counts=[7,9,12,4,15,10]` → `len(all_bits)=57` → 報告寫成 `456`，**8.00x 膨脹**；而實際寫入檔案的位元組數約 7 B。也就是說：`zllm bench` 印出來的 bpb 大約是它自己寫出檔案大小的 8 倍。

**(3) 解碼端根本不是逆向，而且結構上不可能無損**

`cmd_decode` 不逐位還原，而是**重跑模型**、用「已解出的前綴」重新生成機率再讀符號；遇到 escape（top-K 以外的詞元，報告說約 0.6%）直接取 `rest[0]`。程式自己也印：

```
NOTE: decode is approximate (reconstructed probabilities, not exact bitstream reversal)
For lossless roundtrip, use the full encoder/decoder pipeline in v13
```

而 v13 沒有任何 decoder（見發現 A）。另外 `PRESETS` 的說明字串直接複製 v13 的數字（`"0.9003 bpb"`、`"0.9187 bpb, 27KB/s"`），但這條程式路徑沒有 chunk／gather／blend／cache，**量不出那些數字**；repo 內沒有任何 `.zllm` 檔，帳本也查不到 zllm 的執行紀錄。

**其他基建**：`pyproject.toml` 的 `build-backend = "setuptools.backends._legacy:_Backend"` 是無效路徑（應為 `setuptools.build_meta`），§34 的「`pip install -e .` 一鍵安裝」不成立；`requirements.txt` 只有 `torch`、`numpy`，缺 §34 聲稱的 `transformers`、`numba`；`src/index.js` 是空殼。

---

## 6. 哪些數字可以信、哪些不能

| 類別 | 內容 | 依據 |
|---|---|---|
| ✅ **可信（內部自洽 + 有 artifact）** | SmolLM2-135M 在 100KB／1MB／10MB／100MB 的 bpb、時間、KB/s、VRAM（JSON 與報告逐項對得上：0.9003/92195 bits/102400 B；KB/s ↔ 秒數 ↔ 段數自洽）；WSL 與 Windows 的 100MB 同為 0.8968；工程優化幅度（chunked −63% 峰值、−37% 時間） | `data/smollm2_ensemble_v13_*.json`、帳本 |
| ⚠️ **未驗（只有帳本文字，無 artifact）** | Qwen 全線：0.6450／0.6996／0.8391／0.8442／0.6646(100MB)、SmolLM2-360M 0.7808、Qwen-7B 0.6216。repo 內找不到任何 Qwen 的 JSON／log／檔案 | `data/` 內 0 個 Qwen 檔 |
| ❌ **已被程式碼否證** | 「無損」、「220/220 lossless 全檔」、「zllm 四 preset 已驗證」、「pip install -e .」 | 見發現 A、D |
| ❌ **對外基準錯誤** | CMIX ~0.90、NNCP ~0.94（見發現 B）、25x 加速、WSL 2.23x | 見發現 B、C |
| 🟡 **性質正確但被誤讀** | 「bpb」＝**模型機率下的理想碼長**。若正名為 rate estimate，SmolLM2 線（全檔 0.8968 vs Nacrith 同為 135M 的 0.9389）是合理可比的，差 4.5% 在可信範圍；Qwen-3B 的 0.6646（比全體已發表結果好 29%）則遠超合理邊界 | 需要真實檔案才能裁決 |

---

## 7. 附帶：持續學習那條線（`paper.md`、`continual-learning-report.md`）

- 主張溫和（「小規模技術報告，不宣稱通用優越性」），replay-size sweep 與 `data/replay_sweep.json` 一致（起始 A=3.7817、B=5.0867 對得上）。
- **但**報告主結果（A 3.9473→3.9185、B 5.2451→4.6102）在 repo 內沒有對應 JSON；唯一形狀相近的 `data/continual_results.json` 反而是**退化實驗**：baseline / after_english / after_chinese 三階段 loss **完全相同到 15 位有效數字**（4.559871101379395），且 `cn_learned: false`——那不是「沒有遺忘」，是「什麼都沒學到」。
- 定位問題：repo 名叫 *continual-replay-llm*、README 開頭寫「Replay-first 持續學習——用在神經壓縮」，但**壓縮主線用的是凍結 LLM + n-gram 快取，沒有任何 replay／EWC／持續學習成分**。兩條線其實沒有交會。

---

## 8. 要讓這些主張站得住：最小行動清單

1. **寫出真正的位元流**：stage-1 每個 segment 一條、escape 併入同一條流；存檔、報**檔案位元組數**（不是 `nb_encode_count_32`）。這一步會立刻暴露 bpb 與真實大小的差距（finish bits、對齊、標頭）。
2. **寫 v13 的反向 pipeline**：解碼端從頭重建 top-K／blend／cache／位置對齊。跨平台要嘛鎖定 bit-determinism（CPU fp32 或固定 kernel），要嘛把「解碼需同一環境」寫成格式的一部分。
3. **把驗證改成全量**：編完存 SHA-256，解完比對原文；220 點抽樣只留在 smoke test。
4. **修 `zllm`**：8× bug、單流設計、escape 失真、`build-backend`、`requirements.txt`；並補上「preset 數字由誰跑出來的」。
5. **重做對照表**：同語料同指標（enwik8 100MB：cmix v21 1.17／nncp v3.2 1.19／ts_zip 1.11／FineZip 1.024／Nacrith 0.9389），並在表頭註明「本系統數字為 ideal code length，尚無壓縮檔」。
6. **修正 §39／§45／§47／§30 的數字與標籤**，並把 README 的「無損檢查」欄位改名為「抽樣往返（220 點）」。

做完 1–3 之前，「世界第一」的結論**不成立**；做完之後，這份工作有可能真的很有意思——尤其是 SmolLM2-135M 全檔 0.8968 那一條線，它跟 Nacrith 同樣用 135M 模型、只差 4.5%，是這份報告裡最扎實、最值得繼續推的結果。

---

## 附錄：主要查證來源

| 項目 | 位置 |
|---|---|
| 計數而非編碼 | `ensemble/bpe_ensemble_v13.py` L1088, L1956, L2152, L2213–2216；`ac32.py::nb_encode_count_32` |
| 抽樣驗證 | `ensemble/bpe_ensemble_v13.py` L1373–1375, L1437–1460 |
| 自述無檔案／無解碼器 | `ARCHITECTURE.md`（"COUNTED ONLY, no file I/O"；"no reverse pipeline"；缺口表） |
| CLI 缺陷與可重現實驗 | `zllm/cli.py`（PRESETS L24–60、encode L107–210、`total_bits` L210、decode L269–366）；`tools/audit_zllm_stream.py`（本審閱新增） |
| 帳本 | `data/sota_loop.json`（312 條；Qwen 條目索引 231–311） |
| 舊管線真實驗證 | `ensemble/verify_enwik8_full.py`、`data/enwik8_full_verify.json`（798/798，5.407 bpc） |
| 外部基準 | LTCB `mattmahoney.net/dc/text.html`（cmix v21 enwik8 14,623,723 B／1.17 bpb）；Bellard NNCP v2/v3（enwik8 1.25／1.19）；Nacrith arXiv 2602.19626 + nacrith.com（0.9389 bpb／11,737,280 B）；sota2 enwik8 表（ts_zip 1.11、FineZip 1.024、PAQ8px 1.27、Delétang ≈1.6） |

---

# 附錄 B：批次修正清單（對應「第 2、3 項」，可直接套用）

> 用途：一次改完 `FINAL-REPORT.md`、`FINAL-REPORT.en.md`、`README.md`、`README.zh-Hant.md` 的對外引述與內部數字。
> 原則：**改完之後，你的數字會變好，不會變差**（因為原本引用的對手數字被低估了——詳見 B.1 註）。

## B.1 對外基準：權威數字（同語料、同指標）

| 系統 | enwik8 bpb | enwik9 bpb | 速度 | 來源 |
|---|---|---|---|---|
| CMIX v21 | **1.170**（14,623,723 B） | 0.864 | ~1.6 KB/s（622,949 ns/byte） | LTCB `mattmahoney.net/dc/text.html` |
| NNCP v2 | 1.250 | **0.914** | 3.25 KB/s（enwik9） | Bellard NNCP v2 論文 Table 1 |
| NNCP v3.2 | **1.193** | — | ~4.1 KB/s（enwik8, 241,871 ns/byte） | LTCB |
| PAQ8px | ~1.27 | — | — | LTCB / sota2 |
| ts_zip（RWKV-169M） | ~1.11 | — | — | Nacrith §2.4、sota2 |
| FineZip（LLaMA-3-8B） | 1.024 | — | — | arXiv 2409.17141 |
| Chinchilla-70B（Delétang） | ≈1.6 | 0.664 | — | Nacrith Table 1 引述 |
| Nacrith（SmolLM2-135M） | **0.9389** | — | — | arXiv 2602.19626 |

**註（重要）**：舊報告把 **enwik9** 的 0.90（CMIX）與 0.94（NNCP）當成 **enwik8** 數字。修正後你的領先幅度**變大**：

| 比較 | 報告現值 | 修正後 | 變化 |
|---|---|---|---|
| SmolLM2 0.8968 vs CMIX | 「贏 CMIX ~0.90」＝−0.4% | vs **1.170** ＝ **−23.4%** | 大幅變好 |
| Qwen-3B 0.6646 vs CMIX | −26.2% | vs **1.170** ＝ **−43.2%** | 變好 |
| Qwen-1.5B 0.6996 vs CMIX | −22.3% | −40.2% | 變好 |
| Qwen-1.5B 0.6996 vs NNCP | −25.6% | vs **1.193** ＝ −41.4% | 變好 |
| 速度 vs CMIX（SmolLM2 24.9 KB/s） | 「快 25x」 | 24.9/1.57 ＝ **15.9x** | **變差，須下修** |
| 速度 vs CMIX（Qwen-3B 6.88 KB/s） | 「快 7x」 | 6.88/1.57 ＝ **4.4x** | **變差，須下修** |
| 速度 vs NNCP（24.9 KB/s） | 「快 7.7x」 | 24.9/4.04 ＝ **6.2x** | 微降 |

### 逐處替換（含行號）

**`FINAL-REPORT.md`**

| 行 | 現在 | 改成 |
|---|---|---|
| 3 | `帳本：data/sota_loop.json（282 條）` | `（314 條）` |
| 17 | `每 3× 參數量 ≈ −600~800e-4 bpb` | `實測逐步：135M→0.5B −697e-4／0.5B→1.5B −1418e-4／1.5B→3B −574e-4／3B→7B −234e-4（非定值，邊際遞減）` |
| 358 | `CMIX ~0.9 → 我們 0.6996 = −22.3%` | `CMIX v21 enwik8 1.170 → 我們 0.6996 = −40.2%` |
| 359 | `NNCP ~0.94 → 我們 0.6996 = −25.6%` | `NNCP v3.2 enwik8 1.193 → 我們 0.6996 = −41.4%` |
| 399 | `SmolLM2-360M：2.22 bpb 毀滅性差…跳過。` | 保留，但加註：`（該筆為設定錯誤；§46 在正確設定下為 0.7960，已取代此結論）` |
| 423 | `每 3× 模型量約 −600~800e-4 bpb。外推 Qwen-7B 估 ~0.58` | 改為實測值（同上行 17 寫法），並註明 7B 實測為 0.6216、不需外推 |
| 428 | `**Qwen-3B 100KB + 1MB 矩陣（全 220/220）：**` | `（各設定驗證點數見下表，非全部 220/220）` |
| 443 | `每 3× 模型量 ≈ −600~800e-4 bpb（近似線性縮放）` | 同上行 17 的實測四步；**「近似線性」不成立**（−1418 是平均值的兩倍） |
| 507 | `（SOTA+CMIX+NNCP 全贏）` | 保留，但把 CMIX/NNCP 數字改為 1.170／1.193 |
| 520 | `- CMIX: ~0.90 bpb (full 100MB)` | `- CMIX v21 enwik8: 1.170 bpb（0.864 是 enwik9，勿混用）` |
| 524 | `- NNCP: ~0.94 bpb, 3.25 KB/s` | `- NNCP v3.2 enwik8: 1.193 bpb, ~4.1 KB/s（0.914@3.25KB/s 為 v2/enwik9）` |
| 525 | `better ratio AND 7.7x faster` | `better ratio AND 6.2x faster` |
| 538 | `| CMIX | ~0.90 | ~1 KB/s | 200+ | CPU ensemble |` | `| CMIX v21 | 1.170 (enwik8) | ~1.6 KB/s | 2000+ | CPU context mixing |` |
| 539 | `| NNCP v2 | ~0.94 | 3.25 KB/s | 1 | CPU neural |` | `| NNCP v3.2 | 1.193 (enwik8) | ~4.1 KB/s | 1 | CPU/GPU neural |` |
| 539 後 | — | 補列 `ts_zip ~1.11`、`FineZip 1.024`、`PAQ8px ~1.27`、`Delétang Chinchilla ≈1.6`（皆 enwik8） |
| 543 | `being 25x faster than CMIX` | `being 15.9x faster than CMIX` |
| 543 | `We achieve the best published bpb on enwik8 full 100MB` | `Among the systems we surveyed, …`（**刪除「best published」這種全稱斷言**；報告沒有文獻調查章節） |
| 562 | `| CMIX | ~0.90 | ~1 KB/s | 200+ | CPU ensemble |` | 同 538 |
| 563 | `| NNCP v2 | ~0.94 | 3.25 KB/s | 1 | CPU neural |` | 同 539＋補列 |
| 566 | `beats CMIX by 26.2% (0.6646 vs ~0.90) … and 7x faster` | `beats CMIX v21 (1.170) by 43.2% … and 4.4x faster` |
| 587 | WSL 表 Windows 列 `59.5s / 0.9432 / 17.2 KB/s` | `29.8s / 0.936 / 34.3 KB/s`（＝帳本 `1m-chunk`，**才是同設定**；原 59.5s/17.2KB/s 與帳本 Qwen-1.5B `qwen15b-k2b8` 完全相同，疑為他模型數據） |
| 590 | `結論：**WSL 快 2.23 倍！**` | `結論：**WSL 快約 1.12 倍**（同 chunked 設定 29.8s→26.7s）` |
| 599 | `（不是 2.23x 因為…）` | `（不是 1.12x 因為…）` |
| 614 | `**快 2.23x** (1MB)` | `**快 1.12x** (1MB)` |
| 670 | `WSL 最快：~55 KB/s（2.23x 加速）→ 100MB = 31 min` | `WSL 最快：~47 KB/s（1.12x）→ 100MB ≈ 36 min` |
| §47 標題 | `**結果（Windows）：**` | 該表第一列（0.6712/28.2s/3.6KB/s/7.33GB）實為 **WSL** 執行；Windows 應為 `0.6706 / 15.4s / 6.5 KB/s / 7.28GB` |
| §30 | `**0.7 判決：算術死**` | 開頭加 `（**已於 §35 被推翻**：Qwen-1.5B 0.6996）` |
| §34 | `zllm 四個 preset 對應四條已驗證線（全 220/220）` | `（**未經端到端驗證**：該 CLI 路徑未產生任何 .zllm 檔，亦無帳本紀錄）` |

**`FINAL-REPORT.en.md`**：同構修正（行 353/354/441/445/449/473/482/497/553）。

**`README.md`／`README.zh-Hant.md`**

| 位置 | 現在 | 改成 |
|---|---|---|
| README.md:5 / zh-Hant:3 | `beats CMIX ~0.90`（1 個模型贏 200+ 26%，快 7 倍） | `beats CMIX v21 on enwik8 (1.170 bpb) by 43.2%`（速度：Qwen-3B 4.4x／SmolLM2 15.9x） |
| README.md:16 / zh-Hant:14 | `| CMIX … | ~0.90 | ~1 KB/s | — |` | `| CMIX v21 | 1.170 (enwik8) / 0.864 (enwik9) | ~1.6 KB/s | — |` |
| README.md:17 / zh-Hant:15 | `| NNCP v2 … | ~0.94 | 3.25 KB/s | — |` | `| NNCP v3.2 | 1.193 (enwik8) / 0.914 (enwik9) | ~4.1 KB/s | — |` |
| README.md:9 / zh-Hant:7 | 欄名 `Lossless check`／`無損檢查` | `抽樣往返 (220 點)`——**直到 B.3 的自包含 decoder 做完才可改回「無損」** |

## B.2 逐節驗證點數（取代「全 220/220」的籠統說法）

帳本實際值：

| 設定 | 帳本 id | 實際驗證 |
|---|---|---|
| K2048/B28672 | `qwen3b-k2b28` | 219/219 |
| K2048/B16384 | `qwen3b-k2b16` | 219/219 |
| K2048/B8192 | `qwen3b-k2b8` | 219/219 |
| K2048/B4096 | `qwen3b-k2b4` | 218/218 |
| K1024/B4096+gate | `qwen3b-k1b4` | **218/220（2 fail）** |
| K1024/B2048+gate | `qwen3b-k1024b2` | **218/220** |
| K512/B4096+gate | `qwen3b-k512b4` | **216/220（4 fail）** |
| K1024/B4096 1MB | `qwen3b-1mb-k1b4` | 220/220 |
| K2048/B4096 1MB | `qwen3b-1mb-k2b4` | 220/220 |
| K4096/B4096+gate | `qwen3b-k4b4` | 220/220 |
| B2048 K1024 | `qwen3b-b2048-k1024` | 218/220 |
| B2048 K512 | `qwen3b-b2048-k512` | **216/220** |

**寫法建議**：把「（全 220/220）」改成「（驗證點數見表；部分設定 216–219/220）」，並在 §7 但書補一句「Qwen 線部分設定的抽樣往返有 2–4 點失敗，未逐一追查」。

## B.3 c67a82a 新結果的正確寫法（**這點很重要，別寫錯**）

**已證明的（可以寫）**
- 真實 bitstream 存在，`11994` bytes 檔案：`ceil(95757/8) + 24 (header) = 11970 + 24 = 11994` ✓ 算術完全吻合。
- **檔案級 bpb = 0.9370**（`11994×8/102400`），不是 0.9351——0.9351 只算 payload，漏了 24 bytes 標頭（0.2%）。跟 Nacrith 的 0.9389 比要用 **0.9370**。
- 全檔 30791/30791 token 往返 + SHA-256 相符——**在 coder 層級**是完整的。
- 抓到並修掉 bitpack 尾端位移 bug（一行 `ljust`）：`ceil` 補齊後 `bytes(...)` 才與 `f"{b:08b}"` 對稱。這是實質貢獻。

**尚未證明的（不能寫）**
- `tools/real_compressor.py` 的 decoder **直接取用 encoder 算好的 `logits`**（L70 用 `ids` 跑前向 → L270 在 decode 迴圈裡 `p_lm = logits[pos]`）。也就是說：**只有檔案的人無法解碼**——真實 decoder 必須從「已解出的 token」重算機率。
- 因此現在的結論上限是：「**編碼器在給定基準機率下，產生的位元流可被同一組機率完整還原**」。這比 220 點抽樣強很多，但還不是「自包含的壓縮檔」。
- 建議用詞：`coder-level full-file roundtrip (ground-truth LM probabilities)`，不要寫 `verified lossless`。

**已解鎖的新數字（可安全入帳）**
- 100KB：真實檔 11994 B / **0.9370 bpb（檔案級）** / 0.9351（payload 級）
- 對照 v13 估算 0.9276 → 真實 0.9351 ＝ **+75e-4（+0.81%）**：這是「估算 vs 真實」的第一次量化落差，正是以前缺的那個數字。
