# Run log: real-bitstream experiments (versioned copy)

`logs/` is in `.gitignore`, so the per-run JSON written by the compression
scripts never entered version control. This file is the durable record: it
transcribes those results, says what each one does and does not prove, and notes
where the artifacts live. Numbers are quoted from the run output as recorded in
`data/sota_loop.json`; percentages marked *(derived)* are simple arithmetic on
those numbers.

## Why this file exists

Three claims ("100KB roundtrip 100%", "self-contained", "0.9351 bpb") were being
cited from run logs that the repository cannot see, and one of them
(`self-contained`) was wrong for the 100KB run: the decoder in
`tools/real_compressor.py` recomputes logits from the **source** token chunk
(`ctx_chunk = ids[ci:ci + chunk_len + 1]`), so only the n-gram half was mirrored
from decoded tokens. Labeling corrected in the ledger
(`ledger-wording-correction`, 2026-09-16).

## Runs

| # | id (ledger) | what ran | result | verdict |
|---|---|---|---|---|
| 1 | `real-bitstream-100kb` | SmolLM2-135M, enwik8 @50MB, 100KB, per-chunk forward, one stream, 32-bit AC | 95757 bits, **11970 B** file, 30790/30791 tokens (99.997%) | superseded |
| 2 | `bitstream-fix-100` | same + `ljust` packing fix | 30791/30791 (**100%**), SHA-256 token match, 95757 bits, **11994 B** file | coder-level ✓ |
| 3 | `self-contained-decoder-2kb` | same config, per-token KV-cache forward on **both** sides, 2KB | 554/554 (100%), logits diff `0.00e+00`, SHA-256 match | true self-contained ✓ (2KB only) |
| 4 | `bitstream-100kb-self-contained` | run 2 re-run / re-labelled | 30791/30791, SHA-256 match, 95757 bits, 11994 B | coder-level only |
| 5 | `seg8kb-sweep-fd2935b` | `tools/seg_token_compressor.py` (format v2), SmolLM2-135M, enwik8 @50MB, 8KB, K = 1/4/8/16/32 | all roundtrip=True; bpb_payload 1.1262 → 1.4741, escapes 2.20% → 4.12%, encode 157.8 s → 3.9 s | speed OK, ratio open |
| 6 | `seg-vs-v13-8kb-k1` | same slice, F1/F2 codec (K=1) against v13 counted at TOP_K=1024 / PF=2048 / tc3.0 | codec 1.1245 bpb, 56 escapes, decodable `.zllm`; v13 1.1323 bpb, 56 escapes, counted only | ratio parity reached; 64-bit gap unexplained |
| 7 | `seg-vs-v13-100kb-k1` | same slice (100 KB, offset 50 MB), codec K=1 `--overlap 0` vs v13 counted at TOP_K=1024 / PF=2048 / tc3.0 / OVERLAP=4096 | codec **94400 bits** / 0.9219 bpb / 383 escapes / roundtrip ✓; v13 **93588 bits** / 0.9139 bpb / 377 escapes / counted only | **812-bit gap = context deficit, fixable** |

## What run 2/4 actually prove — and don't

**Proven.** The bitstream, the 32-bit coder, and the n-gram (bigram/trigram)
state are mutually consistent at full file length: 30791/30791 tokens recover
exactly, token-level SHA-256 matches. The container is byte-exact through
`write → disk → read` after the padding fix.

**Not proven.** That the file can be decoded without the source text. The
decoder calls the model on `ids[ci : ci + chunk_len + 1]` — the *original*
tokens — so `chunk_logits` on the encode side and the decode side are the same
call with the same input. "Encoder and decoder logits identical" is therefore
vacuous in runs 2 and 4, not evidence.

**Proven by run 3 only.** Encoder and decoder each derive logits from **their
own** token stream (the decoder uses decoded tokens), with per-token KV-cache
forwards, and still get 554/554. That is the real self-contained test — but it
runs at 2KB and the script was session-local, so it is not reproducible from this
repository, and at ~30 min per 100KB (per the run note) it does not scale:
100MB would be ≈ 500 h ≈ 21 days.

## Derived numbers worth keeping

| quantity | value | how |
|---|---|---|
| payload bpb (runs 2/4) | 0.9351 | 95757 / (100 × 1024 × 8) |
| **file-level bpb** (runs 2/4) | **0.9370** | 11994 × 8 / 102400 |
| header cost | 24 B = 0.20% of payload *(derived)* | 11994 − ceil(95757/8) |
| v13 count estimate → real bitstream | 0.9276 → 0.9351 = **+75e-4 (+0.81%)** *(derived)* | first measured gap between the old "counted" bpb and a real stream |
| Nacrith, same 135M family, 100MB | 0.9389 | arXiv 2602.19626 — compare against the **file-level** 0.9370, not 0.9351 |
| segmentation seeds | K × ~16 bits | one uniform(V) symbol per segment, paid in-stream |

## The 1-token mismatch (run 1)

Cause was **not** arithmetic coding. `finish()` implements the standard
Witten–Neal–Cleary termination (`pending += 1`, emit one bit; the decoder pads
with zeros past EOF), and the same scheme yields 798/798 exact chunks in
`ensemble/verify_enwik8_full.py`. The corrupting step was packing:

```python
bytes(int(bitstr[i:i + 8], 2) for i in range(0, len(bitstr), 8))
```

For a final chunk of k < 8 bits these bits land in the **low** k bits of the
last byte, while reading back (`f"{b:08b}"` per byte, then truncate) expects them
high-first — so the tail comes back shifted. `"1010101"` → `0b01010101` →
`"0101010"`. The bug is invisible when the bit count is a multiple of 8; 95757
is 7 (mod 8), which is why one token at the tail was wrong. Fix: right-pad the
final chunk (`ljust(8, "0")`) before packing. Regression test:
`tools/audit_ac32_packing.py` (no torch needed).

## The 8KB K-sweep (run 5) — what it establishes

`K` here is the number of independent segments, i.e. how many streams are coded
in lockstep; it is not the alphabet size (`top_k` = 1024, `prefilter` = 2048,
matching v13's small-slice control).

| K | steps | per-step | enc s | dec s | bpb_payload | bpb_file | escapes | esc % |
|---|---|---|---|---|---|---|---|---|
| 1 | 2594 | 60.8 ms | 157.8 | 157.4 | 1.1262 | 1.4395 | 57 | 2.20% |
| 4 | 648 | 142.4 ms | 19.6 | 17.9 | 1.1809 | 1.4941 | 66 | 2.54% |
| 8 | 324 | 494.6 ms | 12.9 | 11.6 | 1.2334 | 1.5469 | 73 | 2.81% |
| 16 | 162 | 743.6 ms | 6.1 | 5.7 | 1.3287 | 1.6436 | 84 | 3.24% |
| 32 | — | — | 3.9 | 3.4 | 1.4741 | 1.7891 | 107 | 4.12% |

**Proven.** The batched forward (`[K,1]` in one call, single cache) works and
`K` trades ratio for wall-clock at ~T/K steps: 157.8 s → 3.9 s from K = 1 to
K = 32, a 40× swing. Every row roundtripped.

**Not proven / caveats.**

* `K` is *not* a speed-only knob: each segment sees only its own tokens, so
  larger K means shorter per-segment context and a worse model. The bpb column
  is that cost (1.1262 → 1.4741), not noise. The pre-batching "K = 4 is the
  sweet spot" reading was an artefact of per-row forwards and no longer holds.
* 1.1262 bpb here is **not** comparable to v13's 0.9276 (100KB) or 0.9003
  (100KB, K/PF = 8192): different slice length, and the cache needs data before
  it pays. The decisive control is v13 on the same 8KB slice:
  `ENWIK8_KB=8 TOP_K=1024 OVERLAP=0 PREFILTER=2048 python -u ensemble/bpe_ensemble_v13.py`.
* Escapes are the ratio story: 2.20% at K = 1 versus 1.25% for v13 at
  100KB/K = 1024, and 0.09% at K = 8192. The codec's escape price is
  `log2(1/esc_mass)` + uniform over `V - top_k` ≈ 15.5 bits each, so 2.2% of
  positions paying that is where the gap lives. Cause identified by comparison
  with `nb_blend_row` in `ac32.py`, which the 0.9003/0.9139 numbers come from:
  the codec's candidate set was the LM top-K only, so tokens the cache had
  evidence for but the LM ranked outside top-K could never be coded directly.
  Fixed (F1: prefilter ∪ cache-row keys, then top-K by blended score; F2: v13's
  per-count weights `_sb`/`_st`). Not yet measured on the GPU.

## Run 6 in detail — and what it does *not* show

Numbers reported from the user's machine (the v13 side is a counted run: it
writes no decodable file, so it cannot be re-verified here; the codec side
roundtripped). Both bit counts are exact arithmetic-coder counts
(`nb_encode_count_32` in `ac32.py` is the coder's own accounting, not an
entropy estimate), and bpb = bits / 8192 on both sides.

| side | escapes | bpb | decodable |
|---|---|---|---|
| v13 (counted) | 56 | 1.1323 | ✗ (no artifact) |
| codec K=1 | 56 | 1.1245 | ✓ `.zllm` |

**Established.** The F1/F2 port put the codec's candidate set and weights on
v13's math: escape counts agree exactly (56 vs 56) on identical input.

**Not established, and previously mis-stated here.** (a) F1/F2's own effect at
this slice length is small, not decisive: the same codec measured 57 escapes
and 1.1262 bpb *before* the fix (run 5), i.e. −1 escape and −14 bits. (b) The
0.0078 bpb was called unexplained here; run 7 identifies it as the context
deficit (v13 feeds OVERLAP=4096 of history, the codec fed none), and the fp16 /
blend-gate candidates listed here were not needed to explain it.

**Why 8KB cannot answer the ratio question.** Both sides escape ~2.16% here
because the cache only has 2 595 tokens to learn from; escapes at this length
are dominated by "target has no cache evidence at all", which no distribution
change can fix. v13's 1.22% figures come from 100 KB (≈30 791 tokens, 12×
deeper cache). The like-for-like measurement is the 100 KB run, not this one.

## Run 7 — the 100 KB comparison, and where the 812 bits go

`data/seg_verify_1seg_100kb.json` (codec, roundtrip `tokens_identical=True`) vs
`data/smollm2_ensemble_v11_off50_..._k1024_ov4096_..._kb100.json` (v13, counted
only, no artifact). Both are exact coder counts, both divide by 102 400 bytes.

| side | bits | bpb | escapes |
|---|---|---|---|
| v13 counted | 93 588 | 0.9139 | 377 |
| codec K=1 | 94 400 | 0.9219 | 383 |

**The gap is 812 bits (0.0079 bpb), not 64.** 64 bits is the *8 KB* figure
(0.0078 bpb × 8192 B) carried over by mistake; the per-byte gap is about the
same, so the absolute gap scales with the file. That matters: a constant
per-position bias produces a gap proportional to length, which is what we see.

**Cause: the codec's tokens see LESS context than v13's, not a different kind of
context.** `BLOCK_TOKENS=8192` and the reference run used `OVERLAP=4096`, so
v13 feeds `_tail(4096) + _new(4096)` per block: every coded token sees 4097–8192
tokens of context (mean 6144.5). The codec's reset-every-8192 (`--overlap 0`)
restarts from an empty cache, so its tokens see 1–8192 (mean 4096.5, min 1).
812 bits over the 22 600 tokens past the first chunk is 0.036 bits/token, which
is the right size for losing ~2000 tokens of mean context. So the codec is the
deficient side here and the deficit is addressable.

**Correction to "each token sees the full 8192 context".** True of positions
inside the model's window, false of this loop: with `--overlap 0` the token at
chunk offset j sees j+1 tokens. Note also that the reset is a *correctness* fix,
not a precaution: the previous sliding-trim code looked like a sliding window but
the model derives each position from the cache length, so once the cache filled,
every new token got position 8192 while the stored entries kept their own --
recent tokens collapsed onto one position and the relative geometry was gone.
That never fired at 8 KB (2 595 tokens < 8192), which is why it survived the 8 KB
sweep. A true sliding window needs the stored K re-based (RoPE-rotated) per step;
resetting is the honest version.

**Fix implemented, not yet measured.** `tools/seg_token_compressor.py` now takes
`--overlap O`: a chunk is `O` re-fed history + `kv_window - O` new tokens, and
the history is re-fed as ONE batched prefill. `--overlap 4096` reproduces v13's
context distribution exactly (min 4097, mean 6144.5); `--overlap 7680` gives
min 7681, mean 7936.5. The prefill is batched, so it costs one forward per
`kv_window - O` tokens rather than one per history token -- under 1% of the
per-token loop, i.e. this is a ratio knob that is nearly free in wall clock.
`tools/test_chunk_schedule.py` pins the position budget (max position ≤
`kv_window - 1`), the boundary/prefill arithmetic, and the rolling-buffer index
mapping (which caught a real off-by-one that included the current token in the
prefill). Default remains `--overlap 0`, so the measured runs above stay
reproducible.

## Next step and its scaling limit

`tools/seg_token_compressor.py` (format v2) is the self-contained design: the
decoder holds only the file + model; token 0 of each of K segments is coded
uniform(V) in-stream, later tokens use per-segment KV caches built from decoded
tokens, and K segments advance in lockstep as one batch.

Scaling note for the 100MB run: SmolLM2-135M KV cache is ≈ 23 KB/token and each
segment holds its own, so total cache ≈ 23 KB × tokens-per-segment. 100MB
(≈27 M tokens) needs segment lengths of ~200–300 K tokens, i.e. **K ≳ 100** —
the K = 1/K = 4 runs are reference points for correctness and batch amortisation,
not the shape of the full-file run.
