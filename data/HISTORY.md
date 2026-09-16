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
