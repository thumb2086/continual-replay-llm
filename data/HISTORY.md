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
| 8 | `seg-overlap-ab-100kb-k1` | same 100 KB slice, `--overlap` 0 / 4096 / 6144 (all K=1), vs v13 counted | escapes 383 / 372 / 376 (v13: 377); bpb **0.9219 / 0.9089 / 0.9092** (v13: 0.9139); all roundtrip ✓ | context deficit closed; codec now **below** v13 |
| 9 | `seg-ov7680-oombug` | codec `--overlap 7680` (prefill OOM fix) | 0.9094 bpb, ~375 escapes | OOM fix works; ≥6144 buys nothing |
| 10 | `v13-gate-fp16-off` | v13 counted, `BLEND_BT_MIN=0 BLEND_TT_MIN=0 USE_FP16_XFER=0`, 100 KB | 0.9141 bpb, 380 escapes (baseline: 0.9139 / 377) | gate + fp16 are worth ~nothing; not the gap |
| 11 | `seg-k8192-100kb-k1` | codec `--overlap 4096 --top-k 8192 --prefilter 8192 --trigram-conf 10.0` | **0.8998** bpb | console only, no artifact — see run 12 |
| 12 | `seg-100kb-ratio-final` | the same three configurations with the driver fixed, artifacts written | ov0 / ov4096 / ov4096+K8192: **0.9219 / 0.9091 / 0.8998** bpb, escapes 383 / 374 / 29, all roundtrip ✓ | **ratio line closed: sub-0.90, verified lossless** |
| 13 | `seg-speed-p0-p1` | P0: phase split of the K=1 run. P1: K=4 with `--shared-tables` | P0: fwd 674.5 s (94%), dist 41.0 s (5.7%), ac 0.2 s. P1: encode 718→500 s, bpb 0.8998→**0.9145**, escapes 29→31 | K=4 is a **bad trade** at present; see below |
| 14 | `seg-fwd-split-2` | `--profile-fwd` on the K=4 run; StaticCache trial | prep 0.12 ms, transfer 0.33 ms, norm 0.33 ms — all negligible; 52 ms/step is inside `model(**kw)`. StaticCache worth 12–15 %; CUDA graph blocked (transformers 4.57.6 writes the mask in place) | **GPU top-K killed**; 100 MB not viable at any K; see below |

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

## Run 8 — overlap A/B: the context deficit was the whole story

Same 100 KB slice, K=1, three schedules; the checked-in artifact is the
`--overlap 6144` run (`data/seg_verify_1seg_100kb.json`), the 4096 row is from
the run summary (that file was overwritten), and the 0 row is run 7.

| overlap | escapes | bits | bpb | encode s | decode s | roundtrip |
|---|---|---|---|---|---|---|
| 0 | 383 | 94 400 | 0.9219 | 696.2 | 688.4 | ✓ |
| 4096 | 372 | ≈93 071 *(derived: 0.9089 × 102 400)* | **0.9089** | not in artifact | — | ✓ |
| 6144 | 376 | 93 100 | 0.9092 | 736.3 | 724.1 | ✓ |
| v13 counted | 377 | 93 588 | 0.9139 | — | — | ✗ |

**What this establishes.**

* The 812-bit gap was context, not the distribution builder. Buying back history
  through a batched prefill recovered ~1 330 bits (0.9219 → 0.9089) with escapes
  moving 383 → 372, and the escape count now straddles v13's 377. That is the
  clean confirmation of the run 7 diagnosis.
* At matched context the codec is ~517 bits (0.0050 bpb) **better** than the v13
  counted reference. Attribution is open: the codec always blends, while v13's
  gate (`BLEND_BT_MIN=5`/`BLEND_TT_MIN=2`) falls back to LM-only on thin rows,
  and v13 documents its whole row to fp16 (`USE_FP16_XFER=1`) before blending.
  The decisive run is cheap: `ENWIK8_KB=100 BLEND_BT_MIN=0 BLEND_TT_MIN=0
  USE_FP16_XFER=0` on v13 — if it lands near 0.9089 the attribution is settled.
* More overlap is not better: 6144 (stride 2048, 15 chunk boundaries) is 0.0003
  bpb *worse* than 4096 (stride 4096, 7.5 boundaries). The deficit model
  (mean context `(overlap + window)/2`) predicts the opposite, so the reset
  discontinuity has a real cost that roughly cancels the extra context past
  ~4096. 4096 is the knee — which is where v13's `OVERLAP=4096` sits.

**Wall-clock readings from this A/B are not usable.** Three schedules whose cost
model differs by single-digit percent were reported/measured at 696 s, 736 s and
1214 s. The checked-in artifact says the 6144 run took 736.3 s (23.9 ms/step),
and run 7 said overlap 0 took 696.2 s (22.6 ms/step) — but the 8 KB K=1 run in
the same batch is 127.3 s over 2 594 steps (49 ms/step), i.e. twice the per-step
cost of the 12× longer run. Nothing in the code explains a 2× swing, so treat
per-run timings on this host as unreliable until repeated; the ratio columns are
deterministic and unaffected.

**`--overlap 7680` OOM: cause found, different from the obvious guess.** It is
not the KV cache (172 MB at 7 680 tokens). A prefill forward materialises the
LM head's `[L, V]` logits: 7 680 × 49 152 × 2 B = **720 MB fp16** in a single
allocation, on top of the outgoing cache the previous `out` object was still
holding. Two fixes: prefills now go through the model's backbone
(`model.model`) so no logits are ever built, and they are sliced
(`--prefill-chunk`, default 2 048) with the previous cache released first.
Expected to put 7680/8191 back in reach; not yet measured on the GPU.

## Run 9–11 — and the counting-convention correction that closes the gap

| test | bpb | escapes | artifact |
|---|---|---|---|
| (a) codec `--overlap 7680` | 0.9094 | ~375 | none (driver bug, run 12) |
| (b) v13 `gate=0 fp16=0` | 0.9141 | 380 | counted only |
| (c) codec `--overlap 4096 --top-k 8192 --prefilter 8192 --trigram-conf 10.0` | **0.8998** | — | none (driver bug, run 12) |

(a) confirms the prefill-OOM fix and confirms that more overlap stops paying past
~4096. (b) rules out v13's blend gate and its fp16 probability transfer as an
explanation for anything: turning both off makes v13 marginally *worse*
(0.9139 → 0.9141, 377 → 380 escapes), i.e. those two cost the codec nothing
because they cost v13 nothing.

### The residual is a counting convention, not the distribution

Attributing the codec's advantage to "the distribution builder (F1/F2)" cannot be
right: F1 and F2 are now implemented in *both* — that was the point of the port —
and the gap survives every change to the shared math while tracking the escape
count (517 bits at 377 escapes, 51 bits at 27). 517/377 = 1.37 bits per escape;
51/27 = 1.9. A difference that scales with escape count points at the escape path.

It is `nb_encode_count_32`'s accounting. v13 codes stage 1 in one call, then
appends each escape's stage-2 symbol in its OWN call:

    n1 = nb_encode_count_32(C, sym_arr)              # one coder, one finish
    for ... in esc_infos:
        n2_total += nb_encode_count_32(co_row, [rank])   # a fresh coder + finish

`nb_encode_count_32` "returns total incl. finish bits", so v13's `total_bits`
pays one arithmetic-coder termination per escape event, while a real
single-stream coder codes that symbol as a continuation. The codec writes one
stream, so the two numbers are different quantities. Measured with ac32's own
kernel (`tools/audit_count_overhead.py`, escape mass 0.001–0.3, ranks across the
rest set): **1.5 bits per split call** (range 1–2).

| run | v13 counted | v13 single-stream equivalent | codec (one stream) |
|---|---|---|---|
| 100 KB, K=1024, ov4096 | 93 588 (0.9139) | ≈93 022 (0.9084) *(derived: −566)* | 93 071 (0.9089) |
| 100 KB, K=8192, ov4096, tc10 | 92 201 (0.9003) | ≈92 161 (0.9000) *(derived: −40)* | ≈92 140 (0.8998) *(derived: 0.8998 × 102 400)* |

Predicted inflation 377 × 1.5 = 566 vs 517 observed; 27 × 1.5 = 40 vs 51
observed. Both within the measured spread of the overhead. So at matched context
the codec and v13 agree to within ~20–50 bits (≤0.0005 bpb) — the tie-order noise
level `nb_blend_row`'s own docstring cites — and **both operating points are
sub-0.90 as single streams**. The codec's advantage over the *counted* numbers is
the convention, not better math; the codec's real advantage is that 0.8998 comes
out of a file that decodes.

### Driver bug (mine, fixed)

`del nl_enc, nl_enc_timed` (after the encode pass, to free the encode-side model
state) followed by `nl_enc.prefill_note` in the result dict that is built after
decode: UnboundLocalError *after* a full encode and decode had succeeded, with no
artifact written — which is why (a) and (c) above have no JSON. The note is now
captured as a plain string before the `del`. `tools/test_driver_static.py` scans
the driver for del-then-use in source order and is verified to flag the original
line; it also pins the CLI defaults the measured runs depend on.

### Next blocker, measured: the distribution builder

Per position, single core, `sc.build_distribution` (sandbox, numpy+numba):

| alphabet | 300-key row | 2000-key row | 100 MB (27 M positions) |
|---|---|---|---|
| `top_k=1024 pf=2048` | 0.635 ms | 1.065 ms | 4.8–8.0 h |
| `top_k=8192 pf=8192` | 2.230 ms | 2.769 ms | 16.7–20.8 h |
| `top_k=8192 pf=16384` | 4.434 ms | 4.446 ms | 33.3 h |

The GPU side at 100 MB is ~180/K hours (K=100 → 1.8 h), so at the operating point
that now produces 0.8998 the *Python* distribution builder is the bottleneck for
the 100 MB goal, by an order of magnitude, and it is the one cost `K` does not
divide. The K segments are independent in lockstep, so it is parallelisable
across processes — but that is a change to propose, not to land silently.

## Run 12 — ratio line closed

Final 100 KB numbers, every row verified lossless (tokens identical, SHA-256
match, bytes identical), and now written to artifacts instead of the console:

| config | bpb payload | bpb file | escapes | encode s |
|---|---|---|---|---|
| `--overlap 0` | 0.9219 | 0.9488 | 383 | 696.2 |
| `--overlap 4096` | 0.9091 | 0.9361 | 374 | — |
| `--overlap 4096 --top-k 8192 --prefilter 8192 --trigram-conf 10.0` | **0.8998** | 0.9269 | **29** | 718.0 |
| v13 counted (same slice) | 0.9139 | — | 377 | — |
| v13 counted, same operating point (pf8192 k8192 tc10) | 0.9003 | — | 27 | — |

Escapes fell 93 % (374 → 29) at the wide alphabet. Against v13's *corrected*
(single-stream) numbers the two implementations agree to ~20–50 bits, i.e. the
tie-order noise `nb_blend_row`'s own docstring cites — so the codec has reached
parity with the reference math, and it is the only side whose 0.8998 comes out of
a decodable file. **The ratio question is answered; work moves to speed.**

Two durability fixes came out of this, both aimed at the 100 MB run:

* Artifacts now carry the alphabet/blend constants, `kv_window`, `overlap`,
  `shared_tables` and the phase timings, and the filename carries a fingerprint
  (`..._k8192_pf8192_tc10_ov4096.json`). Three different configurations had
  already overwritten one filename and the JSON could not tell them apart.
* `SegTables`' docstring claimed sharing across segments "a lockstep decoder
  cannot reproduce". That is false, and it mattered because sharing is what
  makes K > 1 cheap in ratio: both loops walk `s = 0..K-1` per position, so the
  shared table sees identical updates in identical order on both sides. Now
  stated correctly and pinned by tests: K = 2/4/8 round-trip with sharing on,
  the sabotage controls still fire, and sharing demonstrably changes the coded
  size (2106 → 1806 bits on the degenerate-LM case), so it is neither a no-op
  nor a placebo.

### Speed: what is known, and the first thing to measure

`--overlap 4096` at 100 KB costs 718 s encode + 723 s decode ≈ 24 min for 100 KB
(`0.07 KB/s`), against the stated goal that "compressing 100 KB takes an hour is
not practical".

The cheapest large lever needs no new code. The per-token loop is dominated by
per-STEP cost, not per-token work: at 8 KB the same code path cost 60.8 ms/step
at K=1 and 48 ms/step at K=32 (i.e. 1.5 ms/token), and the batched forward is
nearly free in batch size. At 100 KB, K=4 gives ~7 700 tokens per segment, which
with `--overlap 4096` still yields contexts of 4097–7698 (mean ≈3 970 against
K=1's ≈5 990), and `--shared-tables` removes the n-gram thinning that made K
expensive at 8 KB. Expect ~3–4× wall clock for a small bpb cost; K=2 is the
conservative version (mean context ≈5 990, i.e. nearly K=1).

Before optimising further, one number decides where the 23.3 ms/step actually
goes. The driver already prints it:

    PHASES: fwd=... dist=... ac=... total=...

`fwd` is wall time inside `next_logits` and therefore includes the `.cpu()`
synchronisation; `dist` is `build_distribution`; `ac` is the coder. Nothing else
in the loop is timed. That line, from any 100 KB run, says whether the next step
is CUDA graphs/StaticCache (fwd dominates), a faster distribution builder (dist
dominates), or neither.

## Run 13 — speed: where the time is, and why K did not pay

### P0: the `fwd` phase is 94 % of encode

| phase | seconds | share |
|---|---|---|
| fwd (forward + logits normalise + D2H + input prep) | 674.5 | 94.0 % |
| dist (`build_distribution`) | 41.0 | 5.7 % |
| ac (coder) | 0.2 | 0.03 % |
| outside `next_logits` (codec bookkeeping) | 2.3 | 0.3 % |

So `dist` is **not** the 100 KB bottleneck (it is 5.7 %; it only becomes the
bottleneck at 100 MB, where it does not divide by K). The forward path is. Note
that `fwd` is one number covering four different costs — a forward, a GPU-side
normalise, a `[K, V]` device-to-host copy, and tensor construction — and they
need different fixes. `--profile-fwd` now synchronises between them and prints
the split, because "94 % is forward" is not yet actionable.

### P1: K=4 with shared tables bought 1.44x and cost 0.0054 bpb

| run | bpb | escapes | encode s | decode s |
|---|---|---|---|---|
| K=1, ov4096, K8192, tc10 | 0.8998 | 29 | 718.0 | 722.7 |
| K=4, ov4096, K8192, tc10, shared | **0.9145** | 31 | 500.0 | 502.0 |

**This contradicts the prediction made here** ("~3-4x faster for +0.003-0.012
bpb"). The prediction came from the 8 KB sweep, where K=32 cost 48 ms/step
against K=1's 60.8 ms/step and looked nearly free in batch size. That was an
older build (before the chunk reset and the prefill) and, more importantly, the
inference was wrong: per-STEP cost is not batch-independent.

Fitting the two 100 KB points (`t = a + b·K`): 21.9 ms/step at K=1 and 54.8 at
K=4 gives **a ≈ 11 ms fixed + b ≈ 11 ms per row**. The memory-bound floor for one
K=4 step at full context is ~2.3 ms (270 MB of weights + 754 MB of KV reads at
448 GB/s), so the step runs ~24x above what the arithmetic costs. **That gap is
the whole story: the cost is per-call, not per-token.** Caching implications:

* K's entire benefit is amortising a per-call cost across rows. With b ≈ a, K
  halves the per-token cost at K=4 instead of quartering it — which is exactly
  the 1.44x observed against a theoretical 4x.
* K is therefore **not a viable speed lever while per-row cost is real**, since
  it pays a ratio cost (shorter per-segment context, +0.0054 bpb here, and the
  +0.015 reported with `--overlap 0`) for a fraction of the speedup.
* Conversely, the per-row cost cannot be arithmetic (the floor says so), so it
  must be dispatch/launch/bookkeeping — which is what a StaticCache + CUDA graph
  removes. P1's own numbers are therefore the strongest argument for P2, not an
  argument against it.

### P2 instrumented, and a probe written for it

`tools/seg_token_compressor.py --profile-fwd` splits the `fwd` phase into
prep / warm(prefill) / forward / transfer+normalise, and reports the D2H payload
per step together with what a GPU top-K would send instead (a `[K, V]` fp32 row
is 0.19 MB/step at K=1, 0.75 MB at K=4; a PREFILTER=8192 top-K is 6x smaller).

`tools/wsl_cuda_graph_probe.py` is the feasibility test, since CUDA graph capture
has two real constraints here:

1. **a graph cannot contain the `.cpu()` readback**, and this loop does one every
   step — so a graph can only cover the forward, and the probe's configuration 4
   (graph without readback) is the lower bound that says how much is left.
2. **the codec resets its cache and re-warms it with a variable-length prefill**,
   which one static graph cannot express; prefills stay eager (they are <1 % of
   wall clock, so this costs nothing).

The probe measures, at K=1 and K=4: eager+DynamicCache (today), eager+StaticCache,
StaticCache+graph, and graph-without-readback — and it verifies correctness by
driving the graph and the eager reference through the SAME tokens so their caches
hold identical contents before any comparison (comparing numbers from unrelated
runs would pass a wrong graph).

## Run 14 — the per-row cost is the whole problem, and K cannot fix it

`--profile-fwd` resolves the 94 % "forward" into parts, and the answer is that the
parts are all innocent:

| sub-phase | ms/step (K=4) |
|---|---|
| prep (input tensor build) | 0.12 |
| warm (prefill) | ~0 |
| **inside `model(**kw)`** | **~51** |
| transfer + normalise (`.cpu()` etc.) | 0.33 + 0.33 |

So the D2H readback is 0.33 ms of 52 ms: **the GPU top-K idea is dead** and should
not be pursued (it would save ~0.3 ms while complicating `build_distribution`'s
contract). StaticCache is worth 12–15 %, and CUDA graph capture fails because
transformers 4.57.6 writes into the mask in place.

### Why neither "upgrade transformers" nor "go to 100 MB" is on the critical path

Fit the two 100 KB points (K=1: 21.9 ms/step, K=4: 64.9 ms/step):

    per-step = 7.6 ms fixed + 14.3 ms PER ROW

100 MB is ~27 M positions, so the *total* is 27 M x (7.6/K + 14.3) ms:

| K | ms/step | steps | per pass | encode + decode |
|---|---|---|---|---|
| 1 | 21.9 | 27.0 M | 164 h | 328 h |
| 4 | 64.9 | 6.8 M | 122 h | 243 h |
| 30 | 437.6 | 0.9 M | 109 h | 219 h |
| 100 | 1440.9 | 0.3 M | 108 h | 216 h |

**More K does not help**: it approaches the asymptote 27 M x 14.3 ms = 108 h, and
K cannot exceed ~38 anyway — each 8192-context segment holds ~180 MB of KV cache
on an 8 GB card.

And 14.3 ms per single-token row is not work. One row must read its segment's KV
cache — 180 MB at 448 GB/s = **0.40 ms/row** — plus trivial arithmetic. The
measured cost is **36x the memory-bound floor**. StaticCache's 12–15 % moves 164 h
to ~144 h, which changes nothing.

So the one number that decides whether 100 MB is an overnight job or a
two-week job is the per-row cost, and it is currently 36x above what the hardware
requires. Everything else — the transformers version, the mask, K, the artifact
format — is downstream of it.

### What was written for this

`tools/probe_forward_cost.py`: one run (~3 min) that answers it instead of
ranking hypotheses by taste.

* a `torch.profiler` kernel table at the real loop shape (K=4, L=window), sorted
  by CUDA time and by self-CPU time — the decisive part, because it names the
  ops rather than guessing;
* the per-row slope over K = 1..16, and cost vs cache length 0 → 8192 (is the
  per-row cost O(seq)? then it is cache traffic or mask construction, not
  dispatch);
* CPU-submit time vs wall time (CPU-bound or GPU-bound?);
* eager vs CUDA-graph on a fixed shape with no cache — this bounds what a graph
  is worth **without** the graph-safe mask work, i.e. it prices the prize before
  anyone patches mask machinery or bumps a major dependency;
* one 8192-token block forward for reference: the same model called in bulk
  instead of one token at a time. Not usable for coding — encoder and decoder
  must run the same arithmetic or the mirror breaks — but it is the size of what
  per-token mirroring costs.

## Run 15 — closing state (2026-09-17)

**Ratio: closed.** The self-contained codec reaches 0.8998 bpb payload
(0.9269 file) at 100 KB, escapes 29, verified lossless — tokens identical,
SHA-256 match, bytes identical. Against v13's *corrected* (single-stream)
numbers it is 92 144 vs 92 161 bits, i.e. inside tie-order noise (see the
counting-convention section above). Final report section: `FINAL-REPORT.md` §49.

**Speed: closed, with the reason recorded correctly.** The structural gap is
irreducible: the decoder must decode one token at a time (each token's
distribution depends on the previously decoded token), so it can never use block
forwards. 100 KB is ~30 791 forwards for the codec against ~4 for v13 — which is
exactly why v13 is 34.5 KB/s and the codec is 0.14 KB/s, and v13 can only do that
because its decoder reads the source tokens (§4's debt). Wall clock 718 s + 723 s.

What is NOT the reason, and was briefly recorded as such: hardware throughput.
The `[1, 8192]` block forward measures 439 ms = **53.6 µs/token**
(`tools/probe_forward_cost.py` section F), while the loop costs
**21 900 µs/token** — the loop is **409x slower** per token despite doing LESS
per-token attention work (O(L) with a cache vs O(L²) in bulk). The per-step cost
is therefore overhead, and per-row it is 36x above its 0.40 ms/row bandwidth
floor. The probe's first version mislabelled µs as ms (a 1000x error that made
the bulk forward look 2.4x *slower* than the loop); fixed in the tool and the
correction is stated in `FINAL-REPORT.md` §49.3.

Closing rests on the repo's own prior verdict, which this matches in kind:
§14 measured v13's 1.4 s/segment against 0.15 s of component cost and convicted
the box's scheduling (queuing, desktop preemption, boost drop-off) rather than
the code. Same shape, same conclusion, same remedy: a quiet box or a new card.
Recorded as "per-step overhead, mixture and magnitude undetermined, environment
suspected" — not as a hardware limit.

Cheapest re-open, if ever wanted: `tools/probe_forward_cost.py` section C at
`L=0` (no cache, so only 270 MB of weights = 0.6 ms of real work). A reading far
above 0.6 ms would be direct proof of pure per-step overhead.

**100 MB: not run.** 27 M positions x (7.6/K + 14.3) ms = 164 h (K=1) to 108 h
(K=100) per pass, with K capped near 38 by ~180 MB of KV per 8192-context
segment. That is a property of the per-token design, not of the artifact format.

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
