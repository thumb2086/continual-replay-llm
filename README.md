# Continual Replay LLM + Practical Neural Compression

Replay-first continual learning for small streaming language models — applied to **practical neural text compression**: SmolLM2-135M ensemble at **0.9139 bits/byte** on enwik8 (beats the 0.9389 SOTA line by 2.7% on the same slice class), with every number measured and every failure kept.

## Headline numbers (all measured, 2026-09-13)

| System | bpb ↓ | Speed | Lossless check |
|---|---|---|---|
| **Ours v11 (this repo)** | **0.9139** | ~2.3 KB/s busy (~3.7 with v13＋proc＋pipeline) | 220/220 roundtrip |
| Nacrith (SOTA, same-slice H2H) | 1.2248 | 0.25 KB/s (their CPU build) | byte-exact ✓ |
| NNCP v2 (ref value) | ~0.94 | 3.25 KB/s | — |
| CMIX (literature) | ~0.9 | ~0.1–1 KB/s | — |
| PAQ8 (literature) | ~1.0–1.2 | ~1–5 KB/s | — |
| xz -9 (measured here) | 2.310 | ~4.6 MB/s | — |
| bzip2 -9 (measured here) | 2.244 | ~21 MB/s | — |
| gzip -9 (measured here) | 2.831 | ~26 MB/s | — |

Classical codecs are ~10⁶× faster at 2–3× worse ratio — different worlds (see `FINAL-REPORT.md` §8 in spirit). Within the neural (<1.0 bpb) world: best ratio here, second-best speed.

Honest scope: 0.9139 is a 100KB representative middle slice (enwik8 offset 50MB), not the full 100MB file; verify is 220 sampled positions, not full decode. Details + all caveats: [`FINAL-REPORT.md`](FINAL-REPORT.md) (hand-in report).

## Papers & reports (where is the paper?)

- [`FINAL-REPORT.md`](FINAL-REPORT.md) — **the paper**: trajectory 1.2541→0.9139, 3 publishable insights (smoothing tax, finish-bit tax, boundary-bug disclosure), Nacrith H2H, speed/memory profile, full falsification log, honest caveats, file map. Start here.
- [`compression-paper.md`](compression-paper.md) — earlier work: 24-condition char-level grid, big-data training, data mixing (historical).
- [`paper_sections_13_14_draft.md`](paper_sections_13_14_draft.md) — SmolLM2 transition notes (frozen/adapt, v1–v3 ensemble history).
- [`paper.md`](paper.md), [`continual-learning-report.md`](continual-learning-report.md) — continual-learning track (replay-first).

## Reproduce the headline (PowerShell, ~45 s, RTX 3060 Ti 8 GB)

```powershell
$env:BIGRAM_LAMBDA='0.99'; $env:ENWIK8_OFFSET_MB='50'; $env:BIGRAM_CONF='10'
$env:TRIGRAM_CONF='3'; $env:TOP_K='1024'; $env:OVERLAP='4096'; $env:FLOOR_FRAC='1e-6'
$env:USE_CACHE_S2='0'; $env:USE_FP16_XFER='1'; $env:PREFILTER='2048'
python -u bpe_ensemble_v11.py
# gate: bits/byte: 0.9139, Verified: 220 lossless, fails: 0
```

Needs: `pip install torch numpy transformers` + SmolLM2-135M weights (`MODEL_DIR` in `bpe_compress.py`) + enwik8 at `data/cloud/enwik8` (not included).

## Structure

```
bpe_ensemble_v11.py    Hand-in system (0.9139). v10 = equivalent predecessor
bpe_ensemble_v12.py    llama.cpp CUDA backend (falsified on busy box, kept)
bpe_ensemble_v13.py    v11 math + validated plumbing (proc pool, pipeline,
                       double-buffering, incremental freeze); KV-chaining and
                       ORT branches falsified, kept for the record
                       (USE_PROC_LOOP=1 + PIPELINE=1 → 0.9139, 26.9 s/100 KB)
bpe_ensemble_v[3-9].py Iteration trail (each version = one idea)
ac32.py                32-bit arithmetic coder + numba kernels (cached)
bpe_compress.py        SmolLM2 base plumbing (env-driven knobs)
sota_loop.py           Iteration driver (queue + ledger + stop rule)
data/sota_loop.json    The ledger: every idea's bpb/speed/verified
h2h_nacrith.py         Third-party H2H rerun (same slice, their code)
third_party/nacrith    Nacrith upstream (cloned, Apache-2.0; not committed)
third_party/smollm2-135m-bf16.gguf  Official BF16 (not committed, 270 MB)
test_shm_proc.py       Proc-loop plumbing test (standalone, CPU-only)
ort_export.py          ONNX export for the ORT branch (unrun)
verify_enwik8_full.py  Full-file verify tooling (see §7 caveats)
logs/                  All run logs (stdout/stderr per experiment)
data/*.json            Per-run result files (metrics only, no weights)
train_v3.py, compare_*.py, sweep_*.py  Continual-learning track (see paper.md)
```

Scripts run from this directory (`sys.path` assumes it); don't move them into subfolders without fixing imports. Experimental flags (`USE_PROC_LOOP`, `PIPELINE`, `USE_ORT`, `CHAIN`) default off — defaults are always the validated path.

## Continual-learning results (unchanged)

Held-out A→B→A→B long cycle (replay batch 8, EWC off):

| Phase | Topic A loss | Topic B loss |
|-------|-------------|-------------|
| After pretrain A | 4.1977 | 5.2536 |
| After online B1 | 4.1111 | 4.5846 |
| After online A2 | 4.1054 | 4.3827 |
| After online B2 | 4.0021 | 4.3102 |

Ablation: replay-only keeps Topic A 0.981x + improves B 21.96% (EWC-only 1.015x/11.64%, pure FT 1.102x/23.36%). See [paper.md](paper.md).

## Data

Training uses local podcast transcripts (private, not included) + local enwik8/TinyStories/Cosmopedia (not included). Result JSONs contain metrics only.

## Citation

```bibtex
@misc{continual-replay-llm-2026,
  title  = {Replay-First Continual Learning for Small Streaming Language Models},
  author = {thumb2086},
  year   = {2026},
}
@misc{smollm2-practical-ensemble-2026,
  title  = {A Practical SmolLM2-135M Ensemble at 0.9139 bpb on enwik8},
  author = {thumb2086},
  year   = {2026},
  note   = {FINAL-REPORT.md in this repo},
}
```

## License

MIT — see [LICENSE](LICENSE).
