# Architecture Review, Agent Team Plan & 30-Day Roadmap

> 2026-09-13 | 254 ledger entries | SOTA 0.8391* (Qwen) / 0.9003 (SmolLM2) / 34.5 KB/s chunked

---

## Step 1: Codebase Audit

### Core files

| File | Lines | Role | Issue |
|---|---|---|---|
| ensemble/bpe_ensemble_v13.py | 1923 | THE system: encode+loop+gather+chunk+sparse+gate | Monolith, no CLI |
| ac32.py | ~600 | 32-bit arithmetic coder + numba kernels | Shared core, safe to isolate |
| bpe_compress.py | 308 | Old standalone BPE encoder | Superseded |
| real_compression.py | 819 | Huffman baseline + neural arithmetic | Superseded |
| tools/plot*.py | 300+ | Charts | Standalone, clean |

### Infrastructure gaps

| Gap | Severity |
|---|---|
| No CLI entry point (20+ env vars required) | CRITICAL |
| No decode path (half a product) | CRITICAL |
| No pyproject.toml / setup.py | HIGH |
| No unit tests | HIGH |
| No .zllm container format | MEDIUM |
| Flat root import hell | MEDIUM |

### Current data flow

```
Input text -> BPE tokenizer -> 8K chunks -> LLM forward (fp16 flash SDPA)
-> GPU softmax -> topK/prefilter -> gather -> bigram/trigram blend
-> Stage-1: top-K + escape -> 32-bit arithmetic coded
-> Stage-2: uniform over V-K -> 32-bit arithmetic coded
-> Bitstream (COUNTED ONLY, no file I/O)
```

Decode path: ArithmeticDecoder exists but no reverse pipeline.

---

## Step 2: Agent Team

### Agent 1: Algorithm Research
- Survey PAQ8/LZMA+neural/ANS neural coders (2d)
- Analyze ratio gains: model-bound vs algorithm-bound (1d)
- PAQ-style context mixing prototype (3d)
- Compression-aware fine-tuning (1w, needs Linux GPU)

### Agent 2: Performance/Hardware
- Profile nb_blend_row hot loop vectorize/GPU (2d)
- Map VRAM composition (done, 0.5d)
- Fused int4+topk kernel (3d, needs MSVC/nvcc)
- Streaming for >10MB inputs (2d)

### Agent 3: CLI/Infrastructure (P0, in progress)
- Build zllm CLI: encode/decode/bench subcommands
- .zllm container: JSON header + raw bitstream
- Decode path: bitstream -> tokens -> text
- pyproject.toml with deps
- Config presets: fast/balanced/ratio
- Smoke tests

### Agent 4: Paper/Evaluation
- Maintain FINAL-REPORT.md
- Benchmark suite automation
- Comparison table vs PAQ8/NNCP/CMIX
- README/README.zh-Hant sync

---

## Step 3: 30-Day Roadmap

### Week 1 (Days 1-7): CLI + Decode Path
- [ ] zllm CLI with encode/decode/bench
- [ ] .zllm container format
- [ ] Full decode path (bitstream -> text)
- [ ] pyproject.toml
- [ ] Smoke test: roundtrip 100KB enwik8

### Week 2 (Days 8-14): Algorithm Deepening
- [ ] PAQ-style context mixing prototype
- [ ] Qwen deep tuning (gate/CONF/LAMBDA/PF ladder, 6+ runs)
- [ ] Qwen 1MB/10MB scale ladder
- [ ] Survey ANS-based neural coders

### Week 3 (Days 15-21): Performance Push
- [ ] nb_blend_row vectorization
- [ ] Streaming architecture for >10MB
- [ ] Memory-efficient pipeline (overlap chunks)
- [ ] Cross-platform benchmarks (Windows vs WSL)

### Week 4 (Days 22-30): Paper + Polish
- [ ] Complete benchmark comparison table
- [ ] FINAL-REPORT.md final pass
- [ ] README polish
- [ ] GitHub release prep
- [ ] arXiv-ready paper draft (if targets hit)
