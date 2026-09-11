# Continual Replay LLM + Neural Compression

Replay-first continual learning for small streaming language models — applied to **neural text compression** that beats Huffman by up to 23.6% in real measured bits.

## Compression results (best)

| Setting | bpc | vs Huffman 8.475 |
|---|---|---|
| M + big data + full adapt | **6.472** | **+23.6%** |
| S + big data + full adapt | 6.640 | +21.6% |
| enwik8 zero-shot (M) | 5.365 | cross-domain |

See [compression-paper.md](compression-paper.md) for the full 24-condition sweep, data-mixing curves, enwik8 evaluation, and honest limitations.

## Continual-learning results

Held-out A→B→A→B long cycle (replay batch 8, EWC off):

| Phase | Topic A loss | Topic B loss |
|-------|-------------|-------------|
| After pretrain A | 4.1977 | 5.2536 |
| After online B1 | 4.1111 | 4.5846 |
| After online A2 | 4.1054 | 4.3827 |
| After online B2 | 4.0021 | 4.3102 |

Ablation (same checkpoint, Topic B online):

| Method | Topic A | Topic B |
|--------|---------|---------|
| Pure fine-tuning | 1.102x worse | +23.36% |
| EWC-only | 1.015x worse | +11.64% |
| Replay-only | 0.981x kept | +21.96% |
| EWC + replay | 0.993x kept | +11.10% |

See [paper.md](paper.md) for the full technical report.

## Structure

```
train_v3.py          Core model: Transformer + EWC + memory replay
compare_baseline.py  Pure fine-tuning vs EWC + replay (same checkpoint)
compare_ablation.py  Pure / EWC-only / replay-only / EWC+replay
sweep_lambda.py      EWC lambda sweep with fixed replay
sweep_replay.py      Replay-size sweep with EWC off
run_research_loop.py Autonomous A→B→A→B validation with gates
real_compression.py  Huffman vs neural arithmetic coding (real bits)
sweep_all.py         24-condition grid: size x precision x pruning x adapt
train_bigdata.py     Big-data training + regularization + early stopping
showdown.py          Size x distillation x pruning comparison
paper.md             Continual-learning technical paper draft
compression-paper.md Compression technical report (start here for compression)
continual-learning-report.md  Full experiment log
data/                Result JSON files (metrics only, no weights)
```

## Requirements

```bash
pip install torch numpy
```

A CUDA GPU is recommended (tested on RTX 3060 Ti).

## Data

Training uses local podcast transcripts (not included — private data).

Set the data directory before running:

```powershell
$env:PODCAST_DIR = "C:\path\to\your\transcripts"
```

Then update `base_path` in the scripts to use it (currently hardcoded for the author's machine).

Expected layout: a folder of subdirectories, each containing `.txt` transcript files. Topic A = first half of subdirectories, Topic B = second half.

## Reproduce

```bash
# 1. Full continual-learning validation
python run_research_loop.py

# 2. Ablation study
python compare_ablation.py

# 3. Hyperparameter sweeps
python sweep_lambda.py
python sweep_replay.py
```

## Citation

```bibtex
@misc{continual-replay-llm-2026,
  title  = {Replay-First Continual Learning for Small Streaming Language Models},
  author = {thumb2086},
  year   = {2026},
}
```

## License

MIT — see [LICENSE](LICENSE).
