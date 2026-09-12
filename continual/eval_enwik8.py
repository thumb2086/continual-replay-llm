"""Evaluate big-data M model on enwik8 (out-of-domain test)."""

import torch
import numpy as np
import sys
import time
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from train_v3 import OnlineLLMV3, CharTokenizer, load_grouped_texts
from real_compression import frozen_code_all

CFG_M = dict(embed_dim=256, hidden_dim=512, n_layers=6, n_heads=8)


def main():
    print("=" * 70)
    print("ENWIK8 EVALUATION (out-of-domain)")
    print("=" * 70)
    torch.manual_seed(123)
    np.random.seed(123)
    device = torch.device("cuda")

    # Rebuild identical tokenizer (same data, same order -> same mapping)
    print("\n[1/3] Rebuilding tokenizer...")
    base_path = r"C:\Users\CPXru\Music\playlist-admin\podcasts"
    texts_a, texts_b, _, _ = load_grouped_texts(base_path, max_files_per_group=150)
    tok = CharTokenizer()
    tok.fit(texts_a + texts_b)
    print(f"  Vocab: {tok.vocab_size} (expect 4329)")

    # Load enwik8 (first 100KB)
    print("\n[2/3] Loading enwik8...")
    with open("./data/cloud/enwik8", "rb") as f:
        raw = f.read(100 * 1024)
    text = raw.decode("utf-8", errors="ignore")
    print(f"  Chars: {len(text)}")

    # UNK rate
    unk = sum(1 for c in text if c not in tok.char2idx)
    print(f"  UNK chars: {unk}/{len(text)} ({unk/len(text)*100:.2f}%)")

    chunks = [text[i:i + 128] for i in range(0, len(text) - 128, 128)]
    print(f"  Chunks: {len(chunks)}")

    # Load model
    print("\n[3/3] Loading M-bigdata checkpoint + measuring...")
    model = OnlineLLMV3(vocab_size=tok.vocab_size, **CFG_M).to(device)
    model.load_state_dict(torch.load("./data/bigdata_M.pt.best", map_location=device,
                                     weights_only=False))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    t = time.time()
    nbits, _, nver = frozen_code_all(model, tok, chunks, device, verify=False)
    torch.cuda.synchronize()
    dt = time.time() - t
    n_chars = sum(len(c) for c in chunks)
    bpc = nbits / n_chars
    print(f"\n  enwik8 bpc: {bpc:.3f}")
    print(f"  speed: {n_chars/dt:.0f} chars/s")
    print(f"  peak mem: {torch.cuda.max_memory_allocated(device)/1024**2:.0f} MB")

    print("\n" + "=" * 70)
    print("ENWIK8 RESULT (frozen, zero-shot out-of-domain)")
    print("=" * 70)
    print(f"  Our M-bigdata: {bpc:.3f} bpc")
    print(f"  (Reference points: gzip ~2.5-3 bpc, recent neural SOTA ~1 bpc)")
    print(f"  UNK rate: {unk/len(text)*100:.2f}%")

    with open("./data/enwik8_results.json", "w") as f:
        json.dump({"bpc": bpc, "chars_per_s": n_chars / dt,
                   "unk_rate": unk / len(text), "n_chars": n_chars,
                   "verified": nver}, f, indent=2)
    print("\nSaved to data/enwik8_results.json")


if __name__ == "__main__":
    main()
