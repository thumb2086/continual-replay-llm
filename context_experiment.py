"""Context-length experiment: can long context replace weight updates?

Compares on Topic B stream (frozen model, no backward passes):
- context 128 (baseline, matches chunk size)
- context 512 / 1024 / 2048 (in-context adaptation)

If long-context frozen matches adaptive ratio, weight updates during
encoding are unnecessary.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import time
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from train_v3 import OnlineLLMV3, CharTokenizer, load_grouped_texts, create_chunks
from compare_ablation import make_batch
from real_compression import gpu_quantize, nb_encode_count, TOTAL


def measure_with_context(model, tok, stream_ids, device, ctx_len, max_chunks=200):
    """Sliding-window coding: each chunk gets up to ctx_len chars of history."""
    model.eval()
    total_bits, total_chars = 0, 0
    # Concatenate stream into one id sequence
    full = []
    for ids in stream_ids:
        full.extend(ids)
    full = np.array(full, dtype=np.int64)

    torch.cuda.synchronize()
    t = time.time()
    pos = 0
    n_chunks = 0
    with torch.no_grad():
        while pos < len(full) and n_chunks < max_chunks:
            # Predict next 128 symbols given up to ctx_len history
            start = max(0, pos - ctx_len)
            window = full[start : pos + 128]
            if len(window) < 3:
                break
            x = torch.tensor(window, dtype=torch.long, device=device).unsqueeze(0)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
            # We only code the NEW symbols (last up to 128)
            new_start = pos - start  # offset of first new symbol in window
            V = logits.shape[-1]
            cum_all = gpu_quantize(logits).reshape(1, -1, V + 1)[0]
            n_new = min(128, len(full) - pos)
            for k in range(n_new):
                t_idx = new_start + k - 1  # predict full[pos+k] from probs[t_idx]
                if t_idx < 0:
                    continue
                cum = np.ascontiguousarray(cum_all[t_idx])
                nbits = int(nb_encode_count(cum.reshape(1, -1), np.array([full[pos + k]], dtype=np.int64)))
                if nbits < 0:
                    raise ArithmeticError("encoder error")
                total_bits += nbits
                total_chars += 1
            pos += n_new
            n_chunks += 1
    torch.cuda.synchronize()
    dt = time.time() - t
    return total_bits / total_chars, total_chars / dt


def main():
    print("=" * 70)
    print("CONTEXT-LENGTH EXPERIMENT (frozen model, no weight updates)")
    print("=" * 70)
    torch.manual_seed(123)
    np.random.seed(123)
    device = torch.device("cuda")

    base_path = r"C:\Users\CPXru\Music\playlist-admin\podcasts"
    texts_a, texts_b, _, _ = load_grouped_texts(base_path, max_files_per_group=25)
    tok = CharTokenizer()
    tok.fit(texts_a + texts_b)
    chunks_a = create_chunks(texts_a, chunk_len=128)
    chunks_b = create_chunks(texts_b, chunk_len=128)
    np.random.shuffle(chunks_a)
    np.random.shuffle(chunks_b)
    print(f"Vocab: {tok.vocab_size}")

    # Pretrain on A (same recipe as before)
    print("\nPretraining on Topic A...")
    model = OnlineLLMV3(
        vocab_size=tok.vocab_size, embed_dim=256, hidden_dim=512, n_layers=6, n_heads=8
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    model.train()
    for epoch in range(10):
        total, nb = 0.0, 0
        for i in range(0, len(chunks_a) - 8, 8):
            total += model.pretrain_step(make_batch(tok, chunks_a[i:i + 8], device), optimizer)
            nb += 1
        print(f"  Epoch {epoch+1}: loss={total/nb:.4f}")

    stream_ids = []
    for c in chunks_b[:200]:
        ids = tok.encode(c)
        if len(ids) >= 3:
            stream_ids.append(ids)

    results = {}
    for ctx in [128, 512, 1024, 2048]:
        print(f"\nContext {ctx}...")
        bpc, cps = measure_with_context(model, tok, stream_ids, device, ctx)
        results[str(ctx)] = {"bpc": bpc, "chars_per_s": cps}
        print(f"  {bpc:.3f} bits/char, {cps:.0f} chars/s")

    print("\n" + "=" * 70)
    print("CONTEXT vs ADAPTATION")
    print("=" * 70)
    print("  (adaptive weight-update reference: 7.162 bpc @ ~32K chars/s)")
    for ctx, r in results.items():
        print(f"  ctx {ctx:>5s}: {r['bpc']:.3f} bpc, {r['chars_per_s']:.0f} chars/s")

    with open("./data/context_experiment.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to data/context_experiment.json")


if __name__ == "__main__":
    main()
