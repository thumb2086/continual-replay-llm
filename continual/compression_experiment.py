"""Adaptive Compression Experiment.

Compares theoretical compression rate (cross-entropy in bits):
1. Fixed model (pretrained on A, frozen) on Topic B stream
2. Online-adapting model (updates causally as it compresses) on Topic B stream
3. gzip / bz2 / lzma baselines on the same Topic B text

If online adaptation helps, the adaptive model's per-chunk bits should
decrease over the stream while the fixed model stays flat.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import sys
import gzip
import bz2
import lzma
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import sys as _sys_root; _sys_root.path.insert(0, ".")  # run-from-root: core lives at repo root
from train_v3 import OnlineLLMV3, CharTokenizer, load_grouped_texts, create_chunks
from compare_ablation import make_batch


def stream_bits(model, tokenizer, chunks, device, adapt=False, replay_pool=None, opt=None):
    """Compress a stream chunk-by-chunk, return per-chunk bits/char.

    Causal: when adapt=True, the model only updates on past chunks.
    """
    bits_per_chunk = []
    total_bits = 0.0
    total_chars = 0

    model.train() if adapt else model.eval()

    with torch.set_grad_enabled(adapt):
        for chunk in chunks:
            tokens = tokenizer.encode(chunk, return_tensors="pt").to(device)
            if tokens.shape[1] < 3:
                continue

            # 1. Measure bits for this chunk (before seeing it, if adapting)
            inp = tokens[:, :-1]
            tgt = tokens[:, 1:]
            logits = model.forward(inp)
            min_len = min(logits.shape[1], tgt.shape[1])
            nll = F.cross_entropy(
                logits[:, :min_len].reshape(-1, logits.size(-1)),
                tgt[:, :min_len].reshape(-1),
                reduction="sum",
            ).item()
            chunk_bits = nll / np.log(2)  # nats -> bits
            n_chars = min_len
            bits_per_chunk.append(chunk_bits / n_chars)
            total_bits += chunk_bits
            total_chars += n_chars

            # 2. Adapt on this chunk (causal: only after measuring)
            if adapt:
                replay = None
                if replay_pool is not None:
                    picks = [replay_pool[i] for i in np.random.choice(len(replay_pool), 2, replace=False)]
                    replay = make_batch(tokenizer, picks, device)
                model.online_step(tokens, opt, replay, 0)

    return bits_per_chunk, total_bits / max(total_chars, 1)


def classical_baselines(text):
    """Compress text with classical compressors, return bits/char."""
    raw = text.encode("utf-8")
    n_chars = len(text)
    results = {}
    for name, fn in [
        ("gzip", gzip.compress),
        ("bz2", bz2.compress),
        ("lzma", lzma.compress),
    ]:
        comp = fn(raw)
        results[name] = len(comp) * 8 / n_chars
    results["raw_utf8"] = len(raw) * 8 / n_chars
    return results


def main():
    print("=" * 70)
    print("ADAPTIVE COMPRESSION EXPERIMENT")
    print("=" * 70)

    torch.manual_seed(123)
    np.random.seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Data ----
    base_path = r"C:\Users\CPXru\Music\playlist-admin\podcasts"
    texts_a, texts_b, _, _ = load_grouped_texts(base_path, max_files_per_group=25)

    tok = CharTokenizer()
    tok.fit(texts_a + texts_b)
    print(f"Vocab: {tok.vocab_size}")

    chunks_a = create_chunks(texts_a, chunk_len=128)
    chunks_b = create_chunks(texts_b, chunk_len=128)
    np.random.shuffle(chunks_a)
    np.random.shuffle(chunks_b)
    print(f"Chunks: A={len(chunks_a)}, B={len(chunks_b)}")

    # ---- Pretrain on A ----
    print("\nPretraining on Topic A...")
    model = OnlineLLMV3(
        vocab_size=tok.vocab_size, embed_dim=128, hidden_dim=256, n_layers=4, n_heads=4
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    model.train()
    for epoch in range(3):
        total = 0.0
        n_batches = 0
        for i in range(0, len(chunks_a) - 8, 8):
            padded = make_batch(tok, chunks_a[i:i + 8], device)
            total += model.pretrain_step(padded, optimizer)
            n_batches += 1
        print(f"  Epoch {epoch+1}: loss={total/n_batches:.4f}")

    # Clone into fixed and adaptive
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    fixed = OnlineLLMV3(
        vocab_size=tok.vocab_size, embed_dim=128, hidden_dim=256, n_layers=4, n_heads=4
    ).to(device)
    adaptive = OnlineLLMV3(
        vocab_size=tok.vocab_size, embed_dim=128, hidden_dim=256, n_layers=4, n_heads=4
    ).to(device)
    fixed.load_state_dict(state)
    adaptive.load_state_dict(state)

    stream = chunks_b[:400]
    print(f"\nCompressing Topic B stream ({len(stream)} chunks)...")

    # ---- Fixed model ----
    print("\nFixed model...")
    fixed_bits, fixed_avg = stream_bits(fixed, tok, stream, device, adapt=False)
    print(f"  Avg bits/char: {fixed_avg:.3f}")

    # ---- Adaptive model ----
    print("\nAdaptive model (causal online learning)...")
    adapt_opt = torch.optim.Adam(adaptive.parameters(), lr=1e-4)
    adapt_bits, adapt_avg = stream_bits(
        adaptive, tok, stream, device, adapt=True, replay_pool=chunks_a, opt=adapt_opt
    )
    print(f"  Avg bits/char: {adapt_avg:.3f}")

    # Adaptation curve: first 10% vs last 10%
    n = len(adapt_bits)
    first = np.mean(adapt_bits[: n // 10])
    last = np.mean(adapt_bits[-n // 10 :])
    fixed_first = np.mean(fixed_bits[: n // 10])
    fixed_last = np.mean(fixed_bits[-n // 10 :])
    print(f"\n  Adaptive first 10%: {first:.3f} -> last 10%: {last:.3f}")
    print(f"  Fixed first 10%: {fixed_first:.3f} -> last 10%: {fixed_last:.3f}")

    # ---- Classical baselines ----
    print("\nClassical baselines on same text...")
    full_text = "".join(stream)
    classical = classical_baselines(full_text)
    for name, bpc in classical.items():
        print(f"  {name}: {bpc:.3f} bits/char")

    # ---- Verdict ----
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  Fixed model:    {fixed_avg:.3f} bits/char")
    print(f"  Adaptive model: {adapt_avg:.3f} bits/char")
    print(f"  gzip:           {classical['gzip']:.3f} bits/char")
    print(f"  bz2:            {classical['bz2']:.3f} bits/char")
    print(f"  lzma:           {classical['lzma']:.3f} bits/char")

    if adapt_avg < fixed_avg:
        print(f"\n  ADAPTATION HELPS: {(fixed_avg-adapt_avg)/fixed_avg*100:.1f}% better than fixed")
    else:
        print("\n  Adaptation did not help in this run")

    best_classical = min(classical["gzip"], classical["bz2"], classical["lzma"])
    if adapt_avg < best_classical:
        print(f"  BEATS CLASSICAL: adaptive {adapt_avg:.3f} < best classical {best_classical:.3f}")
    else:
        print(f"  Gap to classical: {adapt_avg-best_classical:.3f} bits/char remaining")

    results = {
        "fixed_bpc": fixed_avg,
        "adaptive_bpc": adapt_avg,
        "adaptive_first10": float(first),
        "adaptive_last10": float(last),
        "fixed_first10": float(fixed_first),
        "fixed_last10": float(fixed_last),
        "classical": classical,
        "adapt_curve": [float(x) for x in adapt_bits[::10]],
        "fixed_curve": [float(x) for x in fixed_bits[::10]],
    }
    Path("./data").mkdir(parents=True, exist_ok=True)
    with open("./data/compression_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to data/compression_results.json")


if __name__ == "__main__":
    main()
