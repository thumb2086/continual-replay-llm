"""Adaptation parameter-count sweep: how many params must move during encoding?

Compares (same base model, same frozen stream):
- full: all parameters
- last-layer: last transformer layer + output head
- bias-only (BitFit): bias terms only
- head-only: output head only

Fewer moving params = faster backward + less overfitting risk.
"""

import torch
import numpy as np
import sys
import time
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from train_v3 import OnlineLLMV3, CharTokenizer, load_grouped_texts, create_chunks
from compare_ablation import make_batch
from real_compression import neural_compress_stream


def set_trainable(model, mode):
    """Freeze all params except the target group. Returns trainable count."""
    for p in model.parameters():
        p.requires_grad_(False)
    if mode == "full":
        for p in model.parameters():
            p.requires_grad_(True)
    elif mode == "last-layer":
        for p in model.transformer.layers[-1].parameters():
            p.requires_grad_(True)
        for p in model.head.parameters():
            p.requires_grad_(True)
    elif mode == "bias-only":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad_(True)
    elif mode == "head-only":
        for p in model.head.parameters():
            p.requires_grad_(True)
    else:
        raise ValueError(mode)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    print("=" * 70)
    print("ADAPTATION PARAM-COUNT SWEEP")
    print("=" * 70)
    torch.manual_seed(123)
    np.random.seed(123)
    device = torch.device("cuda")
    t0 = time.time()

    base_path = r"C:\Users\CPXru\Music\playlist-admin\podcasts"
    texts_a, texts_b, _, _ = load_grouped_texts(base_path, max_files_per_group=25)
    tok = CharTokenizer()
    tok.fit(texts_a + texts_b)
    chunks_a = create_chunks(texts_a, chunk_len=128)
    np.random.shuffle(chunks_a)

    with open("./data/frozen_stream.json", encoding="utf-8") as f:
        stream = json.load(f)["chunks"]
    n_chars = sum(len(c) for c in stream)
    print(f"Frozen stream: {len(stream)} chunks, {n_chars} chars")

    # ---- Pretrain base M model once ----
    print("\nPretraining base M model...")
    base = OnlineLLMV3(
        vocab_size=tok.vocab_size, embed_dim=256, hidden_dim=512,
        n_layers=6, n_heads=8,
    ).to(device)
    optimizer = torch.optim.Adam(base.parameters(), lr=3e-4)
    base.train()
    for epoch in range(10):
        total, nb = 0.0, 0
        for i in range(0, len(chunks_a) - 8, 8):
            total += base.pretrain_step(make_batch(tok, chunks_a[i:i + 8], device), optimizer)
            nb += 1
        print(f"  Epoch {epoch+1}: loss={total/nb:.4f}", flush=True)
    base_state = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
    del base
    torch.cuda.empty_cache()

    # ---- Sweep adaptation param groups (prefix40 setting) ----
    print("\nAdaptation sweep (prefix40, block 8)...")
    results = {}
    for mode in ["full", "last-layer", "bias-only", "head-only"]:
        m = OnlineLLMV3(
            vocab_size=tok.vocab_size, embed_dim=256, hidden_dim=512,
            n_layers=6, n_heads=8,
        ).to(device)
        m.load_state_dict(base_state)
        n_train = set_trainable(m, mode)
        n_total = sum(p.numel() for p in m.parameters())
        opt = torch.optim.Adam(
            [p for p in m.parameters() if p.requires_grad], lr=1e-4, fused=True
        )
        t = time.time()
        nbits, _, nver = neural_compress_stream(
            m, tok, stream, device, adapt=True, replay_pool=chunks_a,
            opt=opt, block_size=8, freeze_after=40,
        )
        dt = time.time() - t
        results[mode] = {
            "trainable": n_train,
            "trainable_pct": n_train / n_total * 100,
            "bpc": nbits / n_chars,
            "chars_per_s": n_chars / dt,
            "verified": nver,
        }
        print(f"  {mode:12s}: {n_train:>9,} params ({n_train/n_total*100:5.2f}%) | "
              f"{nbits/n_chars:.3f} bpc, {n_chars/dt:.0f} chars/s (verified {nver})",
              flush=True)
        del m
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("ADAPT PARAM-COUNT RESULTS")
    print("=" * 70)
    for mode, r in results.items():
        print(f"  {mode:12s}: {r['trainable_pct']:5.2f}% params move | "
              f"{r['bpc']:.3f} bpc | {r['chars_per_s']:.0f} chars/s")

    with open("./data/adapt_param_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
