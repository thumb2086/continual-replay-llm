"""Scale-up experiment: bigger model + longer training for better ratio.

Same stream, same protocol as sweep_all.py so numbers are comparable.
L config: embed 384 / hidden 768 / 8 layers / 12 heads (~12M params).
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


def main():
    print("=" * 70)
    print("SCALE-UP: L model (~12M) + 20-epoch pretraining")
    print("=" * 70)
    torch.manual_seed(123)
    np.random.seed(123)
    device = torch.device("cuda")
    t0 = time.time()

    base_path = r"C:\Users\CPXru\Desktop\thumb\大拇哥實驗室\online-llm"
    import os
    os.chdir(base_path)

    data_path = r"C:\Users\CPXru\Music\playlist-admin\podcasts"
    texts_a, texts_b, _, _ = load_grouped_texts(data_path, max_files_per_group=25)
    tok = CharTokenizer()
    tok.fit(texts_a + texts_b)
    chunks_a = create_chunks(texts_a, chunk_len=128)
    chunks_b = create_chunks(texts_b, chunk_len=128)
    np.random.shuffle(chunks_a)
    np.random.shuffle(chunks_b)
    stream = chunks_b[:400]
    n_chars = sum(len(c) for c in stream)
    print(f"Vocab: {tok.vocab_size}, stream chars: {n_chars}")

    print("\nTraining L model (20 epochs)...")
    model = OnlineLLMV3(
        vocab_size=tok.vocab_size, embed_dim=384, hidden_dim=768,
        n_layers=8, n_heads=12,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    model.train()
    for epoch in range(20):
        total, nb = 0.0, 0
        for i in range(0, len(chunks_a) - 8, 8):
            total += model.pretrain_step(make_batch(tok, chunks_a[i:i + 8], device), optimizer)
            nb += 1
        print(f"  Epoch {epoch+1}: loss={total/nb:.4f}", flush=True)

    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    del model
    torch.cuda.empty_cache()

    results = {"params": n_params, "settings": {}}
    for name, freeze_after in [("full", None), ("prefix40", 40), ("frozen", "off")]:
        m = OnlineLLMV3(
            vocab_size=tok.vocab_size, embed_dim=384, hidden_dim=768,
            n_layers=8, n_heads=12,
        ).to(device)
        m.load_state_dict(base_state)
        torch.cuda.reset_peak_memory_stats(device)
        t = time.time()
        if freeze_after == "off":
            from real_compression import frozen_code_all
            nbits, _, nver = frozen_code_all(m, tok, stream, device, verify=True)
        else:
            opt = torch.optim.Adam(m.parameters(), lr=1e-4, fused=True)
            nbits, _, nver = neural_compress_stream(
                m, tok, stream, device, adapt=True, replay_pool=chunks_a,
                opt=opt, block_size=8, freeze_after=freeze_after,
            )
        torch.cuda.synchronize()
        dt = time.time() - t
        peak = torch.cuda.max_memory_allocated(device) / 1024**2
        results["settings"][name] = {
            "bpc": nbits / n_chars,
            "chars_per_s": n_chars / dt,
            "peak_mem_mb": peak,
            "verified": nver,
        }
        print(f"  L/{name:10s}: {nbits/n_chars:.3f} bpc, {n_chars/dt:.0f} chars/s, "
              f"peak {peak:.0f} MB (verified {nver})", flush=True)
        del m
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("SCALE-UP vs M (reference: M/dense/full 7.149, prefix40 7.371, frozen 7.563)")
    print("=" * 70)
    for name, r in results["settings"].items():
        print(f"  L/{name:10s}: {r['bpc']:.3f} bpc, {r['chars_per_s']:.0f} chars/s")

    with open("./data/scale_up.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
