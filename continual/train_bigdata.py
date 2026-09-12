"""Big-data training: 6x more podcast transcripts + regularization.

- 300 texts (150/group) instead of 50, all LOCAL (no download needed)
- AdamW with weight decay
- Early stopping on held-out validation loss
- Test stream stays frozen (same 400 chunks) for comparability
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
from real_compression import neural_compress_stream, frozen_code_all

CFG_S = dict(embed_dim=128, hidden_dim=256, n_layers=2, n_heads=4)
CFG_M = dict(embed_dim=256, hidden_dim=512, n_layers=6, n_heads=8)
FILES_PER_GROUP = 150
MAX_EPOCHS = 15
PATIENCE = 3


def train_with_early_stopping(model, tok, train_chunks, val_chunks, device,
                              lr=3e-4, weight_decay=0.05, tag=""):
    """Train with AdamW + early stopping on val loss. Returns best epoch."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    patience_left = PATIENCE

    for epoch in range(MAX_EPOCHS):
        total, nb = 0.0, 0
        for i in range(0, len(train_chunks) - 8, 8):
            total += model.pretrain_step(make_batch(tok, train_chunks[i:i + 8], device), opt)
            nb += 1
        # Validation (no grad)
        model.eval()
        vtotal, vnb = 0.0, 0
        with torch.no_grad():
            for i in range(0, min(len(val_chunks), 400), 8):
                batch_list = val_chunks[i:i + 8]
                maxlen = max(len(tok.encode(c)) for c in batch_list)
                padded = torch.zeros(len(batch_list), maxlen, dtype=torch.long, device=device)
                for j, c in enumerate(batch_list):
                    ids = tok.encode(c)
                    padded[j, : len(ids)] = torch.tensor(ids, device=device)
                inp, tgt = padded[:, :-1], padded[:, 1:]
                logits = model(inp)
                ml = min(logits.shape[1], tgt.shape[1])
                import torch.nn.functional as F
                vtotal += F.cross_entropy(
                    logits[:, :ml].reshape(-1, logits.size(-1)),
                    tgt[:, :ml].reshape(-1)).item()
                vnb += 1
        model.train()
        val = vtotal / max(vnb, 1)
        print(f"  [{tag}] Epoch {epoch+1}: train={total/nb:.4f} val={val:.4f}", flush=True)
        if val < best_val - 1e-4:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"  [{tag}] Early stop at epoch {epoch+1} (best: {best_epoch})", flush=True)
                break

    model.load_state_dict(best_state)
    return best_epoch, best_val


def main():
    print("=" * 70)
    print("BIG DATA + REGULARIZATION TRAINING")
    print("=" * 70)
    torch.manual_seed(123)
    np.random.seed(123)
    device = torch.device("cuda")
    t0 = time.time()

    # ---- Load 6x data ----
    print("\n[1/4] Loading data (150 files/group, local, no download)...")
    base_path = r"C:\Users\CPXru\Music\playlist-admin\podcasts"
    texts_a, texts_b, n_a, n_b = load_grouped_texts(
        base_path, max_files_per_group=FILES_PER_GROUP)
    print(f"  Dir groups: A={n_a}, B={n_b}")
    print(f"  Texts: A={len(texts_a)}, B={len(texts_b)}")

    tok = CharTokenizer()
    tok.fit(texts_a + texts_b)
    print(f"  Vocab: {tok.vocab_size}")

    chunks_a = create_chunks(texts_a, chunk_len=128)
    np.random.shuffle(chunks_a)
    # 90/10 train/val split by chunk (val only for early stopping)
    split = int(len(chunks_a) * 0.9)
    train_a, val_a = chunks_a[:split], chunks_a[split:]
    print(f"  Topic A chunks: train={len(train_a)}, val={len(val_a)}")

    with open("./data/frozen_stream.json", encoding="utf-8") as f:
        stream = json.load(f)["chunks"]
    n_chars = sum(len(c) for c in stream)
    print(f"  Frozen test stream: {len(stream)} chunks (unchanged)")

    results = {}

    for size_name, cfg in [("S", CFG_S), ("M", CFG_M)]:
        print(f"\n[2/4] Training {size_name} (AdamW wd=0.05 + early stopping)...")
        model = OnlineLLMV3(vocab_size=tok.vocab_size, **cfg).to(device)
        best_ep, best_val = train_with_early_stopping(
            model, tok, train_a, val_a, device, tag=size_name)
        n_params = sum(p.numel() for p in model.parameters())
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # ---- Evaluate: frozen + prefix40 + full ----
        print(f"\n[3/4] Evaluating {size_name}...")
        row = {"params": n_params, "best_epoch": best_ep, "best_val": best_val}
        # frozen
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        t = time.time()
        nbits, _, nver = frozen_code_all(model, tok, stream, device, verify=True)
        torch.cuda.synchronize()
        dt = time.time() - t
        row["frozen"] = {"bpc": nbits / n_chars, "chars_per_s": n_chars / dt,
                         "peak_mem_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
                         "verified": nver}
        print(f"  frozen: {nbits/n_chars:.3f} bpc, {n_chars/dt:.0f} c/s", flush=True)
        # prefix40 + full (fresh clones, same start)
        for name, freeze_after in [("prefix40", 40), ("full", None)]:
            m = OnlineLLMV3(vocab_size=tok.vocab_size, **cfg).to(device)
            m.load_state_dict(state)
            opt = torch.optim.Adam(m.parameters(), lr=1e-4, fused=True)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()
            t = time.time()
            nb, _, nv = neural_compress_stream(
                m, tok, stream, device, adapt=True, replay_pool=train_a,
                opt=opt, block_size=8, freeze_after=freeze_after)
            torch.cuda.synchronize()
            dt = time.time() - t
            row[name] = {"bpc": nb / n_chars, "chars_per_s": n_chars / dt,
                         "peak_mem_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
                         "verified": nv}
            print(f"  {name}: {nb/n_chars:.3f} bpc, {n_chars/dt:.0f} c/s", flush=True)
            del m
            torch.cuda.empty_cache()
        results[size_name] = row
        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("BIG-DATA RESULTS (frozen test stream, comparable with all prior runs)")
    print("Reference (small data): M/full 7.149, M/prefix40 7.371, M/frozen 7.563")
    print("=" * 70)
    for name, r in results.items():
        print(f"  {name} ({r['params']:,} params, best epoch {r['best_epoch']}):")
        for k in ["frozen", "prefix40", "full"]:
            print(f"    {k:10s}: {r[k]['bpc']:.3f} bpc, {r[k]['chars_per_s']:.0f} c/s")

    with open("./data/bigdata_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
