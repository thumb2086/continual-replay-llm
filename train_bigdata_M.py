"""Big-data M training with checkpoints (resumable)."""

import torch
import numpy as np
import sys
import time
import json
import torch.nn.functional as F
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from train_v3 import OnlineLLMV3, CharTokenizer, load_grouped_texts, create_chunks
from compare_ablation import make_batch
from real_compression import neural_compress_stream, frozen_code_all

CFG_M = dict(embed_dim=256, hidden_dim=512, n_layers=6, n_heads=8)
FILES_PER_GROUP = 150
MAX_EPOCHS = 15
PATIENCE = 3
CKPT = "./data/bigdata_M.pt"
PROGRESS = "./data/bigdata_M_progress.json"


def main():
    print("=" * 70)
    print("BIG-DATA M (resumable)")
    print("=" * 70)
    torch.manual_seed(123)
    np.random.seed(123)
    device = torch.device("cuda")
    t0 = time.time()

    print("\nLoading data...")
    base_path = r"C:\Users\CPXru\Music\playlist-admin\podcasts"
    texts_a, texts_b, _, _ = load_grouped_texts(base_path, max_files_per_group=FILES_PER_GROUP)
    print(f"  Texts: A={len(texts_a)}, B={len(texts_b)}")
    tok = CharTokenizer()
    tok.fit(texts_a + texts_b)
    print(f"  Vocab: {tok.vocab_size}")
    chunks_a = create_chunks(texts_a, chunk_len=128)
    np.random.shuffle(chunks_a)
    split = int(len(chunks_a) * 0.9)
    train_a, val_a = chunks_a[:split], chunks_a[split:]
    print(f"  Train={len(train_a)}, val={len(val_a)}")

    with open("./data/frozen_stream.json", encoding="utf-8") as f:
        stream = json.load(f)["chunks"]
    n_chars = sum(len(c) for c in stream)

    model = OnlineLLMV3(vocab_size=tok.vocab_size, **CFG_M).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    start_epoch, best_val, patience_left = 0, float("inf"), PATIENCE
    best_state = None

    # Resume if checkpoint exists
    if Path(CKPT).exists() and Path(PROGRESS).exists():
        ck = torch.load(CKPT, map_location=device, weights_only=False)
        model.load_state_dict(ck["state"])
        opt.load_state_dict(ck["opt"])
        with open(PROGRESS) as f:
            pg = json.load(f)
        start_epoch, best_val = pg["next_epoch"], pg["best_val"]
        best_state = {k: v.detach().cpu().clone()
                      for k, v in torch.load(CKPT + ".best", map_location="cpu",
                                             weights_only=False).items()} \
            if Path(CKPT + ".best").exists() else None
        patience_left = pg.get("patience_left", PATIENCE)
        print(f"  Resumed at epoch {start_epoch+1} (best val={best_val:.4f})")

    def save_progress(next_epoch):
        torch.save({"state": model.state_dict(),
                    "opt": opt.state_dict()}, CKPT)
        with open(PROGRESS, "w") as f:
            json.dump({"next_epoch": next_epoch, "best_val": best_val,
                       "patience_left": patience_left}, f)

    # ---- Train ----
    model.train()
    done = False
    for epoch in range(start_epoch, MAX_EPOCHS):
        total, nb = 0.0, 0
        for i in range(0, len(train_a) - 8, 8):
            total += model.pretrain_step(make_batch(tok, train_a[i:i + 8], device), opt)
            nb += 1
        # val
        model.eval()
        vtotal, vnb = 0.0, 0
        with torch.no_grad():
            for i in range(0, min(len(val_a), 400), 8):
                bl = val_a[i:i + 8]
                maxlen = max(len(tok.encode(c)) for c in bl)
                padded = torch.zeros(len(bl), maxlen, dtype=torch.long, device=device)
                for j, c in enumerate(bl):
                    ids = tok.encode(c)
                    padded[j, : len(ids)] = torch.tensor(ids, device=device)
                inp, tgt = padded[:, :-1], padded[:, 1:]
                logits = model(inp)
                ml = min(logits.shape[1], tgt.shape[1])
                vtotal += F.cross_entropy(
                    logits[:, :ml].reshape(-1, logits.size(-1)),
                    tgt[:, :ml].reshape(-1)).item()
                vnb += 1
        model.train()
        val = vtotal / max(vnb, 1)
        print(f"  Epoch {epoch+1}: train={total/nb:.4f} val={val:.4f}", flush=True)
        if val < best_val - 1e-4:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, CKPT + ".best")
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left == 0:
                print("  Early stop", flush=True)
                done = True
                save_progress(MAX_EPOCHS)
                break
        save_progress(epoch + 1)
    else:
        done = True
        save_progress(MAX_EPOCHS)

    # ---- Evaluate best ----
    model.load_state_dict(best_state)
    n_params = sum(p.numel() for p in model.parameters())
    row = {"params": n_params, "best_val": best_val}
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
    for name, freeze_after in [("prefix40", 40), ("full", None)]:
        m = OnlineLLMV3(vocab_size=tok.vocab_size, **CFG_M).to(device)
        m.load_state_dict(best_state)
        opt2 = torch.optim.Adam(m.parameters(), lr=1e-4, fused=True)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        t = time.time()
        nb, _, nv = neural_compress_stream(
            m, tok, stream, device, adapt=True, replay_pool=train_a,
            opt=opt2, block_size=8, freeze_after=freeze_after)
        torch.cuda.synchronize()
        dt = time.time() - t
        row[name] = {"bpc": nb / n_chars, "chars_per_s": n_chars / dt,
                     "peak_mem_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
                     "verified": nv}
        print(f"  {name}: {nb/n_chars:.3f} bpc, {n_chars/dt:.0f} c/s", flush=True)
        del m
        torch.cuda.empty_cache()

    print("\nM BIG-DATA:", {k: (round(v["bpc"], 3) if isinstance(v, dict) else v)
                            for k, v in row.items() if k in ("frozen", "prefix40", "full")})
    with open("./data/bigdata_M_results.json", "w") as f:
        json.dump(row, f, indent=2)
    print(f"Saved ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
