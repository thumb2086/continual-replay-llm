"""Mixed training: podcast Topic A + Cosmopedia (small model first).

- Podcast tokenizer kept (vocab 4329) for comparability
- Per-epoch subsampling keeps epoch time bounded
- AdamW + early stopping on podcast val (same val as big-data run)
- Eval: frozen podcast stream + enwik8 sample
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import time
import json
import glob
import os
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

import pandas as pd
from train_v3 import OnlineLLMV3, CharTokenizer, load_grouped_texts, create_chunks
from compare_ablation import make_batch
from real_compression import neural_compress_stream, frozen_code_all

CFG_S = dict(embed_dim=128, hidden_dim=256, n_layers=2, n_heads=4)
COSMO_SAMPLES = 175
CHUNKS_PER_EPOCH = 5000
MAX_EPOCHS = 20
PATIENCE = 4


def load_cosmo_chunks(tok, n_samples=COSMO_SAMPLES, chunk_len=128):
    base = r"C:\Users\CPXru\Desktop\thumb\大拇哥實驗室"
    pattern = os.path.join(base, "cortexflow", "data", "cosmopedia-100k",
                           "**", "*.parquet")
    files = sorted(glob.glob(pattern, recursive=True))
    chunks = []
    got = 0
    for f in files:
        df = pd.read_parquet(f, columns=["text"])
        for t in df["text"].tolist():
            if not isinstance(t, str) or len(t) < 200:
                continue
            for i in range(0, len(t) - chunk_len, chunk_len // 2):
                c = t[i:i + chunk_len]
                if len(c) >= 32:
                    chunks.append(c)
            got += 1
            if got >= n_samples:
                break
        if got >= n_samples:
            break
    return chunks


def main():
    print("=" * 70)
    print("MIXED TRAINING: podcast + Cosmopedia (S model)")
    print("=" * 70)
    torch.manual_seed(123)
    np.random.seed(123)
    device = torch.device("cuda")
    t0 = time.time()

    print("\n[1/5] Loading podcast data...")
    base_path = r"C:\Users\CPXru\Music\playlist-admin\podcasts"
    texts_a, texts_b, _, _ = load_grouped_texts(base_path, max_files_per_group=150)
    tok = CharTokenizer()
    tok.fit(texts_a + texts_b)
    print(f"  Vocab: {tok.vocab_size}")
    chunks_a = create_chunks(texts_a, chunk_len=128)
    np.random.shuffle(chunks_a)
    split = int(len(chunks_a) * 0.9)
    train_a, val_a = chunks_a[:split], chunks_a[split:]
    print(f"  Podcast A: train={len(train_a)}, val={len(val_a)}")

    print("\n[2/5] Loading Cosmopedia chunks...")
    cosmo = load_cosmo_chunks(tok)
    np.random.shuffle(cosmo)
    print(f"  Cosmopedia chunks: {len(cosmo)}")

    pool = train_a + cosmo
    print(f"  Mixed pool: {len(pool)} chunks "
          f"({len(train_a)/len(pool)*100:.0f}% podcast / {len(cosmo)/len(pool)*100:.0f}% cosmo)")

    with open("./data/frozen_stream.json", encoding="utf-8") as f:
        stream = json.load(f)["chunks"]
    n_chars = sum(len(c) for c in stream)

    print("\n[3/5] Training S (AdamW + early stopping on podcast val)...")
    model = OnlineLLMV3(vocab_size=tok.vocab_size, **CFG_S).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    model.train()
    best_val, best_state, patience_left = float("inf"), None, PATIENCE

    def eval_val():
        model.eval()
        tot, nb = 0.0, 0
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
                tot += F.cross_entropy(
                    logits[:, :ml].reshape(-1, logits.size(-1)),
                    tgt[:, :ml].reshape(-1)).item()
                nb += 1
        model.train()
        return tot / max(nb, 1)

    for epoch in range(MAX_EPOCHS):
        idx = np.random.choice(len(pool), min(CHUNKS_PER_EPOCH, len(pool)), replace=False)
        total, nb = 0.0, 0
        for k in range(0, len(idx), 8):
            batch = [pool[i] for i in idx[k:k + 8]]
            total += model.pretrain_step(make_batch(tok, batch, device), opt)
            nb += 1
        val = eval_val()
        print(f"  Epoch {epoch+1}: train={total/nb:.4f} val={val:.4f}", flush=True)
        if val < best_val - 1e-4:
            best_val = val
            best_state = {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left == 0:
                print("  Early stop", flush=True)
                break
    model.load_state_dict(best_state)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Best val: {best_val:.4f}, params: {n_params:,}")

    print("\n[4/5] Evaluating on frozen podcast stream...")
    results = {"params": n_params, "best_val": best_val}
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    t = time.time()
    nbits, _, nver = frozen_code_all(model, tok, stream, device, verify=True)
    torch.cuda.synchronize()
    dt = time.time() - t
    results["frozen"] = {"bpc": nbits / n_chars, "chars_per_s": n_chars / dt,
                         "verified": nver}
    print(f"  frozen: {nbits/n_chars:.3f} bpc, {n_chars/dt:.0f} c/s", flush=True)

    opt2 = torch.optim.Adam(model.parameters(), lr=1e-4, fused=True)
    torch.cuda.synchronize()
    t = time.time()
    nb, _, nv = neural_compress_stream(
        model, tok, stream, device, adapt=True, replay_pool=train_a,
        opt=opt2, block_size=8, freeze_after=40)
    torch.cuda.synchronize()
    dt = time.time() - t
    results["prefix40"] = {"bpc": nb / n_chars, "chars_per_s": n_chars / dt,
                           "verified": nv}
    print(f"  prefix40: {nb/n_chars:.3f} bpc, {n_chars/dt:.0f} c/s", flush=True)

    print("\n[5/5] Evaluating on enwik8 sample...")
    with open("./data/cloud/enwik8", "rb") as f:
        raw = f.read(100 * 1024)
    etext = raw.decode("utf-8", errors="ignore")
    echunks = [etext[i:i + 128] for i in range(0, len(etext) - 128, 128)][:400]
    unk = sum(1 for c in etext[:len(echunks) * 128] if c not in tok.char2idx)
    torch.cuda.synchronize()
    t = time.time()
    ebits, _, _ = frozen_code_all(model, tok, echunks, device, verify=False)
    torch.cuda.synchronize()
    dt = time.time() - t
    echars = sum(len(c) for c in echunks)
    results["enwik8"] = {"bpc": ebits / echars, "chars_per_s": echars / dt,
                         "unk_rate": unk / (len(echunks) * 128)}
    print(f"  enwik8: {ebits/echars:.3f} bpc, UNK {unk/(len(echunks)*128)*100:.2f}%", flush=True)

    print("\n" + "=" * 70)
    print("MIXED-TRAINING S RESULTS")
    print("Reference: S-bigdata-only frozen 6.736 / prefix40 6.699; enwik8 5.365 (M model)")
    print("=" * 70)
    for k in ["frozen", "prefix40", "enwik8"]:
        print(f"  {k:10s}: {results[k]['bpc']:.3f} bpc")

    with open("./data/mixed_S_balanced_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
