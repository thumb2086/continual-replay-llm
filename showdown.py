"""Size x Distillation x Pruning showdown (same frozen stream).

Compares, at matched parameter budgets:
1. S trained from scratch (1.1M)
2. S distilled from M teacher (1.1M)
3. M pruned to ~S size (FFN channels)
4. M dense reference (4.8M)

Metrics: frozen bpc + speed, plus fast bias-only prefix adaptation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import time
import json
import copy
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from train_v3 import OnlineLLMV3, CharTokenizer, load_grouped_texts, create_chunks
from compare_ablation import make_batch
from real_compression import frozen_code_all, neural_compress_stream
from prune_experiment import prune_ffn_channels

CFG_S = dict(embed_dim=128, hidden_dim=256, n_layers=2, n_heads=4)
CFG_M = dict(embed_dim=256, hidden_dim=512, n_layers=6, n_heads=8)
PRETRAIN_EPOCHS = 10


def pretrain(model, tok, chunks_a, device, epochs, lr=3e-4, tag=""):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total, nb = 0.0, 0
        for i in range(0, len(chunks_a) - 8, 8):
            total += model.pretrain_step(make_batch(tok, chunks_a[i:i + 8], device), opt)
            nb += 1
        print(f"  [{tag}] Epoch {epoch+1}: loss={total/nb:.4f}", flush=True)


def distill(student, teacher, tok, chunks_a, device, epochs=10, T=2.0, alpha=0.7):
    """Train student on teacher soft targets + hard targets."""
    teacher.eval()
    opt = torch.optim.Adam(student.parameters(), lr=3e-4)
    student.train()
    for epoch in range(epochs):
        total, nb = 0.0, 0
        for i in range(0, len(chunks_a) - 8, 8):
            batch_list = chunks_a[i:i + 8]
            maxlen = max(len(tok.encode(c)) for c in batch_list)
            padded = torch.zeros(len(batch_list), maxlen, dtype=torch.long, device=device)
            for j, c in enumerate(batch_list):
                ids = tok.encode(c)
                padded[j, : len(ids)] = torch.tensor(ids, device=device)
            inp, tgt = padded[:, :-1], padded[:, 1:]
            with torch.no_grad():
                t_logits = teacher(inp)
            s_logits = student(inp)
            min_len = min(s_logits.shape[1], tgt.shape[1])
            hard = F.cross_entropy(
                s_logits[:, :min_len].reshape(-1, s_logits.size(-1)),
                tgt[:, :min_len].reshape(-1),
            )
            soft = F.kl_div(
                F.log_softmax(s_logits[:, :min_len] / T, dim=-1),
                F.softmax(t_logits[:, :min_len].detach() / T, dim=-1),
                reduction="batchmean",
            ) * (T * T)
            loss = alpha * soft + (1 - alpha) * hard
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        print(f"  [distill] Epoch {epoch+1}: loss={total/nb:.4f}", flush=True)


def measure_frozen(model, tok, stream, device):
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    t = time.time()
    nbits, _, nver = frozen_code_all(model, tok, stream, device, verify=False)
    torch.cuda.synchronize()
    dt = time.time() - t
    n_chars = sum(len(c) for c in stream)
    return {
        "bpc": nbits / n_chars,
        "chars_per_s": n_chars / dt,
        "peak_mem_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "verified": nver,
    }


def measure_adapt_bias(model, tok, stream, chunks_a, device):
    """Fast bias-only prefix40 adaptation."""
    for p in model.parameters():
        p.requires_grad_(False)
    for n, p in model.named_parameters():
        if "bias" in n:
            p.requires_grad_(True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=1e-4, fused=True)
    n_chars = sum(len(c) for c in stream)
    torch.cuda.synchronize()
    t = time.time()
    nbits, _, nver = neural_compress_stream(
        model, tok, stream, device, adapt=True, replay_pool=chunks_a,
        opt=opt, block_size=8, freeze_after=40,
    )
    torch.cuda.synchronize()
    dt = time.time() - t
    return {"bpc": nbits / n_chars, "chars_per_s": n_chars / dt,
            "verified": nver, "trainable": n_train}


def main():
    print("=" * 70)
    print("SIZE x DISTILLATION x PRUNING SHOWDOWN")
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
    print(f"Vocab: {tok.vocab_size}, stream: {len(stream)} chunks")

    results = {}

    def reg(name, model, extra=None):
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n[{name}] params={n_params:,}")
        fz = measure_frozen(model, tok, stream, device)
        print(f"  frozen: {fz['bpc']:.3f} bpc, {fz['chars_per_s']:.0f} c/s", flush=True)
        ad = measure_adapt_bias(model, tok, stream, chunks_a, device)
        print(f"  bias-adapt: {ad['bpc']:.3f} bpc, {ad['chars_per_s']:.0f} c/s", flush=True)
        results[name] = {"params": n_params, "frozen": fz, "bias_adapt": ad,
                         **(extra or {})}
        return model

    # 1. M teacher
    print("\n--- Training M teacher ---")
    teacher = OnlineLLMV3(vocab_size=tok.vocab_size, **CFG_M).to(device)
    pretrain(teacher, tok, chunks_a, device, PRETRAIN_EPOCHS, tag="M")
    teacher_state = {k: v.detach().cpu().clone() for k, v in teacher.state_dict().items()}
    reg("M-teacher", teacher)

    # 2. S from scratch
    print("\n--- Training S from scratch ---")
    s_scratch = OnlineLLMV3(vocab_size=tok.vocab_size, **CFG_S).to(device)
    pretrain(s_scratch, tok, chunks_a, device, PRETRAIN_EPOCHS, tag="S-scratch")
    reg("S-scratch", s_scratch)

    # 3. S distilled from M
    print("\n--- Distilling S from M ---")
    s_distil = OnlineLLMV3(vocab_size=tok.vocab_size, **CFG_S).to(device)
    distill(s_distil, teacher, tok, chunks_a, device, epochs=PRETRAIN_EPOCHS)
    reg("S-distilled", s_distil)

    # 4. M pruned to ~S size + recovery
    print("\n--- Pruning M + recovery ---")
    m_pruned = OnlineLLMV3(vocab_size=tok.vocab_size, **CFG_M).to(device)
    m_pruned.load_state_dict(teacher_state)
    prune_ffn_channels(m_pruned, keep_ratio=0.25)
    print(f"  pruned params: {sum(p.numel() for p in m_pruned.parameters()):,}")
    pretrain(m_pruned, tok, chunks_a, device, 2, tag="M-pruned-recover")
    reg("M-pruned", m_pruned)

    print("\n" + "=" * 70)
    print("SHOWDOWN (frozen bpc / bias-adapt bpc)")
    print("=" * 70)
    for name, r in results.items():
        print(f"  {name:12s} {r['params']:>9,} params | "
              f"frozen {r['frozen']['bpc']:.3f} bpc @ {r['frozen']['chars_per_s']:.0f} c/s | "
              f"adapt {r['bias_adapt']['bpc']:.3f} bpc @ {r['bias_adapt']['chars_per_s']:.0f} c/s")

    with open("./data/showdown.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
