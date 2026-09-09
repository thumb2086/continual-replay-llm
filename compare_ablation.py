"""Ablation comparison: pure, EWC-only, replay-only, EWC+replay."""

import torch
import torch.nn.functional as F
import numpy as np
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from train_v3 import OnlineLLMV3, CharTokenizer, load_grouped_texts, create_chunks

PRETRAIN_EPOCHS = 5
ONLINE_EPOCHS = 8
ONLINE_CHUNKS = 300
PRETRAIN_BATCH = 8
PRETRAIN_LR = 3e-4
ONLINE_LR = 1e-4
EWC_LAMBDA = 5000
REPLAY_BATCH = 2
FISHER_SAMPLES = 200
EVAL_SAMPLES = 50


def make_batch(tokenizer, chunks, device):
    token_lists = [tokenizer.encode(c, return_tensors="pt").to(device) for c in chunks]
    max_len = max(t.shape[1] for t in token_lists)
    padded = torch.zeros(len(token_lists), max_len, dtype=torch.long, device=device)
    for j, t in enumerate(token_lists):
        padded[j, : t.shape[1]] = t.squeeze(0)
    return padded


def eval_loss(model, tokenizer, data, device):
    total = 0.0
    n = min(EVAL_SAMPLES, len(data))
    for i in range(n):
        tokens = tokenizer.encode(data[i], return_tensors="pt").to(device)
        total += model.evaluate_loss(tokens)
    return total / n


def sample_replay_tokens(tokenizer, pool, device):
    picks = [pool[i] for i in np.random.choice(len(pool), REPLAY_BATCH, replace=False)]
    return make_batch(tokenizer, picks, device)


def main():
    print("=" * 70)
    print("ABLATION: pure / EWC-only / replay-only / EWC+replay")
    print("=" * 70)

    torch.manual_seed(123)
    np.random.seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Data ----
    print("\n[1/4] Loading podcast directory topics...")
    base_path = r"C:\Users\CPXru\Music\playlist-admin\podcasts"
    texts_a, texts_b, n_dirs_a, n_dirs_b = load_grouped_texts(base_path, max_files_per_group=25)
    print(f"  Dir groups: A={n_dirs_a}, B={n_dirs_b}")
    print(f"  Texts: A={len(texts_a)}, B={len(texts_b)}")

    tok = CharTokenizer()
    tok.fit(texts_a + texts_b)
    print(f"  Vocab: {tok.vocab_size}")

    topic_a = create_chunks(texts_a, chunk_len=128)
    topic_b = create_chunks(texts_b, chunk_len=128)
    np.random.shuffle(topic_a)
    np.random.shuffle(topic_b)
    print(f"  Chunks: A={len(topic_a)}, B={len(topic_b)}")

    # ---- Shared pretraining ----
    print("\n[2/4] Shared pretraining on Topic A...")
    common = OnlineLLMV3(
        vocab_size=tok.vocab_size, embed_dim=128, hidden_dim=256, n_layers=4, n_heads=4
    ).to(device)
    optimizer = torch.optim.Adam(common.parameters(), lr=PRETRAIN_LR)
    common.train()
    for epoch in range(PRETRAIN_EPOCHS):
        total = 0.0
        n_batches = 0
        for i in range(0, len(topic_a) - PRETRAIN_BATCH, PRETRAIN_BATCH):
            padded = make_batch(tok, topic_a[i:i + PRETRAIN_BATCH], device)
            total += common.pretrain_step(padded, optimizer)
            n_batches += 1
        print(f"  Epoch {epoch+1}: loss={total/max(n_batches,1):.4f}")

    state = {k: v.detach().cpu().clone() for k, v in common.state_dict().items()}

    # ---- Four identical-start models ----
    print("\n[3/4] Training four conditions from the same checkpoint...")
    conditions = {}
    for name in ["pure", "ewc_only", "replay_only", "ewc_replay"]:
        model = OnlineLLMV3(
            vocab_size=tok.vocab_size, embed_dim=128, hidden_dim=256, n_layers=4, n_heads=4
        ).to(device)
        model.load_state_dict(state)
        conditions[name] = {
            "model": model,
            "opt": torch.optim.Adam(model.parameters(), lr=ONLINE_LR),
        }

    loss_a0 = eval_loss(conditions["pure"]["model"], tok, topic_a, device)
    loss_b0 = eval_loss(conditions["pure"]["model"], tok, topic_b, device)
    print(f"\n  Shared start: A={loss_a0:.4f}, B={loss_b0:.4f}")

    # Fisher for EWC conditions
    conditions["ewc_only"]["model"].compute_fisher(topic_a, tok, n_samples=FISHER_SAMPLES)
    conditions["ewc_replay"]["model"].compute_fisher(topic_a, tok, n_samples=FISHER_SAMPLES)
    print("  Fisher computed for EWC conditions")

    results = {"start": {"a": loss_a0, "b": loss_b0}, "conditions": {}}

    # Pure fine-tuning
    print("\n  Condition: pure fine-tuning...")
    m = conditions["pure"]["model"]
    m.train()
    for epoch in range(ONLINE_EPOCHS):
        total = 0.0
        for chunk in topic_b[:ONLINE_CHUNKS]:
            tokens = tok.encode(chunk, return_tensors="pt").to(device)
            total += m.pretrain_step(tokens, conditions["pure"]["opt"])
        if (epoch + 1) % 2 == 0:
            print(f"    Epoch {epoch+1}: loss={total/ONLINE_CHUNKS:.4f}")

    # EWC only (no explicit old replay; clear retrieval memory to isolate EWC)
    print("\n  Condition: EWC-only...")
    m = conditions["ewc_only"]["model"]
    m.train()
    for epoch in range(ONLINE_EPOCHS):
        total = 0.0
        for chunk in topic_b[:ONLINE_CHUNKS]:
            tokens = tok.encode(chunk, return_tensors="pt").to(device)
            task_l, _, _ = m.online_step(tokens, conditions["ewc_only"]["opt"], None, EWC_LAMBDA)
            m.memory = None
            total += task_l
        if (epoch + 1) % 2 == 0:
            print(f"    Epoch {epoch+1}: loss={total/ONLINE_CHUNKS:.4f}")

    # Replay only (no Fisher/EWC)
    print("\n  Condition: replay-only...")
    m = conditions["replay_only"]["model"]
    m.train()
    for epoch in range(ONLINE_EPOCHS):
        total = 0.0
        for chunk in topic_b[:ONLINE_CHUNKS]:
            tokens = tok.encode(chunk, return_tensors="pt").to(device)
            replay = sample_replay_tokens(tok, topic_a, device)
            task_l, _, _ = m.online_step(tokens, conditions["replay_only"]["opt"], replay, 0)
            total += task_l
        if (epoch + 1) % 2 == 0:
            print(f"    Epoch {epoch+1}: loss={total/ONLINE_CHUNKS:.4f}")

    # EWC + replay
    print("\n  Condition: EWC+replay...")
    m = conditions["ewc_replay"]["model"]
    m.train()
    for epoch in range(ONLINE_EPOCHS):
        total = 0.0
        for chunk in topic_b[:ONLINE_CHUNKS]:
            tokens = tok.encode(chunk, return_tensors="pt").to(device)
            replay = sample_replay_tokens(tok, topic_a, device)
            task_l, _, _ = m.online_step(tokens, conditions["ewc_replay"]["opt"], replay, EWC_LAMBDA)
            total += task_l
        if (epoch + 1) % 2 == 0:
            print(f"    Epoch {epoch+1}: loss={total/ONLINE_CHUNKS:.4f}")

    # ---- Evaluate all ----
    print("\n[4/4] Evaluating all conditions...")
    for name, cond in conditions.items():
        a = eval_loss(cond["model"], tok, topic_a, device)
        b = eval_loss(cond["model"], tok, topic_b, device)
        mem = cond["model"].memory.count if cond["model"].memory is not None else 0
        results["conditions"][name] = {
            "a": a,
            "b": b,
            "a_change": a / max(loss_a0, 1e-8),
            "b_gain": (loss_b0 - b) / max(loss_b0, 1e-8),
            "memory": mem,
        }
        print(f"  {name:10s}: A={a:.4f}, B={b:.4f}")

    Path("./data").mkdir(parents=True, exist_ok=True)
    with open("./data/ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to data/ablation_results.json")


if __name__ == "__main__":
    main()
