"""Baseline comparison: pure fine-tuning vs EWC + replay."""

import torch
import torch.nn.functional as F
import numpy as np
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from train_v3 import OnlineLLMV3, CharTokenizer, load_grouped_texts, create_chunks


def make_batch(tokenizer, chunks, device):
    tokens_list = [tokenizer.encode(c, return_tensors="pt").to(device) for c in chunks]
    max_len = max(t.shape[1] for t in tokens_list)
    padded = torch.zeros(len(tokens_list), max_len, dtype=torch.long, device=device)
    for j, t in enumerate(tokens_list):
        padded[j, : t.shape[1]] = t.squeeze(0)
    return padded


def eval_loss(model, tokenizer, data, device=None, n=50):
    if device is None:
        device = next(model.parameters()).device
    total = 0
    m = min(n, len(data))
    for i in range(m):
        tokens = tokenizer.encode(data[i], return_tensors="pt").to(device)
        total += model.evaluate_loss(tokens)
    return total / m


def sample_replay_tokens(tokenizer, pool, device, batch_size=2, seq_len=128):
    if not pool:
        return None
    picks = [pool[i] for i in np.random.choice(len(pool), min(batch_size, len(pool)), replace=False)]
    return make_batch(tokenizer, [s[:seq_len] for s in picks], device)


def main():
    print("=" * 70)
    print("BASELINE COMPARISON: Pure Fine-tune vs EWC + Replay")
    print("=" * 70)

    torch.manual_seed(123)
    np.random.seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Data ----
    print("\n[1/5] Loading podcast directory topics...")
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

    # ---- Models with identical starting weights ----
    print("\n[2/5] Creating two identical models...")
    common = OnlineLLMV3(
        vocab_size=tok.vocab_size,
        embed_dim=128,
        hidden_dim=256,
        n_layers=4,
        n_heads=4,
    ).to(device)

    optimizer = torch.optim.Adam(common.parameters(), lr=3e-4)
    print("\n[3/5] Shared pretraining on Topic A...")
    common.train()
    batch_size = 8
    for epoch in range(5):
        total = 0
        n_batches = 0
        for i in range(0, len(topic_a) - batch_size, batch_size):
            batch = topic_a[i:i + batch_size]
            padded = make_batch(tok, batch, device)
            total += common.pretrain_step(padded, optimizer)
            n_batches += 1
        print(f"  Epoch {epoch+1}: loss={total/max(n_batches,1):.4f}")

    state = {k: v.detach().cpu().clone() for k, v in common.state_dict().items()}

    baseline = OnlineLLMV3(
        vocab_size=tok.vocab_size, embed_dim=128, hidden_dim=256, n_layers=4, n_heads=4
    ).to(device)
    ewc_model = OnlineLLMV3(
        vocab_size=tok.vocab_size, embed_dim=128, hidden_dim=256, n_layers=4, n_heads=4
    ).to(device)
    baseline.load_state_dict(state)
    ewc_model.load_state_dict(state)

    loss_a0 = eval_loss(ewc_model, tok, topic_a, device)
    loss_b0 = eval_loss(ewc_model, tok, topic_b)
    print(f"\n  Shared start: A={loss_a0:.4f}, B={loss_b0:.4f}")

    # ---- Baseline: pure fine-tune on B ----
    print("\n[4/5] Baseline: pure fine-tune on Topic B...")
    baseline.train()
    baseline_opt = torch.optim.Adam(baseline.parameters(), lr=1e-4)
    for epoch in range(10):
        total = 0
        for chunk in topic_b[:500]:
            tokens = tok.encode(chunk, return_tensors="pt").to(device)
            total += baseline.pretrain_step(tokens, baseline_opt)
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}: loss={total/500:.4f}")

    base_a = eval_loss(baseline, tok, topic_a, device)
    base_b = eval_loss(baseline, tok, topic_b, device)
    print(f"  Baseline after B: A={base_a:.4f}, B={base_b:.4f}")

    # ---- EWC model: Fisher + online B with replay ----
    print("\n[5/5] EWC model: Fisher + online B with replay...")
    ewc_model.compute_fisher(topic_a, tok, n_samples=200)
    ewc_model.train()
    ewc_opt = torch.optim.Adam(ewc_model.parameters(), lr=1e-4)

    for epoch in range(10):
        total_task = 0
        total_ewc = 0
        total_replay = 0
        for chunk in topic_b[:500]:
            tokens = tok.encode(chunk, return_tensors="pt").to(device)
            replay = sample_replay_tokens(tok, topic_a, device, batch_size=2)
            task_l, ewc_l, replay_l = ewc_model.online_step(tokens, ewc_opt, replay, lambda_ewc=5000)
            total_task += task_l
            total_ewc += ewc_l
            total_replay += replay_l
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}: task={total_task/500:.4f} ewc={total_ewc/500:.4f} replay={total_replay/500:.4f}")

    ewc_a = eval_loss(ewc_model, tok, topic_a, device)
    ewc_b = eval_loss(ewc_model, tok, topic_b, device)
    print(f"  EWC after B: A={ewc_a:.4f}, B={ewc_b:.4f}")

    # ---- Results ----
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"  Start:                 A={loss_a0:.4f}, B={loss_b0:.4f}")
    print(f"  Baseline after B:      A={base_a:.4f}, B={base_b:.4f}")
    print(f"  EWC+replay after B:    A={ewc_a:.4f}, B={ewc_b:.4f}")

    baseline_forget = base_a / max(loss_a0, 1e-8)
    ewc_forget = ewc_a / max(loss_a0, 1e-8)
    baseline_b_gain = (loss_b0 - base_b) / max(loss_b0, 1e-8)
    ewc_b_gain = (loss_b0 - ewc_b) / max(loss_b0, 1e-8)

    print(f"\n  Baseline A change: {loss_a0:.4f} -> {base_a:.4f} ({baseline_forget:.3f}x)")
    print(f"  EWC A change:      {loss_a0:.4f} -> {ewc_a:.4f} ({ewc_forget:.3f}x)")
    print(f"  Baseline B gain:   {baseline_b_gain*100:.2f}%")
    print(f"  EWC B gain:        {ewc_b_gain*100:.2f}%")

    results = {
        "start": {"a": loss_a0, "b": loss_b0},
        "baseline": {"a": base_a, "b": base_b},
        "ewc_replay": {"a": ewc_a, "b": ewc_b},
        "a_change": {"baseline": baseline_forget, "ewc": ewc_forget},
        "b_gain": {"baseline": baseline_b_gain, "ewc": ewc_b_gain},
        "ewc_memory": ewc_model.memory.count,
    }
    Path("./data").mkdir(parents=True, exist_ok=True)
    with open("./data/baseline_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to data/baseline_comparison.json")


if __name__ == "__main__":
    main()
