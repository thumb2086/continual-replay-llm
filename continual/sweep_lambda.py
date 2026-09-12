"""EWC lambda sweep with old-topic replay fixed."""

import torch
import numpy as np
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import sys as _sys_root; _sys_root.path.insert(0, ".")  # run-from-root: core lives at repo root
from train_v3 import OnlineLLMV3, CharTokenizer, load_grouped_texts, create_chunks
from compare_ablation import (
    make_batch,
    eval_loss,
    sample_replay_tokens,
    PRETRAIN_EPOCHS,
    PRETRAIN_BATCH,
    PRETRAIN_LR,
    ONLINE_EPOCHS,
    ONLINE_CHUNKS,
    ONLINE_LR,
    FISHER_SAMPLES,
)

LAMBDAS = [0, 100, 500, 2000, 5000, 20000]


def main():
    print("=" * 70)
    print("EWC LAMBDA SWEEP")
    print("=" * 70)

    torch.manual_seed(123)
    np.random.seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Lambdas: {LAMBDAS}")

    # ---- Data ----
    base_path = r"C:\Users\CPXru\Music\playlist-admin\podcasts"
    texts_a, texts_b, n_dirs_a, n_dirs_b = load_grouped_texts(base_path, max_files_per_group=25)
    print(f"Dir groups: A={n_dirs_a}, B={n_dirs_b}")
    print(f"Texts: A={len(texts_a)}, B={len(texts_b)}")

    tok = CharTokenizer()
    tok.fit(texts_a + texts_b)
    print(f"Vocab: {tok.vocab_size}")

    topic_a = create_chunks(texts_a, chunk_len=128)
    topic_b = create_chunks(texts_b, chunk_len=128)
    np.random.shuffle(topic_a)
    np.random.shuffle(topic_b)
    print(f"Chunks: A={len(topic_a)}, B={len(topic_b)}")

    # ---- Shared pretraining ----
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
        print(f"Pretrain epoch {epoch+1}: loss={total/max(n_batches,1):.4f}")

    state = {k: v.detach().cpu().clone() for k, v in common.state_dict().items()}
    start_a = eval_loss(common, tok, topic_a, device)
    start_b = eval_loss(common, tok, topic_b, device)
    print(f"\nShared start: A={start_a:.4f}, B={start_b:.4f}")

    results = {"start": {"a": start_a, "b": start_b}, "lambdas": {}}

    # ---- Sweep ----
    for lam in LAMBDAS:
        print(f"\n--- Lambda {lam} ---")
        torch.manual_seed(888)
        np.random.seed(777)

        model = OnlineLLMV3(
            vocab_size=tok.vocab_size, embed_dim=128, hidden_dim=256, n_layers=4, n_heads=4
        ).to(device)
        model.load_state_dict(state)

        if lam > 0:
            model.compute_fisher(topic_a, tok, n_samples=FISHER_SAMPLES)

        opt = torch.optim.Adam(model.parameters(), lr=ONLINE_LR)
        model.train()

        final_task = 0.0
        for epoch in range(ONLINE_EPOCHS):
            total_task = 0.0
            for chunk in topic_b[:ONLINE_CHUNKS]:
                tokens = tok.encode(chunk, return_tensors="pt").to(device)
                replay = sample_replay_tokens(tok, topic_a, device)
                task_l, _, _ = model.online_step(tokens, opt, replay, lam)
                total_task += task_l
            final_task = total_task / ONLINE_CHUNKS
            if (epoch + 1) % 2 == 0:
                print(f"  Epoch {epoch+1}: task={final_task:.4f}")

        a = eval_loss(model, tok, topic_a, device)
        b = eval_loss(model, tok, topic_b, device)
        mem = model.memory.count if model.memory is not None else 0
        results["lambdas"][str(lam)] = {
            "a": a,
            "b": b,
            "a_change": a / max(start_a, 1e-8),
            "b_gain": (start_b - b) / max(start_b, 1e-8),
            "final_task": final_task,
            "memory": mem,
        }
        print(f"  Result: A={a:.4f}, B={b:.4f}")

    Path("./data").mkdir(parents=True, exist_ok=True)
    with open("./data/lambda_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to data/lambda_sweep.json")


if __name__ == "__main__":
    main()
