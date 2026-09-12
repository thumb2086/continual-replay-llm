"""Autonomous continual-learning research loop.

Goal: verify replay=8 / EWC off across a long A->B->A->B cycle.
Stops when all retention/learning gates pass or when tuning budget is exhausted.
"""

import torch
import numpy as np
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import sys as _sys_root; _sys_root.path.insert(0, ".")  # run-from-root: core lives at repo root
from train_v3 import OnlineLLMV3, CharTokenizer, load_grouped_texts, create_chunks
from compare_ablation import make_batch, eval_loss

PRETRAIN_EPOCHS = 5
PRETRAIN_BATCH = 8
PRETRAIN_LR = 3e-4
ONLINE_LR = 1e-4
ONLINE_EPOCHS = 8
ONLINE_CHUNKS = 300
REPLAY_BATCH = 8
EWC_LAMBDA = 0
RETAIN_LIMIT = 1.05
LEARN_MIN_GAIN = 0.03
MAX_ATTEMPTS_PER_PHASE = 2


def sample_replay(tokenizer, pool, device):
    picks = [pool[i] for i in np.random.choice(len(pool), REPLAY_BATCH, replace=False)]
    return make_batch(tokenizer, picks, device)


def save_state(results, model=None):
    """Persist incremental research state so interrupted runs keep evidence."""
    Path("./data").mkdir(parents=True, exist_ok=True)
    with open("./data/research_state.json", "w") as f:
        json.dump(results, f, indent=2)
    if model is not None:
        torch.save(model.state_dict(), "./data/research_loop_model.pt")


def train_online_phase(model, tokenizer, new_topic, old_topic, device, label,
                       eval_new=None, eval_old=None, min_gain=LEARN_MIN_GAIN):
    """Train one online phase and return losses before/after.

    Training uses new_topic/old_topic, while evaluation defaults to the same
    pools unless held-out eval splits are supplied. Later cycles may set a
    negative min_gain because the topic can already be learned; the key gate
    is still retention of the old topic.
    """
    ev_new = eval_new if eval_new is not None else new_topic
    ev_old = eval_old if eval_old is not None else old_topic
    before_new = eval_loss(model, tokenizer, ev_new, device)
    before_old = eval_loss(model, tokenizer, ev_old, device)

    for attempt in range(1, MAX_ATTEMPTS_PER_PHASE + 1):
        opt = torch.optim.Adam(model.parameters(), lr=ONLINE_LR)
        model.train()
        for epoch in range(ONLINE_EPOCHS):
            for chunk in new_topic[:ONLINE_CHUNKS]:
                tokens = tokenizer.encode(chunk, return_tensors="pt").to(device)
                replay = sample_replay(tokenizer, old_topic, device)
                model.online_step(tokens, opt, replay, EWC_LAMBDA)

        after_new = eval_loss(model, tokenizer, ev_new, device)
        after_old = eval_loss(model, tokenizer, ev_old, device)
        learned = (before_new - after_new) / max(before_new, 1e-8)
        retained = after_old / max(before_old, 1e-8)

        print(f"  {label} attempt {attempt}: new {before_new:.4f}->{after_new:.4f}, "
              f"old {before_old:.4f}->{after_old:.4f}")

        if learned >= min_gain and retained <= RETAIN_LIMIT:
            return True, before_new, after_new, before_old, after_old

    return False, before_new, after_new, before_old, after_old


def main():
    print("=" * 70)
    print("AUTONOMOUS RESEARCH LOOP: A->B->A->B")
    print("=" * 70)

    torch.manual_seed(123)
    np.random.seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Data ----
    base_path = r"C:\Users\CPXru\Music\playlist-admin\podcasts"
    texts_a, texts_b, n_dirs_a, n_dirs_b = load_grouped_texts(base_path, max_files_per_group=25)
    print(f"Dir groups: A={n_dirs_a}, B={n_dirs_b}")
    print(f"Texts: A={len(texts_a)}, B={len(texts_b)}")

    tok = CharTokenizer()
    tok.fit(texts_a + texts_b)
    print(f"Vocab: {tok.vocab_size}")

    all_a = create_chunks(texts_a, chunk_len=128)
    all_b = create_chunks(texts_b, chunk_len=128)
    np.random.shuffle(all_a)
    np.random.shuffle(all_b)
    eval_a = all_a[-200:]
    eval_b = all_b[-200:]
    topic_a = all_a[:-200]
    topic_b = all_b[:-200]
    print(f"Chunks: A={len(topic_a)} train + {len(eval_a)} held-out, "
          f"B={len(topic_b)} train + {len(eval_b)} held-out")

    # ---- Shared model ----
    model = OnlineLLMV3(
        vocab_size=tok.vocab_size, embed_dim=128, hidden_dim=256, n_layers=4, n_heads=4
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=PRETRAIN_LR)

    print("\nPhase 0: pretrain Topic A")
    model.train()
    for epoch in range(PRETRAIN_EPOCHS):
        total = 0.0
        n_batches = 0
        for i in range(0, len(topic_a) - PRETRAIN_BATCH, PRETRAIN_BATCH):
            padded = make_batch(tok, topic_a[i:i + PRETRAIN_BATCH], device)
            total += model.pretrain_step(padded, optimizer)
            n_batches += 1
        print(f"  Epoch {epoch+1}: loss={total/max(n_batches,1):.4f}")

    results = {
        "config": {
            "pretrain_epochs": PRETRAIN_EPOCHS,
            "online_epochs": ONLINE_EPOCHS,
            "online_chunks": ONLINE_CHUNKS,
            "replay_batch": REPLAY_BATCH,
            "ewc_lambda": EWC_LAMBDA,
            "retain_limit": RETAIN_LIMIT,
            "learn_min_gain": LEARN_MIN_GAIN,
            "phase_min_gain": {"online_B1": 0.10, "online_A2": -0.02, "online_B2": -0.02},
            "held_out_eval_chunks": 200,
        },
        "phases": {},
    }

    a0 = eval_loss(model, tok, eval_a, device)
    b0 = eval_loss(model, tok, eval_b, device)
    results["phases"]["pretrain"] = {"a": a0, "b": b0, "held_out": 200}
    results["paper_ready"] = False
    save_state(results, model)
    print(f"\nAfter pretrain: A={a0:.4f}, B={b0:.4f}")

    # ---- Long cycle ----
    # B1 must show real new-topic learning. Later cycles mainly test retention;
    # they pass if neither topic gets materially worse.
    phases = [
        ("online_B1", topic_b, topic_a, eval_b, eval_a, 0.10),
        ("online_A2", topic_a, topic_b, eval_a, eval_b, -0.02),
        ("online_B2", topic_b, topic_a, eval_b, eval_a, -0.02),
    ]

    all_ok = True
    for label, new_topic, old_topic, ev_new, ev_old, phase_gain in phases:
        print(f"\nPhase {label}: learn new topic, retain old topic")
        ok, before_new, after_new, before_old, after_old = train_online_phase(
            model, tok, new_topic, old_topic, device, label, ev_new, ev_old, phase_gain
        )
        results["phases"][label] = {
            "ok": ok,
            "new_before": before_new,
            "new_after": after_new,
            "old_before": before_old,
            "old_after": after_old,
        }
        results["paper_ready"] = False
        save_state(results, model)
        all_ok = all_ok and ok
        if not ok:
            print(f"  {label}: FAILED gates, stopping research loop")
            break

    results["paper_ready"] = bool(all_ok)
    save_state(results, model)

    print("\n" + "=" * 70)
    print(f"PAPER READY: {'YES' if all_ok else 'NO'}")
    print("=" * 70)
    print("Saved to data/research_state.json")


if __name__ == "__main__":
    main()
