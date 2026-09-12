"""Hybrid Training: Backprop pretrain + Hebbian online learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import glob
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')


class MemoryBank:
    def __init__(self, capacity=50000, dim=128, device="cpu"):
        self.capacity = capacity
        self.dim = dim
        self.device = device
        self.keys = torch.zeros(capacity, dim, device=device)
        self.values = torch.zeros(capacity, dim, device=device)
        self.importance = torch.zeros(capacity, device=device)
        self.ptr = 0
        self.count = 0

    def store(self, key, value, importance=1.0):
        key = key.to(self.device)[:self.dim]
        value = value.to(self.device)[:self.dim]
        if self.count < self.capacity:
            self.keys[self.ptr] = key
            self.values[self.ptr] = value
            self.importance[self.ptr] = importance
            self.ptr = (self.ptr + 1) % self.capacity
            self.count += 1
        else:
            min_idx = torch.argmin(self.importance)
            if importance > self.importance[min_idx]:
                self.keys[min_idx] = key
                self.values[min_idx] = value
                self.importance[min_idx] = importance

    def retrieve(self, query, k=5):
        if self.count == 0:
            return None, None
        query = query[:self.dim].to(self.device)
        sims = F.cosine_similarity(query.unsqueeze(0), self.keys[:self.count], dim=-1)
        top_k = torch.topk(sims, min(k, self.count))
        return self.values[top_k.indices], top_k.values


class HebbianLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.lr = 0.001
        self.decay_rate = 0.999

    def forward(self, x, update=False):
        output = x @ self.weight
        if update and self.training:
            with torch.no_grad():
                hu = torch.outer(x.mean(0), output.mean(0))
                hu = hu / (hu.norm() + 1e-8)
                self.weight.data += self.lr * hu
                self.weight.data *= self.decay_rate
        return output


class OnlineLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, n_layers=4, n_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads,
                                           dim_feedforward=hidden_dim, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(embed_dim, vocab_size)
        self.hebbian = HebbianLayer(embed_dim)
        self.memory = None

    def _init_memory(self):
        if self.memory is None:
            device = next(self.parameters()).device
            self.memory = MemoryBank(capacity=50000, dim=self.embed_dim, device=device)

    def forward(self, x, use_memory=True, update_hebbian=False):
        self._init_memory()
        batch, seq = x.shape
        h = self.embedding(x)

        if use_memory and self.memory.count > 0:
            query = h.mean(1)
            mv, _ = self.memory.retrieve(query.squeeze(0))
            if mv is not None:
                ctx = mv.mean(0).unsqueeze(0).unsqueeze(0).expand(batch, seq, -1)
                h = h + 0.1 * ctx

        h = self.transformer(h)

        if update_hebbian:
            hm = h.mean(1)
            hh = self.hebbian(hm, update=True)
            h = h + hh.unsqueeze(1) * 0.1

        return self.head(h)

    def pretrain_step(self, tokens, optimizer):
        """Standard backprop training step."""
        inp = tokens[:, :-1]
        tgt = tokens[:, 1:]
        logits = self.forward(inp, use_memory=False, update_hebbian=False)
        min_len = min(logits.shape[1], tgt.shape[1])
        loss = F.cross_entropy(logits[:, :min_len].reshape(-1, logits.size(-1)),
                                tgt[:, :min_len].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def online_step(self, tokens):
        """Hebbian online learning step (no backprop)."""
        inp = tokens[:, :-1]
        tgt = tokens[:, 1:]
        logits = self.forward(inp, use_memory=False, update_hebbian=True)
        min_len = min(logits.shape[1], tgt.shape[1])
        loss = F.cross_entropy(logits[:, :min_len].reshape(-1, logits.size(-1)),
                                tgt[:, :min_len].reshape(-1))
        with torch.no_grad():
            h = self.embedding(inp).mean(1).squeeze(0)
            self.memory.store(h, h, importance=1.0 / (1.0 + loss.item()))
        return loss.item()

    @torch.no_grad()
    def evaluate_loss(self, tokens):
        self.eval()
        inp = tokens[:, :-1]
        tgt = tokens[:, 1:]
        logits = self.forward(inp, use_memory=False, update_hebbian=False)
        min_len = min(logits.shape[1], tgt.shape[1])
        loss = F.cross_entropy(logits[:, :min_len].reshape(-1, logits.size(-1)),
                                tgt[:, :min_len].reshape(-1))
        self.train()
        return loss.item()


class CharTokenizer:
    def __init__(self):
        self.char2idx = {"<PAD>": 0}
        self.idx2char = {0: "<PAD>"}
        self.vocab_size = 1

    def fit(self, texts):
        for t in texts:
            for c in t:
                if c not in self.char2idx:
                    idx = self.vocab_size
                    self.char2idx[c] = idx
                    self.idx2char[idx] = c
                    self.vocab_size += 1

    def encode(self, text, return_tensors=None):
        ids = [self.char2idx.get(c, 0) for c in text]
        if return_tensors == "pt":
            return torch.tensor([ids])
        return ids


def load_data(base_path, max_files=50):
    txt_files = glob.glob(os.path.join(base_path, '**', '*.txt'), recursive=True)
    txt_files = [f for f in txt_files if 'README' not in f]

    texts = []
    for f in txt_files[:max_files]:
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                text = fp.read()
                if len(text) > 100:
                    texts.append(text[:5000])
        except:
            continue

    return texts


def create_chunks(texts, chunk_len=128):
    chunks = []
    for text in texts:
        for i in range(0, len(text) - chunk_len, chunk_len // 2):
            chunk = text[i:i + chunk_len]
            if len(chunk) >= 32:
                chunks.append(chunk)
    return chunks


def main():
    print("=" * 70)
    print("HYBRID TRAINING: Backprop Pretrain + Hebbian Online")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load
    print("\n[1/6] Loading data...")
    base_path = r"C:\Users\CPXru\Music\playlist-admin\podcasts"
    texts = load_data(base_path, max_files=50)
    print(f"  Loaded {len(texts)} texts")

    # Tokenizer
    tok = CharTokenizer()
    tok.fit(texts)
    print(f"  Vocab: {tok.vocab_size}")

    chunks = create_chunks(texts, chunk_len=128)
    np.random.shuffle(chunks)
    split = len(chunks) // 2
    topic_a = chunks[:split]
    topic_b = chunks[split:]
    print(f"  Chunks: {len(chunks)} (A={len(topic_a)}, B={len(topic_b)})")

    # Model
    print("\n[2/6] Creating model...")
    model = OnlineLLM(
        vocab_size=tok.vocab_size,
        embed_dim=128,
        hidden_dim=256,
        n_layers=4,
        n_heads=4,
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    def eval_loss(data):
        total = 0
        n = min(50, len(data))
        for i in range(n):
            tokens = tok.encode(data[i], return_tensors="pt").to(device)
            total += model.evaluate_loss(tokens)
        return total / n

    # ---- Phase 1: Backprop Pretrain on Topic A ----
    print("\n[3/6] Phase 1: Backprop pretrain on Topic A...")
    model.train()
    batch_size = 8
    for epoch in range(5):
        total = 0
        n_batches = 0
        for i in range(0, len(topic_a) - batch_size, batch_size):
            batch = topic_a[i:i + batch_size]
            tokens_list = [tok.encode(c, return_tensors="pt").to(device) for c in batch]

            # Pad and stack
            max_len = max(t.shape[1] for t in tokens_list)
            padded = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
            for j, t in enumerate(tokens_list):
                padded[j, :t.shape[1]] = t.squeeze(0)

            loss = model.pretrain_step(padded, optimizer)
            total += loss
            n_batches += 1

        if (epoch + 1) % 1 == 0:
            print(f"  Epoch {epoch+1}: loss={total/n_batches:.4f}")

    loss_a0 = eval_loss(topic_a)
    loss_b0 = eval_loss(topic_b)
    print(f"\n  After pretrain:")
    print(f"    Topic A: {loss_a0:.4f}")
    print(f"    Topic B: {loss_b0:.4f}")

    # ---- Phase 2: Hebbian Online on Topic A ----
    print("\n[4/6] Phase 2: Hebbian online on Topic A...")
    model.train()
    for epoch in range(5):
        total = 0
        for chunk in topic_a[:500]:
            tokens = tok.encode(chunk, return_tensors="pt").to(device)
            total += model.online_step(tokens)
        if (epoch + 1) % 1 == 0:
            print(f"  Epoch {epoch+1}: loss={total/500:.4f}")

    loss_a1 = eval_loss(topic_a)
    loss_b1 = eval_loss(topic_b)
    print(f"\n  After online A:")
    print(f"    Topic A: {loss_a1:.4f}")
    print(f"    Topic B: {loss_b1:.4f}")

    # ---- Phase 3: Hebbian Online on Topic B ----
    print("\n[5/6] Phase 3: Hebbian online on Topic B (should not forget A)...")
    model.train()
    for epoch in range(5):
        total = 0
        for chunk in topic_b[:500]:
            tokens = tok.encode(chunk, return_tensors="pt").to(device)
            total += model.online_step(tokens)
        if (epoch + 1) % 1 == 0:
            print(f"  Epoch {epoch+1}: loss={total/500:.4f}")

    loss_a2 = eval_loss(topic_a)
    loss_b2 = eval_loss(topic_b)
    print(f"\n  After online B:")
    print(f"    Topic A: {loss_a2:.4f}")
    print(f"    Topic B: {loss_b2:.4f}")

    # ---- Verdict ----
    print("\n[6/6] Verdict")
    print("=" * 70)

    a_preserved = loss_a2 <= loss_a1 * 1.2
    b_learned = loss_b2 < loss_b0 * 0.8

    print(f"  Topic A preserved: {'YES' if a_preserved else 'NO'} ({loss_a1:.4f} -> {loss_a2:.4f})")
    print(f"  Topic B learned:   {'YES' if b_learned else 'NO'} ({loss_b0:.4f} -> {loss_b2:.4f})")

    if a_preserved and b_learned:
        print("\n  SUCCESS: Continual learning without catastrophic forgetting!")
    elif a_preserved:
        print("\n  PARTIAL: No forgetting, B not learned")
    elif b_learned:
        print("\n  CATASTROPHIC FORGETTING: A lost while learning B")
    else:
        print("\n  FAILED: Neither preserved nor learned")

    # Save
    Path("./data").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "./data/hybrid_model.pt")
    results = {
        "params": params, "vocab": tok.vocab_size,
        "topic_a": {"pretrain": loss_a0, "online_a": loss_a1, "online_b": loss_a2},
        "topic_b": {"pretrain": loss_b0, "online_a": loss_b1, "online_b": loss_b2},
        "a_preserved": a_preserved, "b_learned": b_learned,
        "memory": model.memory.count,
    }
    with open("./data/hybrid_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to data/hybrid_results.json")


if __name__ == "__main__":
    main()
