"""Online LLM V2 - Stability-Plasticity Balance.

Key innovation: A controller that decides WHEN to learn and WHEN to protect.
- High novelty -> Learn (backprop)
- Low novelty -> Protect (Hebbian only)

This mimics the brain's attention mechanism for learning.
"""

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


# ============================================================
# Memory Bank
# ============================================================
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


# ============================================================
# Novelty Detector (Decides WHEN to learn)
# ============================================================
class NoveltyDetector(nn.Module):
    """Learns to detect novel inputs that are worth learning from."""

    def __init__(self, dim=128, threshold=0.5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.threshold = threshold
        self.history = []

    def forward(self, x):
        """Returns novelty score (0=familiar, 1=novel)."""
        return self.encoder(x)

    def is_novel(self, x):
        """Decide if input is novel enough to learn from."""
        with torch.no_grad():
            score = self.forward(x).item()
        self.history.append(score)
        return score > self.threshold


# ============================================================
# Hebbian Layer (Protects old knowledge)
# ============================================================
class HebbianLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.lr = 0.001
        self.decay_rate = 0.999

    def forward(self, x, update=False):
        output = x @ self.weight
        if update:
            with torch.no_grad():
                hu = torch.outer(x.mean(0), output.mean(0))
                hu = hu / (hu.norm() + 1e-8)
                self.weight.data += self.lr * hu
                self.weight.data *= self.decay_rate
        return output


# ============================================================
# Online LLM V2
# ============================================================
class OnlineLLMV2(nn.Module):
    """Transformer with novelty-controlled online learning."""

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, n_layers=4, n_heads=4):
        super().__init__()
        self.embed_dim = embed_dim

        # Core transformer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads,
                                           dim_feedforward=hidden_dim, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(embed_dim, vocab_size)

        # CortexFlow components
        self.hebbian = HebbianLayer(embed_dim)
        self.novelty = NoveltyDetector(embed_dim)
        self.memory = None

        # Plasticity control
        self.plasticity = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def _init_memory(self):
        if self.memory is None:
            device = next(self.parameters()).device
            self.memory = MemoryBank(capacity=50000, dim=self.embed_dim, device=device)

    def forward(self, x, mode="inference"):
        """Forward pass.

        Modes:
        - "train": standard backprop training
        - "inference": use memory + novelty detection
        - "online": novelty-controlled online learning
        """
        self._init_memory()
        batch, seq = x.shape
        h = self.embedding(x)

        # Memory retrieval
        if mode != "train" and self.memory.count > 0:
            query = h.mean(1)
            mv, _ = self.memory.retrieve(query.squeeze(0))
            if mv is not None:
                ctx = mv.mean(0).unsqueeze(0).unsqueeze(0).expand(batch, seq, -1)
                h = h + 0.1 * ctx

        # Transformer
        h = self.transformer(h)

        # Novelty detection
        h_mean = h.mean(1)
        novelty_score = self.novelty(h_mean)

        # Plasticity control
        plasticity = self.plasticity(h_mean)

        if mode == "online":
            # Hebbian update (always, protects old knowledge)
            hh = self.hebbian(h_mean, update=True)
            h = h + hh.unsqueeze(1) * 0.1

            # Novelty-gated backprop update
            if novelty_score.mean() > self.novelty.threshold:
                # High novelty -> allow backprop-like update via plasticity
                h = h + plasticity.unsqueeze(1) * h

        logits = self.head(h)
        return logits, novelty_score, plasticity

    def pretrain_step(self, tokens, optimizer):
        """Standard backprop training."""
        inp = tokens[:, :-1]
        tgt = tokens[:, 1:]
        logits, _, _ = self.forward(inp, mode="train")
        min_len = min(logits.shape[1], tgt.shape[1])
        loss = F.cross_entropy(logits[:, :min_len].reshape(-1, logits.size(-1)),
                                tgt[:, :min_len].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def online_step(self, tokens, lr=0.0001):
        """Novelty-controlled online learning."""
        inp = tokens[:, :-1]
        tgt = tokens[:, 1:]
        logits, novelty, plasticity = self.forward(inp, mode="online")
        min_len = min(logits.shape[1], tgt.shape[1])
        loss = F.cross_entropy(logits[:, :min_len].reshape(-1, logits.size(-1)),
                                tgt[:, :min_len].reshape(-1))

        # Store in memory if novel
        with torch.no_grad():
            h = self.embedding(inp).mean(1).squeeze(0)
            is_novel = novelty.mean() > self.novelty.threshold
            importance = novelty.mean().item() if is_novel else 0.1
            self.memory.store(h, h, importance=importance)

            # Update novelty detector
            if is_novel:
                self.novelty.threshold = max(0.3, self.novelty.threshold * 0.999)

        return loss.item(), novelty.mean().item(), plasticity.mean().item()

    @torch.no_grad()
    def evaluate_loss(self, tokens):
        self.eval()
        inp = tokens[:, :-1]
        tgt = tokens[:, 1:]
        logits, _, _ = self.forward(inp, mode="inference")
        min_len = min(logits.shape[1], tgt.shape[1])
        loss = F.cross_entropy(logits[:, :min_len].reshape(-1, logits.size(-1)),
                                tgt[:, :min_len].reshape(-1))
        self.train()
        return loss.item()


# ============================================================
# Tokenizer
# ============================================================
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


# ============================================================
# Data
# ============================================================
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


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("ONLINE LLM V2 - Stability-Plasticity Balance")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("\n[1/6] Loading data...")
    base_path = r"C:\Users\CPXru\Music\playlist-admin\podcasts"
    texts = load_data(base_path, max_files=50)
    print(f"  Loaded {len(texts)} texts")

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
    model = OnlineLLMV2(
        vocab_size=tok.vocab_size,
        embed_dim=128,
        hidden_dim=256,
        n_layers=4,
        n_heads=4,
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    # Optimizer (only for pretrain)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    def eval_loss(data):
        total = 0
        n = min(50, len(data))
        for i in range(n):
            tokens = tok.encode(data[i], return_tensors="pt").to(device)
            total += model.evaluate_loss(tokens)
        return total / n

    # ---- Phase 1: Pretrain on A ----
    print("\n[3/6] Phase 1: Backprop pretrain on Topic A...")
    model.train()
    batch_size = 8
    for epoch in range(5):
        total = 0
        n_batches = 0
        for i in range(0, len(topic_a) - batch_size, batch_size):
            batch = topic_a[i:i + batch_size]
            tokens_list = [tok.encode(c, return_tensors="pt").to(device) for c in batch]
            max_len = max(t.shape[1] for t in tokens_list)
            padded = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
            for j, t in enumerate(tokens_list):
                padded[j, :t.shape[1]] = t.squeeze(0)
            loss = model.pretrain_step(padded, optimizer)
            total += loss
            n_batches += 1
        print(f"  Epoch {epoch+1}: loss={total/n_batches:.4f}")

    loss_a0 = eval_loss(topic_a)
    loss_b0 = eval_loss(topic_b)
    print(f"\n  After pretrain: A={loss_a0:.4f}, B={loss_b0:.4f}")

    # ---- Phase 2: Online learn A ----
    print("\n[4/6] Phase 2: Online learn Topic A...")
    model.train()
    for epoch in range(5):
        total_loss = 0
        total_novelty = 0
        total_plasticity = 0
        for chunk in topic_a[:500]:
            tokens = tok.encode(chunk, return_tensors="pt").to(device)
            loss, nov, plas = model.online_step(tokens)
            total_loss += loss
            total_novelty += nov
            total_plasticity += plas
        n = 500
        print(f"  Epoch {epoch+1}: loss={total_loss/n:.4f} novelty={total_novelty/n:.4f} plasticity={total_plasticity/n:.4f}")

    loss_a1 = eval_loss(topic_a)
    loss_b1 = eval_loss(topic_b)
    print(f"\n  After online A: A={loss_a1:.4f}, B={loss_b1:.4f}")

    # ---- Phase 3: Online learn B ----
    print("\n[5/6] Phase 3: Online learn Topic B (test forgetting)...")
    model.train()
    for epoch in range(5):
        total_loss = 0
        total_novelty = 0
        total_plasticity = 0
        for chunk in topic_b[:500]:
            tokens = tok.encode(chunk, return_tensors="pt").to(device)
            loss, nov, plas = model.online_step(tokens)
            total_loss += loss
            total_novelty += nov
            total_plasticity += plas
        n = 500
        print(f"  Epoch {epoch+1}: loss={total_loss/n:.4f} novelty={total_novelty/n:.4f} plasticity={total_plasticity/n:.4f}")

    loss_a2 = eval_loss(topic_a)
    loss_b2 = eval_loss(topic_b)
    print(f"\n  After online B: A={loss_a2:.4f}, B={loss_b2:.4f}")

    # ---- Verdict ----
    print("\n[6/6] Verdict")
    print("=" * 70)

    a_preserved = loss_a2 <= loss_a1 * 1.2
    b_learned = loss_b2 < loss_b0 * 0.9

    print(f"  A preserved: {'YES' if a_preserved else 'NO'} ({loss_a1:.4f} -> {loss_a2:.4f})")
    print(f"  B learned:   {'YES' if b_learned else 'NO'} ({loss_b0:.4f} -> {loss_b2:.4f})")

    if a_preserved and b_learned:
        print("\n  SUCCESS!")
    elif a_preserved:
        print("\n  PARTIAL: No forgetting")
    else:
        print("\n  CATASTROPHIC FORGETTING")

    # Save
    Path("./data").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "./data/v2_model.pt")
    results = {
        "params": params,
        "topic_a": {"pretrain": loss_a0, "online_a": loss_a1, "online_b": loss_a2},
        "topic_b": {"pretrain": loss_b0, "online_a": loss_b1, "online_b": loss_b2},
        "a_preserved": a_preserved, "b_learned": b_learned,
        "memory": model.memory.count,
        "novelty_threshold": model.novelty.threshold,
    }
    with open("./data/v2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to data/v2_results.json")


if __name__ == "__main__":
    main()
