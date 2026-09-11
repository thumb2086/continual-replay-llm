"""Continual Learning Test - Quantitative Evaluation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path


class MemoryBank:
    def __init__(self, capacity=10000, dim=64, device="cpu"):
        self.capacity = capacity
        self.dim = dim
        self.device = device
        self.keys = torch.zeros(capacity, dim, device=device)
        self.values = torch.zeros(capacity, dim, device=device)
        self.importance = torch.zeros(capacity, device=device)
        self.ptr = 0
        self.count = 0

    def store(self, key, value, importance=1.0):
        key = key.to(self.device)
        value = value.to(self.device)
        if self.count < self.capacity:
            self.keys[self.ptr] = key[:self.dim]
            self.values[self.ptr] = value[:self.dim]
            self.importance[self.ptr] = importance
            self.ptr = (self.ptr + 1) % self.capacity
            self.count += 1
        else:
            min_idx = torch.argmin(self.importance)
            if importance > self.importance[min_idx]:
                self.keys[min_idx] = key[:self.dim]
                self.values[min_idx] = value[:self.dim]
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
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, n_layers=2, n_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads,
                                           dim_feedforward=hidden_dim, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(embed_dim, vocab_size)
        self.hebbian = HebbianLayer(embed_dim)
        self.memory = None
        self.update_count = 0

    def _init_memory(self):
        if self.memory is None:
            device = next(self.parameters()).device
            self.memory = MemoryBank(capacity=10000, dim=self.embed_dim, device=device)

    def forward(self, x, use_memory=True, update=True):
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
        if update:
            hm = h.mean(1)
            hh = self.hebbian(hm, update=True)
            h = h + hh.unsqueeze(1) * 0.1

        return self.head(h)

    def learn(self, tokens):
        if tokens.shape[1] < 3:
            return 0.0
        inp = tokens[:, :-1]
        tgt = tokens[:, 1:]
        logits = self.forward(inp, use_memory=False, update=True)
        min_len = min(logits.shape[1], tgt.shape[1])
        loss = F.cross_entropy(logits[:, :min_len].reshape(-1, logits.size(-1)),
                                tgt[:, :min_len].reshape(-1))
        with torch.no_grad():
            h = self.embedding(inp).mean(1).squeeze(0)
            self.memory.store(h, h, importance=1.0 / (1.0 + loss.item()))
        self.update_count += 1
        return loss.item()

    @torch.no_grad()
    def evaluate_loss(self, tokens):
        """Evaluate loss without updating."""
        self.eval()
        inp = tokens[:, :-1]
        tgt = tokens[:, 1:]
        logits = self.forward(inp, use_memory=False, update=False)
        min_len = min(logits.shape[1], tgt.shape[1])
        loss = F.cross_entropy(logits[:, :min_len].reshape(-1, logits.size(-1)),
                                tgt[:, :min_len].reshape(-1))
        self.train()
        return loss.item()


class Tokenizer:
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


def main():
    print("=" * 70)
    print("CONTINUAL LEARNING TEST - Quantitative")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    en = [
        "The cat sat on the mat.",
        "A dog runs in the park.",
        "The sun is bright today.",
        "I love programming.",
        "Machine learning is great.",
        "She reads a book.",
        "He walks to school.",
        "The weather is nice.",
        "We enjoy learning.",
        "The fox jumps high.",
    ]

    cn = [
        "今天天氣很好。",
        "我喜歡學習。",
        "貓坐在墊子上。",
        "狗在公園裡跑。",
        "太陽今天很亮。",
        "我愛寫程式。",
        "她每天讀書。",
        "他走去學校。",
        "外面天氣很好。",
        "我們享受學習。",
    ]

    # Tokenizer
    tok = Tokenizer()
    tok.fit(en + cn)

    # Model
    model = OnlineLLM(vocab_size=tok.vocab_size, embed_dim=64, hidden_dim=128, n_layers=2, n_heads=4).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # Helper
    def measure_loss(texts, label):
        total = 0
        for t in texts:
            tokens = tok.encode(t, return_tensors="pt").to(device)
            total += model.evaluate_loss(tokens)
        return total / len(texts)

    # ---- Baseline ----
    print("\n--- Baseline (untrained) ---")
    en_loss_0 = measure_loss(en, "en")
    cn_loss_0 = measure_loss(cn, "cn")
    print(f"  English loss: {en_loss_0:.4f}")
    print(f"  Chinese loss: {cn_loss_0:.4f}")

    # ---- Phase 1: Learn English ----
    print("\n--- Phase 1: Learning English (20 epochs) ---")
    model.train()
    for epoch in range(20):
        total = 0
        for t in en:
            tokens = tok.encode(t, return_tensors="pt").to(device)
            total += model.learn(tokens)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: loss={total/len(en):.4f}")

    en_loss_1 = measure_loss(en, "en")
    cn_loss_1 = measure_loss(cn, "cn")
    print(f"  English loss: {en_loss_1:.4f} (was {en_loss_0:.4f})")
    print(f"  Chinese loss: {cn_loss_1:.4f} (was {cn_loss_0:.4f})")

    # ---- Phase 2: Learn Chinese ----
    print("\n--- Phase 2: Learning Chinese (20 epochs) ---")
    for epoch in range(20):
        total = 0
        for t in cn:
            tokens = tok.encode(t, return_tensors="pt").to(device)
            total += model.learn(tokens)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: loss={total/len(cn):.4f}")

    en_loss_2 = measure_loss(en, "en")
    cn_loss_2 = measure_loss(cn, "cn")
    print(f"  English loss: {en_loss_2:.4f} (was {en_loss_1:.4f})")
    print(f"  Chinese loss: {cn_loss_2:.4f} (was {cn_loss_1:.4f})")

    # ---- Verdict ----
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    en_preserved = en_loss_2 <= en_loss_1 * 1.2  # Within 20% of before
    cn_learned = cn_loss_2 < cn_loss_0 * 0.8      # At least 20% improvement

    print(f"  English preserved: {'YES' if en_preserved else 'NO'} (loss {en_loss_1:.4f} -> {en_loss_2:.4f})")
    print(f"  Chinese learned:   {'YES' if cn_learned else 'NO'} (loss {cn_loss_0:.4f} -> {cn_loss_2:.4f})")

    if en_preserved and cn_learned:
        print("\n  SUCCESS: No catastrophic forgetting!")
    elif en_preserved:
        print("\n  PARTIAL: English preserved, Chinese not well learned")
    elif cn_learned:
        print("\n  PARTIAL: Chinese learned, English forgotten (catastrophic forgetting)")
    else:
        print("\n  FAILED")

    # Save
    Path("./data").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "./data/continual_model.pt")

    results = {
        "baseline": {"en": en_loss_0, "cn": cn_loss_0},
        "after_english": {"en": en_loss_1, "cn": cn_loss_1},
        "after_chinese": {"en": en_loss_2, "cn": cn_loss_2},
        "en_preserved": en_preserved,
        "cn_learned": cn_learned,
        "total_updates": model.update_count,
    }
    with open("./data/continual_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to data/continual_results.json")


if __name__ == "__main__":
    main()
