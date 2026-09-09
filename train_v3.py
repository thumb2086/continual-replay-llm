"""Online LLM V3 - EWC + Memory Replay.

Key: Use backprop for online learning, but protect old knowledge with EWC.
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
        query = query[:self.dim].reshape(-1).to(self.device)
        sims = F.cosine_similarity(query.unsqueeze(0), self.keys[:self.count], dim=-1).reshape(-1)
        top_k = torch.topk(sims, min(k, self.count))
        idx = top_k.indices.reshape(-1)
        return self.values[idx], top_k.values.reshape(-1)

    def sample(self, n):
        if self.count == 0:
            return None
        indices = np.random.choice(self.count, min(n, self.count), replace=False)
        return self.keys[indices], self.values[indices]


class OnlineLLMV3(nn.Module):
    """Transformer with EWC protection and memory replay."""

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, n_layers=4, n_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads,
                                           dim_feedforward=hidden_dim, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(embed_dim, vocab_size)
        self.memory = None

        # EWC: Fisher information matrix
        self.fisher = {}
        self.optimal_params = {}

    def _init_memory(self):
        if self.memory is None:
            device = next(self.parameters()).device
            self.memory = MemoryBank(capacity=50000, dim=self.embed_dim, device=device)

    def forward(self, x):
        self._init_memory()
        batch, seq = x.shape
        h = self.embedding(x)

        if self.memory.count > 0:
            query = h.float().mean(dim=(0, 1))
            mv, _ = self.memory.retrieve(query)
            if mv is not None:
                ctx = mv.mean(0).unsqueeze(0).unsqueeze(0).expand(batch, seq, -1)
                h = h + 0.1 * ctx.to(h.dtype)

        h = self.transformer(h)
        return self.head(h)

    def compute_fisher(self, data, tokenizer, n_samples=200):
        """Compute Fisher information (importance of each parameter)."""
        self.eval()
        self.fisher = {n: torch.zeros_like(p) for n, p in self.named_parameters()}

        for i in range(min(n_samples, len(data))):
            tokens = tokenizer.encode(data[i], return_tensors="pt").to(next(self.parameters()).device)
            if tokens.shape[1] < 3:
                continue
            inp = tokens[:, :-1]
            tgt = tokens[:, 1:]
            logits = self(inp)
            min_len = min(logits.shape[1], tgt.shape[1])
            loss = F.cross_entropy(logits[:, :min_len].reshape(-1, logits.size(-1)),
                                    tgt[:, :min_len].reshape(-1))
            self.zero_grad()
            loss.backward()
            for n, p in self.named_parameters():
                if p.grad is not None:
                    self.fisher[n] += p.grad.data ** 2

        for n in self.fisher:
            self.fisher[n] /= n_samples

        # Save optimal parameters
        self.optimal_params = {n: p.data.clone() for n, p in self.named_parameters()}
        self.train()

    def ewc_loss(self, lambda_ewc=1000):
        """EWC regularization loss."""
        loss = 0
        for n, p in self.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.optimal_params[n]) ** 2).sum()
        return lambda_ewc * loss

    def pretrain_step(self, tokens, optimizer):
        inp = tokens[:, :-1]
        tgt = tokens[:, 1:]
        logits = self(inp)
        min_len = min(logits.shape[1], tgt.shape[1])
        loss = F.cross_entropy(logits[:, :min_len].reshape(-1, logits.size(-1)),
                                tgt[:, :min_len].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def online_step(self, tokens, optimizer, replay_batch=None, lambda_ewc=1000):
        """Online learning with EWC protection + memory replay."""
        inp = tokens[:, :-1]
        tgt = tokens[:, 1:]
        logits = self(inp)
        min_len = min(logits.shape[1], tgt.shape[1])
        task_loss = F.cross_entropy(logits[:, :min_len].reshape(-1, logits.size(-1)),
                                     tgt[:, :min_len].reshape(-1))

        # EWC regularization
        ewc_loss = self.ewc_loss(lambda_ewc)

        # Memory replay
        replay_loss = torch.tensor(0.0, device=task_loss.device)
        if replay_batch is not None:
            r_inp = replay_batch[:, :-1]
            r_tgt = replay_batch[:, 1:]
            r_logits = self(r_inp)
            r_min = min(r_logits.shape[1], r_tgt.shape[1])
            replay_loss = F.cross_entropy(r_logits[:, :r_min].reshape(-1, r_logits.size(-1)),
                                           r_tgt[:, :r_min].reshape(-1))

        total_loss = task_loss + ewc_loss + 0.5 * replay_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Store in memory
        with torch.no_grad():
            h = self.embedding(inp).mean(1).squeeze(0)
            self.memory.store(h, h, importance=1.0 / (1.0 + task_loss.item()))

        ewc_value = ewc_loss.item() if torch.is_tensor(ewc_loss) else float(ewc_loss)
        replay_value = replay_loss.item() if torch.is_tensor(replay_loss) else float(replay_loss)
        return task_loss.item(), ewc_value, replay_value

    @torch.no_grad()
    def evaluate_loss(self, tokens):
        self.eval()
        inp = tokens[:, :-1]
        tgt = tokens[:, 1:]
        logits = self(inp)
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


def load_grouped_texts(base_path, max_files_per_group=25, max_chars=5000):
    """Load texts grouped by immediate podcast subdirectory.

    Topic A = first half of subdirectories, Topic B = second half.
    This is a directory-based proxy for different podcast series/topics.
    """
    base = Path(base_path)
    subdirs = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name)
    mid = max(1, len(subdirs) // 2)
    group_a_dirs = subdirs[:mid]
    group_b_dirs = subdirs[mid:]

    texts_a, texts_b = [], []
    files_a, files_b = [], []
    for d in group_a_dirs:
        files_a.extend(sorted(d.rglob("*.txt")))
    for d in group_b_dirs:
        files_b.extend(sorted(d.rglob("*.txt")))
    files_a = [f for f in files_a if "README" not in f.name][:max_files_per_group]
    files_b = [f for f in files_b if "README" not in f.name][:max_files_per_group]

    for f in files_a:
        try:
            text = f.read_text(encoding="utf-8")
            if len(text) > 100:
                texts_a.append(text[:max_chars])
        except Exception:
            continue
    for f in files_b:
        try:
            text = f.read_text(encoding="utf-8")
            if len(text) > 100:
                texts_b.append(text[:max_chars])
        except Exception:
            continue

    return texts_a, texts_b, len(group_a_dirs), len(group_b_dirs)


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
    print("ONLINE LLM V3 - EWC + Memory Replay")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load
    print("\n[1/7] Loading data by podcast directory...")
    base_path = r"C:\Users\CPXru\Music\playlist-admin\podcasts"
    texts_a, texts_b, n_dirs_a, n_dirs_b = load_grouped_texts(base_path, max_files_per_group=25)
    print(f"  Directory groups: A={n_dirs_a}, B={n_dirs_b}")
    print(f"  Texts: A={len(texts_a)}, B={len(texts_b)}")
    texts = texts_a + texts_b

    tok = CharTokenizer()
    tok.fit(texts)
    print(f"  Vocab: {tok.vocab_size}")

    topic_a = create_chunks(texts_a, chunk_len=128)
    topic_b = create_chunks(texts_b, chunk_len=128)
    np.random.shuffle(topic_a)
    np.random.shuffle(topic_b)
    print(f"  Chunks: {len(topic_a) + len(topic_b)} (A={len(topic_a)}, B={len(topic_b)})")

    # Model
    print("\n[2/7] Creating model...")
    model = OnlineLLMV3(
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

    def get_replay_batch(pool, batch_size=2, seq_len=128):
        if not pool:
            return None
        samples = [pool[i] for i in np.random.choice(len(pool), min(batch_size, len(pool)), replace=False)]
        ids_list = [tok.encode(s[:seq_len], return_tensors="pt").to(device) for s in samples]
        max_len = max(t.shape[1] for t in ids_list)
        padded = torch.zeros(len(ids_list), max_len, dtype=torch.long, device=device)
        for j, t in enumerate(ids_list):
            padded[j, :t.shape[1]] = t.squeeze(0)
        return padded

    # ---- Phase 1: Pretrain on A ----
    print("\n[3/7] Phase 1: Pretrain on Topic A...")
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

    # Compute Fisher before learning B
    print("\n[4/7] Computing Fisher information...")
    model.compute_fisher(topic_a, tok, n_samples=200)
    print(f"  Fisher computed for {len(model.fisher)} parameters")

    loss_a0 = eval_loss(topic_a)
    loss_b0 = eval_loss(topic_b)
    print(f"  After pretrain: A={loss_a0:.4f}, B={loss_b0:.4f}")

    # ---- Phase 2: Online learn B with EWC ----
    print("\n[5/7] Phase 2: Online learn Topic B with EWC + replay...")
    model.train()
    online_lr = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        total_task = 0
        total_ewc = 0
        total_replay = 0
        n = 0
        for chunk in topic_b[:500]:
            tokens = tok.encode(chunk, return_tensors="pt").to(device)
            replay = get_replay_batch(topic_a, 2)
            task_l, ewc_l, replay_l = model.online_step(tokens, online_lr, replay, lambda_ewc=5000)
            total_task += task_l
            total_ewc += ewc_l
            total_replay += replay_l
            n += 1
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}: task={total_task/n:.4f} ewc={total_ewc/n:.4f} replay={total_replay/n:.4f}")

    loss_a1 = eval_loss(topic_a)
    loss_b1 = eval_loss(topic_b)
    print(f"\n  After online B: A={loss_a1:.4f}, B={loss_b1:.4f}")

    # ---- Phase 3: Online learn A again (test round-trip) ----
    print("\n[6/7] Phase 3: Online learn Topic A again...")
    model.compute_fisher(topic_b, tok, n_samples=200)

    for epoch in range(5):
        total_task = 0
        n = 0
        for chunk in topic_a[:300]:
            tokens = tok.encode(chunk, return_tensors="pt").to(device)
            replay = get_replay_batch(topic_b, 2)
            task_l, _, _ = model.online_step(tokens, online_lr, replay, lambda_ewc=5000)
            total_task += task_l
            n += 1
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}: task={total_task/n:.4f}")

    loss_a2 = eval_loss(topic_a)
    loss_b2 = eval_loss(topic_b)
    print(f"\n  Final: A={loss_a2:.4f}, B={loss_b2:.4f}")

    # ---- Verdict ----
    print("\n[7/7] Verdict")
    print("=" * 70)

    a_preserved = loss_a1 <= loss_a0 * 1.15
    b_learned = loss_b1 < loss_b0 * 0.9
    a_roundtrip = loss_a2 <= loss_a0 * 1.15

    print(f"  A after B:     {'YES' if a_preserved else 'NO'} ({loss_a0:.4f} -> {loss_a1:.4f})")
    print(f"  B learned:     {'YES' if b_learned else 'NO'} ({loss_b0:.4f} -> {loss_b1:.4f})")
    print(f"  A round-trip:  {'YES' if a_roundtrip else 'NO'} ({loss_a0:.4f} -> {loss_a2:.4f})")

    if a_preserved and b_learned:
        print("\n  SUCCESS: No catastrophic forgetting!")
    else:
        print("\n  NEEDS WORK")

    # Save
    Path("./data").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "./data/v3_model.pt")
    results = {
        "params": params,
        "topic_a": {"pretrain": loss_a0, "after_b": loss_a1, "roundtrip": loss_a2},
        "topic_b": {"pretrain": loss_b0, "learned": loss_b1},
        "a_preserved": a_preserved, "b_learned": b_learned, "a_roundtrip": a_roundtrip,
        "memory": model.memory.count,
    }
    with open("./data/v3_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to data/v3_results.json")


if __name__ == "__main__":
    main()
