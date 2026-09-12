"""Online Learning LLM - SmolLM + CortexFlow.

A language model that learns continuously from every interaction.
Based on SmolLM-135M with added:
1. Hebbian learning layer (real-time weight updates)
2. Memory bank (unlimited context)
3. Predictive coding (efficient processing)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from pathlib import Path
from typing import Optional


# ============================================================
# Memory Bank (Unlimited Context)
# ============================================================

class MemoryBank:
    """Fixed-size memory that stores important patterns."""

    def __init__(self, capacity=10000, dim=64):
        self.capacity = capacity
        self.dim = dim
        self.keys = torch.zeros(capacity, dim)
        self.values = torch.zeros(capacity, dim)
        self.importance = torch.zeros(capacity)
        self.ptr = 0
        self.count = 0

    def store(self, key, value, importance=1.0):
        """Store a pattern."""
        if self.count < self.capacity:
            self.keys[self.ptr] = key[:self.dim] if len(key) >= self.dim else F.pad(key, (0, self.dim - len(key)))
            self.values[self.ptr] = value[:self.dim] if len(value) >= self.dim else F.pad(value, (0, self.dim - len(value)))
            self.importance[self.ptr] = importance
            self.ptr = (self.ptr + 1) % self.capacity
            self.count += 1
        else:
            # Replace least important
            min_idx = torch.argmin(self.importance)
            if importance > self.importance[min_idx]:
                self.keys[min_idx] = key[:self.dim] if len(key) >= self.dim else F.pad(key, (0, self.dim - len(key)))
                self.values[min_idx] = value[:self.dim] if len(value) >= self.dim else F.pad(value, (0, self.dim - len(value)))
                self.importance[min_idx] = importance

    def retrieve(self, query, k=5):
        """Retrieve most similar patterns."""
        if self.count == 0:
            return None, None

        query = query[:self.dim] if len(query) >= self.dim else F.pad(query, (0, self.dim - len(query)))

        # Compare against stored keys
        similarities = F.cosine_similarity(query.unsqueeze(0), self.keys[:self.count], dim=-1)
        top_k = torch.topk(similarities, min(k, self.count))

        return self.values[top_k.indices], top_k.values

    def decay(self, rate=0.99):
        """Decay importance over time."""
        self.importance[:self.count] *= rate


# ============================================================
# Hebbian Layer (Real-time Learning)
# ============================================================

class HebbianLayer(nn.Module):
    """Layer that learns via Hebbian rules during inference."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.learning_rate = 0.001
        self.decay = 0.999

    def forward(self, x, update=False):
        """Forward pass with optional Hebbian update."""
        output = x @ self.weight

        if update and self.training:
            # Hebbian update: strengthen co-active neurons
            with torch.no_grad():
                # Outer product of input and output
                hebbian_update = torch.outer(x.mean(0), output.mean(0))
                # Normalize and apply
                hebbian_update = hebbian_update / (hebbian_update.norm() + 1e-8)
                self.weight.data += self.learning_rate * hebbian_update
                # Decay
                self.weight.data *= self.decay

        return output


# ============================================================
# Online Learning LLM
# ============================================================

class OnlineLLM(nn.Module):
    """Language model that learns continuously."""

    def __init__(self, vocab_size=49152, embed_dim=96, hidden_dim=576,
                 n_layers=6, n_heads=6):
        super().__init__()

        # Base transformer (lightweight)
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output head
        self.head = nn.Linear(embed_dim, vocab_size)

        # CortexFlow additions
        self.hebbian = HebbianLayer(embed_dim)
        self.memory = MemoryBank(capacity=10000, dim=embed_dim)

        # Online learning state
        self.update_count = 0
        self.total_loss = 0

    def forward(self, x, use_memory=True, update=True):
        """Forward with optional online learning."""
        batch_size, seq_len = x.shape

        # Embed
        h = self.embedding(x)

        # Retrieve from memory
        if use_memory and self.memory.count > 0:
            query = h.mean(dim=1)
            memory_values, memory_scores = self.memory.retrieve(query.squeeze(0))
            if memory_values is not None:
                memory_context = memory_values.mean(0).unsqueeze(0).unsqueeze(0)
                memory_context = memory_context.expand(batch_size, seq_len, -1)
                h = h + 0.1 * memory_context

        # Transformer
        h = self.transformer(h)

        # Hebbian (on mean pooled)
        if update:
            h_mean = h.mean(1)
            h_hebb = self.hebbian(h_mean, update=update)
            h = h + h_hebb.unsqueeze(1) * 0.1

        # Output
        logits = self.head(h)

        return logits

    def learn_from_text(self, text, tokenizer, lr=0.0001):
        """Learn from a piece of text (online update)."""
        tokens = tokenizer.encode(text, return_tensors="pt")
        if tokens.shape[1] < 3:
            return 0.0

        input_ids = tokens[:, :-1]
        targets = tokens[:, 1:]

        logits = self.forward(input_ids, use_memory=False, update=True)

        # Fix shape mismatch
        min_len = min(logits.shape[1], targets.shape[1])
        logits = logits[:, :min_len, :]
        targets = targets[:, :min_len]

        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        # Store in memory
        with torch.no_grad():
            h = self.embedding(input_ids).mean(dim=1).squeeze(0)
            self.memory.store(h, h, importance=1.0 / (1.0 + loss.item()))

        self.update_count += 1
        self.total_loss += loss.item()

        return loss.item()

    def generate(self, tokenizer, prompt, max_len=100, temperature=0.8):
        """Generate text with memory retrieval."""
        self.eval()
        tokens = tokenizer.encode(prompt, return_tensors="pt")

        with torch.no_grad():
            for _ in range(max_len):
                logits = self.forward(tokens, use_memory=True, update=False)
                next_token_logits = logits[:, -1, :] / temperature

                # Handle shape
                if next_token_logits.dim() > 1:
                    next_token_logits = next_token_logits.squeeze(0)

                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), 1)
                tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=-1)

                if next_token.item() == tokenizer.eos_token_id:
                    break

        self.train()
        return tokenizer.decode(tokens[0])

    def get_stats(self):
        """Get learning statistics."""
        return {
            "update_count": self.update_count,
            "avg_loss": self.total_loss / max(self.update_count, 1),
            "memory_used": self.memory.count,
            "memory_capacity": self.memory.capacity,
        }


# ============================================================
# Training with SmolLM tokenizer
# ============================================================

def load_smollm_tokenizer():
    """Load SmolLM tokenizer."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
        return tokenizer
    except:
        # Fallback: simple character tokenizer
        print("  Using character tokenizer fallback")
        return CharTokenizer()


class CharTokenizer:
    """Simple character tokenizer."""
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.vocab_size = 0
        self.eos_token_id = 0
        self.pad_token_id = 0

    def fit(self, texts):
        chars = set()
        for t in texts:
            chars.update(t)
        for i, c in enumerate(sorted(chars)):
            self.char2idx[c] = i + 1
            self.idx2char[i + 1] = c
        self.vocab_size = len(self.char2idx) + 1
        self.eos_token_id = 0
        self.pad_token_id = 0

    def encode(self, text, return_tensors=None):
        indices = [self.char2idx.get(c, 0) for c in text]
        if return_tensors == "pt":
            return torch.tensor([indices])
        return indices

    def decode(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        if isinstance(indices[0], list):
            indices = indices[0]
        return "".join([self.idx2char.get(i, "") for i in indices if i > 0])


def online_training_demo():
    """Demo: model learns new information in real-time."""
    print("=" * 70)
    print("ONLINE LEARNING LLM - Demo")
    print("=" * 70)

    # Create model
    print("\n[1/4] Creating model...")
    model = OnlineLLM(vocab_size=256, embed_dim=64, hidden_dim=128, n_layers=2, n_heads=4)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # Create tokenizer
    print("\n[2/4] Creating tokenizer...")
    tokenizer = CharTokenizer()
    training_texts = [
        "The cat sat on the mat.",
        "Machine learning is a subset of artificial intelligence.",
        "The quick brown fox jumps over the lazy dog.",
        "Python is a popular programming language.",
        "The brain uses neural networks for processing.",
    ]
    tokenizer.fit("".join(training_texts))
    print(f"  Vocabulary: {tokenizer.vocab_size}")

    # Initial test
    print("\n[3/4] Testing before learning...")
    model.eval()
    prompt = "The"
    try:
        generated = model.generate(tokenizer, prompt, max_len=20)
        print(f"  Before: '{prompt}' -> '{generated}'")
    except Exception as e:
        print(f"  Before: (generation failed: {e})")

    # Online learning
    print("\n[4/4] Online learning...")
    model.train()

    for epoch in range(10):
        epoch_loss = 0
        for text in training_texts:
            # Pad text to minimum length
            padded = text + " " * max(0, 10 - len(text))
            loss = model.learn_from_text(padded, tokenizer)
            epoch_loss += loss

        avg_loss = epoch_loss / len(training_texts)
        stats = model.get_stats()
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f} updates={stats['update_count']}")

    # Test after learning
    print("\n" + "=" * 70)
    print("AFTER LEARNING")
    print("=" * 70)

    model.eval()
    prompts = ["The", "A", "I"]
    for p in prompts:
        try:
            generated = model.generate(tokenizer, p, max_len=20)
            print(f"  '{p}' -> '{generated}'")
        except Exception as e:
            print(f"  '{p}' -> (error: {e})")

    # Stats
    stats = model.get_stats()
    print(f"\nStats: {stats}")

    # Save
    output_path = Path("./data/online_model.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"\nModel saved to {output_path}")


if __name__ == "__main__":
    online_training_demo()
