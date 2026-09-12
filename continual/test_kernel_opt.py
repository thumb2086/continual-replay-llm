"""Test torch.compile speedup and GPU-side quantization."""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from train_v3 import OnlineLLMV3

TOTAL = 16384


def gpu_quantize(logits):
    """GPU-side quantization: logits (B, T, V) -> cum (B*T, V+1) int32 on CPU.

    All heavy ops run as fused GPU kernels; only the final int32 cum
    crosses PCIe.
    """
    B, T, V = logits.shape
    probs = F.softmax(logits.float(), dim=-1).reshape(-1, V)  # (N, V) fp32
    rowsum = probs.sum(dim=1, keepdim=True)
    scale = (TOTAL - V) / rowsum
    freqs = (probs * scale).to(torch.int32) + 1
    diff = TOTAL - freqs.sum(dim=1)
    idx = probs.argmax(dim=1)
    freqs[torch.arange(freqs.shape[0], device=freqs.device), idx] += diff
    cum = torch.empty(freqs.shape[0], V + 1, dtype=torch.int32, device=freqs.device)
    cum[:, 0] = 0
    torch.cumsum(freqs, dim=1, out=cum[:, 1:])
    return cum.cpu().numpy()


def main():
    device = torch.device("cuda")
    torch.manual_seed(0)
    model = OnlineLLMV3(
        vocab_size=3239, embed_dim=256, hidden_dim=512, n_layers=6, n_heads=8
    ).to(device)
    model.eval()

    xb = torch.randint(0, 3239, (8, 128), device=device)

    # Baseline eager
    for _ in range(5):
        model(xb)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(50):
        model(xb)
    torch.cuda.synchronize()
    print(f"eager fwd batch8 x50: {time.time()-t:.3f}s")

    # Backward comparison (eager; inductor/triton unavailable on this box)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    tgt = torch.randint(0, 3239, (1024,), device=device)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(20):
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, 3239), tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    print(f"eager bwd batch8 x20: {time.time()-t:.3f}s")

    # GPU quantize timing + correctness
    with torch.no_grad():
        logits = model(xb)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(50):
        cum = gpu_quantize(logits)
    torch.cuda.synchronize()
    print(f"gpu quantize x50: {time.time()-t:.3f}s")
    print("cum valid:", bool((cum[:, -1] == TOTAL).all()),
          "floor:", bool((np.diff(cum, axis=1) >= 1).all()))

    # Equivalence with CPU version on same logits
    from real_compression import probs_to_freqs_batch
    p = F.softmax(logits.float(), dim=-1).reshape(-1, 3239).cpu().numpy()
    c_cpu = probs_to_freqs_batch(p)
    print("gpu==cpu cum:", bool((cum == c_cpu).all()))


if __name__ == "__main__":
    main()
