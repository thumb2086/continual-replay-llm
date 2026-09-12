"""Profile each phase precisely."""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from train_v3 import OnlineLLMV3
from real_compression import probs_to_freqs_batch, nb_encode_count


def main():
    device = torch.device("cuda")
    torch.manual_seed(0)
    model = OnlineLLMV3(
        vocab_size=3239, embed_dim=256, hidden_dim=512, n_layers=6, n_heads=8
    ).to(device)
    model.eval()

    xb = torch.randint(0, 3239, (8, 128), device=device)
    for _ in range(5):
        model(xb)

    torch.cuda.synchronize()
    t = time.time()
    for _ in range(50):
        model(xb)
    torch.cuda.synchronize()
    print(f"fp32 fwd batch8 x50: {time.time()-t:.3f}s")

    torch.cuda.synchronize()
    t = time.time()
    for _ in range(50):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(xb)
            probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
    torch.cuda.synchronize()
    print(f"fp16 fwd+softmax+DtoH x50: {time.time()-t:.3f}s")

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    tgt = torch.randint(0, 3239, (1024,), device=device)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(50):
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, 3239), tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    print(f"fp32 bwd batch8 x50: {time.time()-t:.3f}s")

    P = np.random.rand(8, 127, 3239).astype(np.float64)
    syms = np.random.randint(0, 3239, (8, 127)).astype(np.int64)
    # warmup numba
    c0 = probs_to_freqs_batch(P[0])
    nb_encode_count(np.ascontiguousarray(c0), np.ascontiguousarray(syms[0]))
    t = time.time()
    for _ in range(50):
        cum = probs_to_freqs_batch(P.reshape(-1, 3239)).reshape(8, 127, 3240)
        tot = 0
        for j in range(8):
            tot += nb_encode_count(
                np.ascontiguousarray(cum[j]), np.ascontiguousarray(syms[j])
            )
    print(f"numpy cumsum + numba count 50 blocks: {time.time()-t:.3f}s")


if __name__ == "__main__":
    main()
