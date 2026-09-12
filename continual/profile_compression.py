"""Profile the neural compression pipeline to find bottlenecks."""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import sys as _sys_root; _sys_root.path.insert(0, ".")  # run-from-root: core lives at repo root
from train_v3 import OnlineLLMV3, CharTokenizer, load_grouped_texts, create_chunks
from compare_ablation import make_batch
from real_compression import probs_to_freqs, ArithmeticEncoder


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda")

    base_path = r"C:\Users\CPXru\Music\playlist-admin\podcasts"
    texts_a, texts_b, _, _ = load_grouped_texts(base_path, max_files_per_group=5)
    tok = CharTokenizer()
    tok.fit(texts_a + texts_b)
    chunks_b = create_chunks(texts_b, chunk_len=128)[:20]

    model = OnlineLLMV3(
        vocab_size=tok.vocab_size, embed_dim=128, hidden_dim=256, n_layers=4, n_heads=4
    ).to(device)
    model.eval()

    # Warmup
    ids = tok.encode(chunks_b[0])
    x = torch.tensor([ids], device=device)
    for _ in range(5):
        model(x)

    # 1. Forward pass
    torch.cuda.synchronize()
    t = time.time()
    for c in chunks_b:
        ids = tok.encode(c)
        x = torch.tensor([ids], device=device)
        with torch.no_grad():
            logits = model(x)
    torch.cuda.synchronize()
    fwd_time = time.time() - t
    print(f"Forward (20 chunks): {fwd_time:.3f}s")

    # 2. probs_to_freqs per position
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits[0], dim=-1).cpu().numpy()
    t = time.time()
    for _ in range(20):
        for p in probs:
            probs_to_freqs(p)
    print(f"probs_to_freqs (20 chunks x 128 pos): {time.time()-t:.3f}s")

    # 3. encode_symbol loop
    t = time.time()
    for _ in range(20):
        enc = ArithmeticEncoder(store=False)
        for p in probs:
            _, cum = probs_to_freqs(p)
            enc.encode_symbol(cum, 5)
    print(f"encode loop (20 chunks x 128 sym): {time.time()-t:.3f}s")

    # 4. online_step (backward + optimizer)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    torch.cuda.synchronize()
    t = time.time()
    for c in chunks_b:
        ids = tok.encode(c)
        tokens = torch.tensor([ids], device=device)
        model.online_step(tokens, opt, None, 0)
    torch.cuda.synchronize()
    print(f"online_step x20 (fwd+bwd+optim): {time.time()-t:.3f}s")

    # 5. Decode loop
    enc = ArithmeticEncoder(store=True)
    for p in probs:
        _, cum = probs_to_freqs(p)
        enc.encode_symbol(cum, 5)
    bitstr, _ = enc.finish()
    from real_compression import ArithmeticDecoder
    t = time.time()
    for _ in range(20):
        dec = ArithmeticDecoder(bitstr)
        for p in probs:
            _, cum = probs_to_freqs(p)
            dec.decode_symbol(cum)
    print(f"decode loop (20 chunks x 128 sym): {time.time()-t:.3f}s")


if __name__ == "__main__":
    main()
