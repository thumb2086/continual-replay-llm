"""Speed ceiling: fastest honest config (S + bias-only + prefix40)."""

import torch
import numpy as np
import sys
import time
import json

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from train_v3 import OnlineLLMV3, CharTokenizer, load_grouped_texts
from real_compression import neural_compress_stream, frozen_code_all

CFG_S = dict(embed_dim=128, hidden_dim=256, n_layers=2, n_heads=4)


def main():
    print("=" * 70)
    print("SPEED CEILING TEST")
    print("=" * 70)
    torch.manual_seed(123)
    np.random.seed(123)
    device = torch.device("cuda")

    base_path = r"C:\Users\CPXru\Music\playlist-admin\podcasts"
    texts_a, texts_b, _, _ = load_grouped_texts(base_path, max_files_per_group=150)
    tok = CharTokenizer()
    tok.fit(texts_a + texts_b)
    print(f"Vocab: {tok.vocab_size}")

    with open("./data/frozen_stream.json", encoding="utf-8") as f:
        stream = json.load(f)["chunks"]
    n_chars = sum(len(c) for c in stream)

    model = OnlineLLMV3(vocab_size=tok.vocab_size, **CFG_S).to(device)
    model.load_state_dict(torch.load("./data/bigdata_S.pt.best", map_location=device,
                                     weights_only=False))
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_total:,}")

    # Bias-only setup
    for p in model.parameters():
        p.requires_grad_(False)
    for n, p in model.named_parameters():
        if "bias" in n:
            p.requires_grad_(True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable (bias-only): {n_train} ({n_train/n_total*100:.2f}%)")

    results = {}

    # 1. Frozen fp32
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    t = time.time()
    nbits, _, nver = frozen_code_all(model, tok, stream, device, verify=True,
                                     use_fp16=False)
    torch.cuda.synchronize()
    dt = time.time() - t
    results["frozen_fp32"] = {"bpc": nbits / n_chars, "chars_per_s": n_chars / dt,
                              "peak_mem_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
                              "verified": nver}
    print(f"frozen fp32: {nbits/n_chars:.3f} bpc, {n_chars/dt:.0f} c/s", flush=True)

    # 2. Bias-only prefix40 fp32
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=1e-4, fused=True)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    t = time.time()
    nbits, _, nver = neural_compress_stream(
        model, tok, stream, device, adapt=True, replay_pool=None,
        opt=opt, block_size=8, freeze_after=40, use_fp16=False)
    torch.cuda.synchronize()
    dt = time.time() - t
    results["bias_prefix40_fp32"] = {
        "bpc": nbits / n_chars, "chars_per_s": n_chars / dt,
        "peak_mem_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "verified": nver}
    print(f"bias+prefix40 fp32: {nbits/n_chars:.3f} bpc, {n_chars/dt:.0f} c/s", flush=True)

    print("\n" + "=" * 70)
    for k, r in results.items():
        print(f"  {k:22s}: {r['bpc']:.3f} bpc, {r['chars_per_s']:.0f} chars/s, "
              f"peak {r['peak_mem_mb']:.0f} MB")
    with open("./data/speed_ceiling.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to data/speed_ceiling.json")


if __name__ == "__main__":
    main()
