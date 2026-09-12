"""Comprehensive sweep: size x precision x pruning x adaptation.

Compares neural codecs against Huffman on speed, ratio, and resources.
All conditions share data, tokenizer, and stream for comparability.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import time
import json
import heapq
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from train_v3 import OnlineLLMV3, CharTokenizer, load_grouped_texts, create_chunks
from compare_ablation import make_batch
from real_compression import (
    neural_compress_stream,
    frozen_code_all,
    build_huffman,
    huffman_compress,
)
from prune_experiment import prune_ffn_channels

CONFIGS = {
    "S": dict(embed_dim=128, hidden_dim=256, n_layers=2, n_heads=4),
    "M": dict(embed_dim=256, hidden_dim=512, n_layers=6, n_heads=8),
}
PRETRAIN_EPOCHS = 10


def gpu_peak_mb(device):
    return torch.cuda.max_memory_allocated(device) / 1024**2


def run_condition(model, tok, stream, chunks_a, device, adapt, freeze_after,
                  use_fp16, verify_first=False):
    """Run one coding condition, return metrics dict."""
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    t = time.time()
    if adapt is None:
        nbits, _, nver = frozen_code_all(
            model, tok, stream, device, verify=verify_first, use_fp16=use_fp16
        )
    else:
        opt = torch.optim.Adam(model.parameters(), lr=1e-4, fused=True)
        nbits, _, nver = neural_compress_stream(
            model, tok, stream, device, adapt=True, replay_pool=chunks_a,
            opt=opt, block_size=8, freeze_after=freeze_after,
            verify=verify_first, use_fp16=use_fp16,
        )
    torch.cuda.synchronize()
    dt = time.time() - t
    return {
        "bpc": nbits / N_CHARS,
        "chars_per_s": N_CHARS / dt,
        "peak_mem_mb": gpu_peak_mb(device),
        "verified": nver,
    }


def main():
    print("=" * 70)
    print("COMPREHENSIVE SWEEP: size x precision x pruning x adaptation")
    print("=" * 70)
    torch.manual_seed(123)
    np.random.seed(123)
    device = torch.device("cuda")

    base_path = r"C:\Users\CPXru\Music\playlist-admin\podcasts"
    texts_a, texts_b, _, _ = load_grouped_texts(base_path, max_files_per_group=25)
    tok = CharTokenizer()
    tok.fit(texts_a + texts_b)
    chunks_a = create_chunks(texts_a, chunk_len=128)
    chunks_b = create_chunks(texts_b, chunk_len=128)
    np.random.shuffle(chunks_a)
    np.random.shuffle(chunks_b)
    stream = chunks_b[:400]
    global N_CHARS
    N_CHARS = sum(len(c) for c in stream)
    print(f"Vocab: {tok.vocab_size}, stream chars: {N_CHARS}")

    # ---- Huffman baseline ----
    stream_ids = [tok.encode(c) for c in stream]
    all_ids = [i for ids in stream_ids for i in ids]
    codes, _ = build_huffman(Counter(all_ids))
    huff_bits = sum(len(codes[t]) for t in all_ids)
    t = time.time()
    for _ in range(5):
        huffman_compress(all_ids, codes)
    huff_dt = (time.time() - t) / 5
    huff_bpc = huff_bits / N_CHARS
    print(f"\nHuffman: {huff_bpc:.3f} bpc, {N_CHARS/huff_dt:.0f} chars/s")

    results = {"huffman": {"bpc": huff_bpc, "chars_per_s": N_CHARS / huff_dt},
               "conditions": {}}

    for size_name, cfg in CONFIGS.items():
        # ---- Pretrain once per size ----
        print(f"\n--- Pretraining {size_name} ---")
        base = OnlineLLMV3(vocab_size=tok.vocab_size, **cfg).to(device)
        opt = torch.optim.Adam(base.parameters(), lr=3e-4)
        base.train()
        for epoch in range(PRETRAIN_EPOCHS):
            total, nb = 0.0, 0
            for i in range(0, len(chunks_a) - 8, 8):
                total += base.pretrain_step(make_batch(tok, chunks_a[i:i + 8], device), opt)
                nb += 1
            print(f"  Epoch {epoch+1}: loss={total/nb:.4f}")
        base_state = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
        del base
        torch.cuda.empty_cache()

        variants = {"dense": None}
        # Pruned variant: clone, prune FFN-50%, brief recovery. Keep the
        # whole model object: pruning changes weight shapes, so a plain
        # state_dict cannot load into a fresh full-size instance.
        print(f"--- Pruning {size_name} (FFN-50% + 2-epoch recovery) ---")
        import copy
        pm = OnlineLLMV3(vocab_size=tok.vocab_size, **cfg).to(device)
        pm.load_state_dict(base_state)
        prune_ffn_channels(pm, keep_ratio=0.5)
        ropt = torch.optim.Adam(pm.parameters(), lr=3e-4)
        pm.train()
        for epoch in range(2):
            total, nb = 0.0, 0
            for i in range(0, len(chunks_a) - 8, 8):
                total += pm.pretrain_step(make_batch(tok, chunks_a[i:i + 8], device), ropt)
                nb += 1
        pruned_template = copy.deepcopy(pm).cpu()
        del pm
        torch.cuda.empty_cache()

        for vname in ["dense", "pruned"]:
            for adapt, freeze in [("frozen", None), ("prefix40", 40), ("full", None)]:
                use_adapt = None if adapt == "frozen" else True
                fa = None if adapt != "prefix40" else 40
                if adapt == "full":
                    fa = None
                for prec in ["fp16", "fp32"]:
                    import copy as _copy
                    if vname == "dense":
                        m = OnlineLLMV3(vocab_size=tok.vocab_size, **cfg).to(device)
                        m.load_state_dict(base_state)
                    else:
                        m = _copy.deepcopy(pruned_template).to(device)
                    n_params = sum(p.numel() for p in m.parameters())
                    r = run_condition(m, tok, stream, chunks_a, device,
                                      use_adapt, fa, prec == "fp16",
                                      verify_first=(vname == "dense" and adapt == "frozen" and prec == "fp16"))
                    key = f"{size_name}/{vname}/{adapt}/{prec}"
                    results["conditions"][key] = {
                        "params": n_params,
                        "ckpt_mb": n_params * 4 / 1024**2,
                        **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()},
                    }
                    print(f"  {key:28s}: {r['bpc']:.3f} bpc, {r['chars_per_s']:.0f} c/s, "
                          f"peak {r['peak_mem_mb']:.0f} MB")
                    del m
                    torch.cuda.empty_cache()

    # ---- Summary table ----
    print("\n" + "=" * 70)
    print("FULL COMPARISON (sorted by bpc)")
    print("=" * 70)
    print(f"  {'condition':28s} {'params':>9s} {'bpc':>7s} {'chars/s':>9s} {'peakMB':>7s}")
    rows = sorted(results["conditions"].items(), key=lambda kv: kv[1]["bpc"])
    for key, r in rows:
        print(f"  {key:28s} {r['params']:>9,} {r['bpc']:>7.3f} {r['chars_per_s']:>9.0f} {r['peak_mem_mb']:>7.0f}")
    print(f"  {'huffman':28s} {'-':>9s} {huff_bpc:>7.3f} {N_CHARS/huff_dt:>9.0f} {'~0':>7s}")

    with open("./data/sweep_all.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to data/sweep_all.json")


if __name__ == "__main__":
    main()
