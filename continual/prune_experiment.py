"""Structured pruning experiment: smaller dense models via channel/layer pruning."""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import time
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from train_v3 import OnlineLLMV3, CharTokenizer, load_grouped_texts, create_chunks
from compare_ablation import make_batch
from real_compression import gpu_quantize, nb_encode_count, TOTAL


def prune_ffn_channels(model, keep_ratio=0.5):
    """Magnitude-prune FFN intermediate channels in every encoder layer."""
    for layer in model.transformer.layers:
        lin1, lin2 = layer.linear1, layer.linear2
        assert lin1.out_features == lin2.in_features
        n_keep = max(8, int(lin1.out_features * keep_ratio))
        with torch.no_grad():
            importance = lin1.weight.norm(dim=1)  # per-channel L2
            _, idx = torch.topk(importance, n_keep)
            idx, _ = torch.sort(idx)
            lin1.weight = torch.nn.Parameter(lin1.weight[idx].clone())
            if lin1.bias is not None:
                lin1.bias = torch.nn.Parameter(lin1.bias[idx].clone())
            lin2.weight = torch.nn.Parameter(lin2.weight[:, idx].clone())
            lin1.out_features = n_keep
            lin2.in_features = n_keep
    return model


def drop_layers(model, n_drop=2):
    """Drop the last n_drop encoder layers."""
    model.transformer.layers = model.transformer.layers[: len(model.transformer.layers) - n_drop]
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def measure_stream(model, tok, stream_chunks, device):
    """Measure bits/char (no adaptation) + coding speed."""
    id_lists = []
    for c in stream_chunks:
        ids = tok.encode(c)
        if len(ids) >= 3:
            id_lists.append(ids)
    model.eval()
    total_bits, total_chars = 0, 0
    torch.cuda.synchronize()
    t = time.time()
    with torch.no_grad():
        for i in range(0, len(id_lists), 16):
            block = id_lists[i : i + 16]
            maxlen = max(len(ids) for ids in block)
            batch = torch.zeros(len(block), maxlen, dtype=torch.long, device=device)
            for j, ids in enumerate(block):
                batch[j, : len(ids)] = torch.tensor(ids, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(batch)
            V = logits.shape[-1]
            cum = gpu_quantize(logits).reshape(len(block), -1, V + 1)
            for j, ids in enumerate(block):
                L = len(ids)
                c = np.ascontiguousarray(cum[j, : L - 1])
                s = np.array(ids[1:], dtype=np.int64)
                total_bits += int(nb_encode_count(c, s))
                total_chars += L - 1
    torch.cuda.synchronize()
    dt = time.time() - t
    return total_bits / total_chars, total_chars / dt


def main():
    print("=" * 70)
    print("STRUCTURED PRUNING EXPERIMENT")
    print("=" * 70)
    torch.manual_seed(123)
    np.random.seed(123)
    device = torch.device("cuda")
    t0 = time.time()

    base_path = r"C:\Users\CPXru\Music\playlist-admin\podcasts"
    texts_a, texts_b, _, _ = load_grouped_texts(base_path, max_files_per_group=25)
    tok = CharTokenizer()
    tok.fit(texts_a + texts_b)
    chunks_a = create_chunks(texts_a, chunk_len=128)
    chunks_b = create_chunks(texts_b, chunk_len=128)
    np.random.shuffle(chunks_a)
    np.random.shuffle(chunks_b)
    stream = chunks_b[:200]
    print(f"Vocab: {tok.vocab_size}, stream chunks: {len(stream)}")

    def make_model():
        return OnlineLLMV3(
            vocab_size=tok.vocab_size, embed_dim=256, hidden_dim=512,
            n_layers=6, n_heads=8,
        ).to(device)

    def pretrain(model, epochs, chunks):
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        model.train()
        for _ in range(epochs):
            for i in range(0, len(chunks) - 8, 8):
                model.pretrain_step(make_batch(tok, chunks[i:i + 8], device), opt)

    results = {}

    # ---- Full model ----
    print("\n[Full] training...")
    full = make_model()
    pretrain(full, 5, chunks_a)
    bpc, cps = measure_stream(full, tok, stream, device)
    results["full"] = {"params": count_params(full), "bpc": bpc, "chars_per_s": cps}
    print(f"  params={results['full']['params']:,}, bpc={bpc:.3f}, {cps:.0f} chars/s")

    # ---- FFN 50% pruned + recovery ----
    print("\n[FFN-50%] pruning + recovery...")
    import copy
    pruned = make_model()
    pruned.load_state_dict(full.state_dict())
    prune_ffn_channels(pruned, keep_ratio=0.5)
    print(f"  params after prune: {count_params(pruned):,}")
    pretrain(pruned, 2, chunks_a)  # recovery
    bpc, cps = measure_stream(pruned, tok, stream, device)
    results["ffn50"] = {"params": count_params(pruned), "bpc": bpc, "chars_per_s": cps}
    print(f"  params={results['ffn50']['params']:,}, bpc={bpc:.3f}, {cps:.0f} chars/s")

    # ---- FFN 50% + drop 2 layers + recovery ----
    print("\n[FFN-50%+L4] pruning + recovery...")
    small = make_model()
    small.load_state_dict(full.state_dict())
    prune_ffn_channels(small, keep_ratio=0.5)
    drop_layers(small, n_drop=2)
    print(f"  params after prune: {count_params(small):,}")
    pretrain(small, 2, chunks_a)  # recovery
    bpc, cps = measure_stream(small, tok, stream, device)
    results["ffn50_l4"] = {"params": count_params(small), "bpc": bpc, "chars_per_s": cps}
    print(f"  params={results['ffn50_l4']['params']:,}, bpc={bpc:.3f}, {cps:.0f} chars/s")

    print("\n" + "=" * 70)
    print("PRUNING RESULTS (no adaptation, pure inference speed)")
    print("Warmed-up re-measurement, interleaved trials:")
    print("=" * 70)
    import time as _time
    models = {"full": full, "ffn50": pruned, "ffn50_l4": small}
    # Warmup on all models first
    for m in models.values():
        measure_stream(m, tok, stream[:10], device)
    trials = {k: [] for k in models}
    for _ in range(3):
        for k in ["full", "ffn50", "ffn50_l4"]:
            _, cps = measure_stream(models[k], tok, stream, device)
            trials[k].append(cps)
    base_cps = float(np.median(trials["full"]))
    base_bpc = results["full"]["bpc"]
    for k in ["full", "ffn50", "ffn50_l4"]:
        med = float(np.median(trials[k]))
        results[k]["chars_per_s_warmed"] = med
        print(f"  {k:10s}: {results[k]['params']:>9,} params | {results[k]['bpc']:.3f} bpc "
              f"({(results[k]['bpc']-base_bpc)/base_bpc*100:+.1f}%) | {med:.0f} chars/s "
              f"({med/base_cps:.2f}x)")

    with open("./data/pruning_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
