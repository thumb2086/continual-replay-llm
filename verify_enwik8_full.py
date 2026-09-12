"""Full lossless verification on enwik8 (every chunk, M-bigdata model)."""

import torch
import numpy as np
import sys
import time
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from train_v3 import OnlineLLMV3, CharTokenizer, load_grouped_texts
from real_compression import (
    gpu_quantize, nb_encode_count,
    ArithmeticEncoder, ArithmeticDecoder,
)

CFG_M = dict(embed_dim=256, hidden_dim=512, n_layers=6, n_heads=8)


def main():
    print("=" * 70)
    print("ENWIK8 FULL LOSSLESS VERIFICATION")
    print("=" * 70)
    torch.manual_seed(123)
    np.random.seed(123)
    device = torch.device("cuda")
    t0 = time.time()

    print("\n[1/3] Rebuilding tokenizer...")
    base_path = r"C:\Users\CPXru\Music\playlist-admin\podcasts"
    texts_a, texts_b, _, _ = load_grouped_texts(base_path, max_files_per_group=150)
    tok = CharTokenizer()
    tok.fit(texts_a + texts_b)
    print(f"  Vocab: {tok.vocab_size}")

    print("\n[2/3] Loading enwik8 + M-bigdata checkpoint...")
    with open("./data/cloud/enwik8", "rb") as f:
        raw = f.read(100 * 1024)
    text = raw.decode("utf-8", errors="ignore")
    chunks = [text[i:i + 128] for i in range(0, len(text) - 128, 128)]
    print(f"  Chunks: {len(chunks)}")

    model = OnlineLLMV3(vocab_size=tok.vocab_size, **CFG_M).to(device)
    model.load_state_dict(torch.load("./data/bigdata_M.pt.best", map_location=device,
                                     weights_only=False))
    model.eval()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    print("\n[3/3] Encode + decode EVERY chunk...")
    total_bits, total_chars = 0, 0
    n_ok, n_fail = 0, 0
    t = time.time()
    with torch.no_grad():
        for ci in range(0, len(chunks), 32):
            block = chunks[ci:ci + 32]
            id_lists = []
            for c in block:
                ids = tok.encode(c)
                if len(ids) >= 3:
                    id_lists.append(ids)
            if not id_lists:
                continue
            maxlen = max(len(ids) for ids in id_lists)
            batch = torch.zeros(len(id_lists), maxlen, dtype=torch.long, device=device)
            for j, ids in enumerate(id_lists):
                batch[j, : len(ids)] = torch.tensor(ids, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(batch)
            V = logits.shape[-1]
            cum_block = gpu_quantize(logits).reshape(len(id_lists), -1, V + 1)

            for j, ids in enumerate(id_lists):
                L = len(ids)
                cum = np.ascontiguousarray(cum_block[j, : L - 1]).astype(np.int64)
                syms = np.array(ids[1:], dtype=np.int64)
                # Reference Python path: materialize + decode + compare
                enc = ArithmeticEncoder(store=True)
                for tt in range(L - 1):
                    enc.encode_symbol(cum[tt], int(syms[tt]))
                bitstr, nbits = enc.finish()
                dec = ArithmeticDecoder(bitstr)
                out = [dec.decode_symbol(cum[tt]) for tt in range(L - 1)]
                if out == ids[1:]:
                    n_ok += 1
                else:
                    n_fail += 1
                    print(f"  FAIL at chunk {ci+j}")
                total_bits += nbits
                total_chars += L - 1
            if (ci // 32 + 1) % 5 == 0:
                print(f"  ... {ci+len(id_lists)}/{len(chunks)} chunks", flush=True)

    dt = time.time() - t
    print(f"\n  Verified: {n_ok}/{n_ok+n_fail} chunks lossless")
    print(f"  Total bits: {total_bits}, bpc: {total_bits/total_chars:.3f}")
    print(f"  Time: {dt:.1f}s")

    results = {
        "n_chunks": n_ok + n_fail,
        "verified_ok": n_ok,
        "verified_fail": n_fail,
        "bpc": total_bits / total_chars,
        "total_bits": total_bits,
        "elapsed_s": time.time() - t0,
    }
    with open("./data/enwik8_full_verify.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to data/enwik8_full_verify.json")


if __name__ == "__main__":
    main()
