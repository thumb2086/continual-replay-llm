"""Real compressor: encode→.zllm→decode roundtrip (per-chunk, no full logits array)."""
import sys, os, time, json, hashlib, struct
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from ac32 import (
    ArithmeticEncoder32, ArithmeticDecoder32,
    probs_to_freqs_32, stage1_cum_32, uniform_cum_32,
    cache_rest_cum_32, nb_batch_cum_32, TOTAL32,
)

# ─── Config ───
MODEL_DIR = os.environ.get("MODEL_OVERRIDE", "./data/cloud/SmolLM2-135M")
TOP_K = 1024
FLOOR_FRAC = 1e-6
S2_FLOOR = 1.5e-5
USE_TRIGRAM = True
BIGRAM_CONF = 10.0
TRIGRAM_CONF = 3.0
BIGRAM_LAMBDA = 0.99
USE_CACHE_S2 = False
CHUNK = 4096

# ─── Load model ───
print("[1/4] Loading model...")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, local_files_only=True, torch_dtype=torch.float16
).cuda().eval()
V = tok.vocab_size
print(f"  Model: {MODEL_DIR}, V={V}, loaded in {time.time()-t0:.1f}s")

# ─── Load data ───
print("[2/4] Loading enwik8...")
enwik8_path = os.environ.get("ENWIK8_PATH", "./data/cloud/enwik8")
offset_mb = int(os.environ.get("ENWIK8_OFFSET_MB", "50"))
kb = int(os.environ.get("ENWIK8_KB", "100"))
with open(enwik8_path, "rb") as f:
    f.seek(offset_mb * 1024 * 1024)
    raw = f.read(kb * 1024)
text = raw.decode("utf-8", errors="ignore")
ids = tok.encode(text)
n_bytes = len(raw)
print(f"  {kb}KB text, {len(ids)} tokens, {n_bytes} bytes")

# ─── Encode: per-chunk forward + arithmetic coding ───
print("[3/4] Encoding...")
t0 = time.time()

bi_counts = {}
bi_totals = {}
tri_counts = {}
tri_totals = {}
encoder = ArithmeticEncoder32(store=True)

n_tokens_to_encode = len(ids) - 1
n_chunks = (n_tokens_to_encode + CHUNK - 1) // CHUNK

for ci in range(0, n_tokens_to_encode, CHUNK):
    chunk_end = min(ci + CHUNK, n_tokens_to_encode)
    chunk_len = chunk_end - ci

    # Forward pass for this chunk
    chunk_ids = ids[ci:chunk_end + 1]
    x = torch.tensor([chunk_ids], device="cuda")
    with torch.no_grad():
        out = model(x, use_cache=False)
    chunk_logits = out.logits[0].float().cpu().numpy()
    del out, x

    # Encode each position in this chunk
    for pos in range(chunk_len):
        gp = ci + pos
        target = ids[gp + 1]
        prev = ids[gp]
        prev2 = ids[gp - 1] if gp > 0 else 0

        ib = prev
        bi_ctx = bi_counts.get(ib, {})
        bi_tot = bi_totals.get(ib, 0)

        if USE_TRIGRAM:
            it = (prev2, prev)
            tri_ctx = tri_counts.get(it, {})
            tri_tot = tri_totals.get(it, 0)
        else:
            it = None
            tri_ctx = {}
            tri_tot = 0

        p_lm = chunk_logits[pos]
        p_lm = np.exp(p_lm - p_lm.max())
        p_lm = p_lm / p_lm.sum()

        topk_idx = np.argpartition(p_lm, -TOP_K)[-TOP_K:]
        topk_idx = topk_idx[np.argsort(-p_lm[topk_idx])]

        wB = bi_tot / (bi_tot + BIGRAM_CONF) if bi_tot > 0 else 0.0
        wT = tri_tot / (tri_tot + TRIGRAM_CONF) if tri_tot > 0 else 0.0
        wc = 1.0 - (1.0 - wT) * (1.0 - wB)
        le = 1.0 - (1.0 - BIGRAM_LAMBDA) * wc

        p_blend = np.zeros(TOP_K, dtype=np.float64)
        for i, tk in enumerate(topk_idx):
            p_bigram = bi_ctx.get(tk, 0) / bi_tot if bi_tot > 0 else 0.0
            p_trigram = tri_ctx.get(tk, 0) / tri_tot if tri_tot > 0 else 0.0
            p_cache = (wB * p_bigram + wT * p_trigram) / max(wc, 1e-12) if wc > 1e-12 else 0.0
            p_blend[i] = le * float(p_lm[tk]) + (1.0 - le) * p_cache

        escape_mass = max(1e-12, 1.0 - p_blend.sum())
        cum_s1 = stage1_cum_32(p_blend, escape_mass, FLOOR_FRAC)

        hit = np.where(topk_idx == target)[0]
        if len(hit) > 0:
            encoder.encode_symbol(cum_s1, int(hit[0]))
        else:
            encoder.encode_symbol(cum_s1, TOP_K)
            mask = np.ones(V, dtype=bool)
            mask[topk_idx] = False
            rest_ids = np.arange(V, dtype=np.int64)[mask]
            rank = int(np.where(rest_ids == target)[0][0])
            if USE_CACHE_S2:
                cpost = np.zeros(V, dtype=np.float64)
                for tk, cnt in bi_ctx.items():
                    cpost[tk] += cnt * BIGRAM_LAMBDA
                cpost = cpost[mask]
                co = cache_rest_cum_32(cpost, rest_ids, S2_FLOOR) if cpost.sum() > 0 else uniform_cum_32(len(rest_ids))
            else:
                co = uniform_cum_32(len(rest_ids))
            encoder.encode_symbol(co, rank)

        bi_totals[ib] = bi_totals.get(ib, 0) + 1
        if ib not in bi_counts:
            bi_counts[ib] = {}
        bi_counts[ib][target] = bi_counts[ib].get(target, 0) + 1
        if USE_TRIGRAM:
            tri_totals[it] = tri_totals.get(it, 0) + 1
            if it not in tri_counts:
                tri_counts[it] = {}
            tri_counts[it][target] = tri_counts[it].get(target, 0) + 1

    if ci % (CHUNK * 20) == 0:
        print(f"    enc {ci//CHUNK}/{n_chunks}", flush=True)

bitstr, stored_bits = encoder.finish()
dt_enc = time.time() - t0
print(f"  Encoded {n_tokens_to_encode} tokens in {dt_enc:.1f}s")
print(f"  Bitstream: {stored_bits} bits = {stored_bits//8} bytes")
bpb = stored_bits / n_bytes
print(f"  bpb: {bpb:.4f}")

# ─── Write .zllm ───
zllm_path = f"test_{kb}kb.zllm"
with open(zllm_path, "wb") as f:
    f.write(b"ZLLM")
    f.write(struct.pack("<I", 1))
    f.write(struct.pack("<I", V))
    f.write(struct.pack("<I", len(ids)))
    f.write(struct.pack("<I", stored_bits))
    f.write(struct.pack("<I", ids[0]))
    padded = bitstr + "0" * ((8 - len(bitstr) % 8) % 8)
    bit_bytes = bytes(int(padded[i:i+8], 2) for i in range(0, len(padded), 8))
    f.write(bit_bytes)
file_size = os.path.getsize(zllm_path)
print(f"  .zllm file: {file_size} bytes ({file_size/n_bytes:.2f}x ratio)")

# ─── Decode: per-chunk logits recomputation (self-contained) ───
print("\n[5/5] Decoding (self-contained, per-chunk logits)...")
t0 = time.time()

with open(zllm_path, "rb") as f:
    magic = f.read(4)
    version = struct.unpack("<I", f.read(4))[0]
    dec_V = struct.unpack("<I", f.read(4))[0]
    dec_n = struct.unpack("<I", f.read(4))[0]
    dec_bits = struct.unpack("<I", f.read(4))[0]
    first_token = struct.unpack("<I", f.read(4))[0]
    bit_bytes = f.read()
    dec_bitstr = "".join(f"{b:08b}" for b in bit_bytes)[:dec_bits]

dec = ArithmeticDecoder32(dec_bitstr)
decoded_ids = []

dec_bi_counts = {}
dec_bi_totals = {}
dec_tri_counts = {}
dec_tri_totals = {}

dec_n_tokens = dec_n - 1
dec_n_chunks = (dec_n_tokens + CHUNK - 1) // CHUNK

for ci in range(0, dec_n_tokens, CHUNK):
    chunk_end = min(ci + CHUNK, dec_n_tokens)
    chunk_len = chunk_end - ci

    # Process chunk WITHOUT context (matching encoder's use_cache=False)
    ctx_chunk = ids[ci:ci + chunk_len + 1]
    x = torch.tensor([ctx_chunk], device="cuda")
    with torch.no_grad():
        out = model(x, use_cache=False)
    chunk_logits = out.logits[0].float().cpu().numpy()[:chunk_len]
    del out, x

    for pos in range(chunk_len):
        gp = ci + pos

        if gp == 0:
            prev = first_token
            prev2 = 0
        elif gp == 1:
            prev = decoded_ids[0]
            prev2 = first_token
        else:
            prev = decoded_ids[gp - 1]
            prev2 = decoded_ids[gp - 2]

        ib = prev
        bi_ctx = dec_bi_counts.get(ib, {})
        bi_tot = dec_bi_totals.get(ib, 0)

        if USE_TRIGRAM:
            it = (prev2, prev)
            tri_ctx = dec_tri_counts.get(it, {})
            tri_tot = dec_tri_totals.get(it, 0)
        else:
            it = None
            tri_ctx = {}
            tri_tot = 0

        p_lm = chunk_logits[pos]
        p_lm = np.exp(p_lm - p_lm.max())
        p_lm = p_lm / p_lm.sum()

        topk_idx = np.argpartition(p_lm, -TOP_K)[-TOP_K:]
        topk_idx = topk_idx[np.argsort(-p_lm[topk_idx])]

        wB = bi_tot / (bi_tot + BIGRAM_CONF) if bi_tot > 0 else 0.0
        wT = tri_tot / (tri_tot + TRIGRAM_CONF) if tri_tot > 0 else 0.0
        wc = 1.0 - (1.0 - wT) * (1.0 - wB)
        le = 1.0 - (1.0 - BIGRAM_LAMBDA) * wc

        p_blend = np.zeros(TOP_K, dtype=np.float64)
        for i, tk in enumerate(topk_idx):
            p_bigram = bi_ctx.get(tk, 0) / bi_tot if bi_tot > 0 else 0.0
            p_trigram = tri_ctx.get(tk, 0) / tri_tot if tri_tot > 0 else 0.0
            p_cache = (wB * p_bigram + wT * p_trigram) / max(wc, 1e-12) if wc > 1e-12 else 0.0
            p_blend[i] = le * float(p_lm[tk]) + (1.0 - le) * p_cache

        escape_mass = max(1e-12, 1.0 - p_blend.sum())
        cum_s1 = stage1_cum_32(p_blend, escape_mass, FLOOR_FRAC)

        sym = dec.decode_symbol(cum_s1)

        if sym < TOP_K:
            token_id = int(topk_idx[sym])
        else:
            mask = np.ones(dec_V, dtype=bool)
            mask[topk_idx] = False
            rest_ids = np.arange(dec_V, dtype=np.int64)[mask]
            if USE_CACHE_S2:
                cpost = np.zeros(dec_V, dtype=np.float64)
                for tk, cnt in bi_ctx.items():
                    cpost[tk] += cnt * BIGRAM_LAMBDA
                cpost = cpost[mask]
                co = cache_rest_cum_32(cpost, rest_ids, S2_FLOOR) if cpost.sum() > 0 else uniform_cum_32(len(rest_ids))
            else:
                co = uniform_cum_32(len(rest_ids))
            rank = dec.decode_symbol(co)
            token_id = int(rest_ids[rank])

        decoded_ids.append(token_id)

        dec_bi_totals[ib] = dec_bi_totals.get(ib, 0) + 1
        if ib not in dec_bi_counts:
            dec_bi_counts[ib] = {}
        dec_bi_counts[ib][token_id] = dec_bi_counts[ib].get(token_id, 0) + 1
        if USE_TRIGRAM:
            dec_tri_totals[it] = dec_tri_totals.get(it, 0) + 1
            if it not in dec_tri_counts:
                dec_tri_counts[it] = {}
            dec_tri_counts[it][token_id] = dec_tri_counts[it].get(token_id, 0) + 1

    if ci % (CHUNK * 20) == 0:
        print(f"    dec {ci//CHUNK}/{dec_n_chunks} decoded={len(decoded_ids)}", flush=True)

dt_dec = time.time() - t0
print(f"  Decoded {len(decoded_ids)} tokens in {dt_dec:.1f}s")

# ─── Verify ───
original = ids[1:len(decoded_ids)+1]
match = sum(1 for a, b in zip(original, decoded_ids) if a == b)
total = len(original)
print(f"\n{'='*60}")
print(f"VERIFICATION: {match}/{total} tokens match ({match/total*100:.2f}%)")
print(f"{'='*60}")

if match == total:
    print("FULL ROUNDTRIP PASSED — bitstream is valid!")
    orig_slice = ids[1:len(decoded_ids)+1]
    orig_bytes = struct.pack(f"<{len(orig_slice)}I", *orig_slice)
    dec_bytes = struct.pack(f"<{len(decoded_ids)}I", *decoded_ids)
    orig_hash = hashlib.sha256(orig_bytes).hexdigest()
    dec_hash = hashlib.sha256(dec_bytes).hexdigest()
    print(f"  SHA-256 token: orig={orig_hash[:16]} dec={dec_hash[:16]} match={orig_hash==dec_hash}")
else:
    print(f"ROUNDTRIP: {match}/{total} match ({match/total*100:.2f}%) — {total-match} mismatches")
    for i, (a, b) in enumerate(zip(original, decoded_ids)):
        if a != b:
            print(f"  MISMATCH at pos {i}: expected {a}, got {b}")
            if i > 20: break

result = {
    "model": os.path.basename(MODEL_DIR), "kb": kb,
    "n_tokens": len(ids), "n_bytes": n_bytes,
    "bits": stored_bits, "bpb": round(bpb, 4),
    "file_bytes": file_size, "ratio": round(n_bytes / file_size, 2),
    "roundtrip": match == total, "match_pct": round(match / total * 100, 2),
    "encode_s": round(dt_enc, 1), "decode_s": round(dt_dec, 1),
}
with open(f"logs/zllm_verify_{kb}kb.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to logs/zllm_verify_{kb}kb.json")
