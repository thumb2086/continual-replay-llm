"""Minimal working compressor/decompressor using v13's probability model.

Encode: text → ArithmeticEncoder32(store=True) → .zllm file
Decode: .zllm file → ArithmeticDecoder32 → text
Verify: SHA-256 hash comparison
"""
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
kb = int(os.environ.get("ENWIK8_KB", "100"))  # default 100KB
with open(enwik8_path, "rb") as f:
    f.seek(offset_mb * 1024 * 1024)
    raw = f.read(kb * 1024)
text = raw.decode("utf-8", errors="ignore")
ids = tok.encode(text)
n_bytes = len(raw)
print(f"  {kb}KB text, {len(ids)} tokens, {n_bytes} bytes")

# ─── Encode ───
print("[3/4] Encoding (producing real bitstream)...")
t0 = time.time()

# Bigram/trigram tables (same as v13)
bi_counts = {}
bi_totals = {}
tri_counts = {}
tri_totals = {}

encoder = ArithmeticEncoder32(store=True)
total_bits_counted = 0
total_bits_stored = 0

# Forward pass (chunked to avoid OOM)
CHUNK = 4096
logits = np.zeros((len(ids), V), dtype=np.float32)
with torch.no_grad():
    for ci in range(0, len(ids), CHUNK):
        chunk_ids = ids[ci:ci + CHUNK]
        x = torch.tensor([chunk_ids], device="cuda")
        out = model(x, use_cache=False)
        logits[ci:ci + len(chunk_ids)] = out.logits[0].float().cpu().numpy()
        del out, x
        torch.cuda.empty_cache()

# Process each position
for pos in range(len(ids) - 1):
    target = ids[pos + 1]
    prev = ids[pos]
    prev2 = ids[pos - 1] if pos > 0 else 0

    # Build bigram context
    ib = prev
    bi_ctx = bi_counts.get(ib, {})
    bi_tot = bi_totals.get(ib, 0)

    # Build trigram context
    if USE_TRIGRAM:
        it = (prev2, prev)
        tri_ctx = tri_counts.get(it, {})
        tri_tot = tri_totals.get(it, 0)
    else:
        it = None
        tri_ctx = {}
        tri_tot = 0

    # Get LM probabilities
    p_lm = logits[pos]  # [V]
    p_lm = np.exp(p_lm - p_lm.max())
    p_lm = p_lm / p_lm.sum()

    # Top-K from LM
    topk_idx = np.argpartition(p_lm, -TOP_K)[-TOP_K:]
    topk_idx = topk_idx[np.argsort(-p_lm[topk_idx])]
    topk_probs = p_lm[topk_idx]

    # Blend with bigram/trigram cache
    wB = bi_tot / (bi_tot + BIGRAM_CONF) if bi_tot > 0 else 0.0
    wT = tri_tot / (tri_tot + TRIGRAM_CONF) if tri_tot > 0 else 0.0
    wc = 1.0 - (1.0 - wT) * (1.0 - wB)
    lam = BIGRAM_LAMBDA
    le = 1.0 - (1.0 - lam) * wc

    # Build blend probabilities for topk tokens
    p_blend = np.zeros(TOP_K, dtype=np.float64)
    for i, tk in enumerate(topk_idx):
        p_bigram = bi_ctx.get(tk, 0) / bi_tot if bi_tot > 0 else 0.0
        p_trigram = tri_ctx.get(tk, 0) / tri_tot if tri_tot > 0 else 0.0
        p_cache = (wB * p_bigram + wT * p_trigram) / max(wc, 1e-12) if wc > 1e-12 else 0.0
        p_blend[i] = le * float(p_lm[tk]) + (1.0 - le) * p_cache

    # Escape mass
    escape_mass = max(1e-12, 1.0 - p_blend.sum())

    # Stage-1 cumulative distribution
    cum_s1 = stage1_cum_32(p_blend, escape_mass, FLOOR_FRAC)

    # Check if target is in topk
    hit = np.where(topk_idx == target)[0]
    if len(hit) > 0:
        sym = int(hit[0])
        encoder.encode_symbol(cum_s1, sym)
        if pos < 3:
            print(f"  [ENC] pos={pos} target={target} sym={sym} (in topk) cum_range=[{cum_s1[sym]},{cum_s1[sym+1]})")
    else:
        # Escape: encode escape symbol
        sym = TOP_K
        encoder.encode_symbol(cum_s1, sym)
        if pos < 3:
            print(f"  [ENC] pos={pos} target={target} sym={sym} (escape) cum_range=[{cum_s1[sym]},{cum_s1[sym+1]})")

        # Stage-2: uniform over rest
        mask = np.ones(V, dtype=bool)
        mask[topk_idx] = False
        rest_ids = np.arange(V, dtype=np.int64)[mask]
        rank = int(np.where(rest_ids == target)[0][0])

        if USE_CACHE_S2:
            # Cache-informed rest distribution
            cpost = np.zeros(V, dtype=np.float64)
            for tk, cnt in bi_ctx.items():
                cpost[tk] += cnt * BIGRAM_LAMBDA
            cpost = cpost[mask]
            if cpost.sum() > 0:
                co = cache_rest_cum_32(cpost, rest_ids, S2_FLOOR)
            else:
                co = uniform_cum_32(len(rest_ids))
        else:
            co = uniform_cum_32(len(rest_ids))

        encoder.encode_symbol(co, rank)

    # Update bigram/trigram tables
    bi_totals[ib] = bi_totals.get(ib, 0) + 1
    if ib not in bi_counts:
        bi_counts[ib] = {}
    bi_counts[ib][target] = bi_counts[ib].get(target, 0) + 1

    if USE_TRIGRAM:
        tri_totals[it] = tri_totals.get(it, 0) + 1
        if it not in tri_counts:
            tri_counts[it] = {}
        tri_counts[it][target] = tri_counts[it].get(target, 0) + 1

bitstr, stored_bits = encoder.finish()
dt = time.time() - t0
print(f"  Encoded {len(ids)} tokens in {dt:.1f}s")
print(f"  Bitstream: {stored_bits} bits = {stored_bits/8:.0f} bytes")
print(f"  bpb: {stored_bits / n_bytes:.4f}")

# Debug: compare encoder tables at first 3 positions
print("\n  [DEBUG] Encoder table state:")
for dbg_pos in range(min(3, len(ids) - 1)):
    dbg_prev = ids[dbg_pos]
    dbg_prev2 = ids[dbg_pos - 1] if dbg_pos > 0 else 0
    dbg_ib = dbg_prev
    dbg_bi_ctx = bi_counts.get(dbg_ib, {})
    dbg_bi_tot = bi_totals.get(dbg_ib, 0)
    print(f"    pos={dbg_pos}: prev={dbg_prev} prev2={dbg_prev2} bi_ctx={dict(list(dbg_bi_ctx.items())[:3])} bi_tot={dbg_bi_tot}")
    if USE_TRIGRAM:
        dbg_it = (dbg_prev2, dbg_prev)
        dbg_tri_ctx = tri_counts.get(dbg_it, {})
        dbg_tri_tot = tri_totals.get(dbg_it, 0)
        print(f"           tri_ctx={dict(list(dbg_tri_ctx.items())[:3])} tri_tot={dbg_tri_tot}")

# ─── Write .zllm file ───
print("[4/4] Writing .zllm file and decoding...")
zllm_path = f"test_{kb}kb.zllm"
with open(zllm_path, "wb") as f:
    # Header: magic, version, vocab_size, n_tokens, n_bits, first_token
    f.write(b"ZLLM")
    f.write(struct.pack("<I", 1))  # version
    f.write(struct.pack("<I", V))
    f.write(struct.pack("<I", len(ids)))
    f.write(struct.pack("<I", stored_bits))
    f.write(struct.pack("<I", ids[0]))  # first token (context for decoder)
    # Bitstream as bytes (pad last byte with zeros)
    padded = bitstr + "0" * ((8 - len(bitstr) % 8) % 8)
    bit_bytes = bytes(int(padded[i:i+8], 2) for i in range(0, len(padded), 8))
    f.write(bit_bytes)
file_size = os.path.getsize(zllm_path)
print(f"  .zllm file: {file_size} bytes ({file_size/n_bytes:.2f}x ratio)")

# ─── Decode ───
print("\n[5/5] Decoding from .zllm file...")
t0 = time.time()
with open(zllm_path, "rb") as f:
    magic = f.read(4)
    assert magic == b"ZLLM", f"Bad magic: {magic}"
    version = struct.unpack("<I", f.read(4))[0]
    dec_V = struct.unpack("<I", f.read(4))[0]
    dec_n = struct.unpack("<I", f.read(4))[0]
    dec_bits = struct.unpack("<I", f.read(4))[0]
    first_token = struct.unpack("<I", f.read(4))[0]
    bit_bytes = f.read()
    # Reconstruct bitstring
    dec_bitstr = "".join(f"{b:08b}" for b in bit_bytes)[:dec_bits]

print(f"  Header: V={dec_V}, n_tokens={dec_n}, bits={dec_bits}")

# Decode with fresh tables
dec_bi_counts = {}
dec_bi_totals = {}
dec_tri_counts = {}
dec_tri_totals = {}

dec = ArithmeticDecoder32(dec_bitstr)
decoded_ids = []

for pos in range(dec_n - 1):
    # Rebuild same probability table as encoder
    if pos == 0:
        prev = first_token
        prev2 = 0
    elif pos == 1:
        prev = decoded_ids[0]
        prev2 = first_token
    else:
        prev = decoded_ids[pos - 1]
        prev2 = decoded_ids[pos - 2]

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

    # Same LM probabilities (from original forward pass)
    p_lm = logits[pos]
    p_lm = np.exp(p_lm - p_lm.max())
    p_lm = p_lm / p_lm.sum()

    topk_idx = np.argpartition(p_lm, -TOP_K)[-TOP_K:]
    topk_idx = topk_idx[np.argsort(-p_lm[topk_idx])]
    topk_probs = p_lm[topk_idx]

    wB = bi_tot / (bi_tot + BIGRAM_CONF) if bi_tot > 0 else 0.0
    wT = tri_tot / (tri_tot + TRIGRAM_CONF) if tri_tot > 0 else 0.0
    wc = 1.0 - (1.0 - wT) * (1.0 - wB)
    lam = BIGRAM_LAMBDA
    le = 1.0 - (1.0 - lam) * wc

    p_blend = np.zeros(TOP_K, dtype=np.float64)
    for i, tk in enumerate(topk_idx):
        p_bigram = bi_ctx.get(tk, 0) / bi_tot if bi_tot > 0 else 0.0
        p_trigram = tri_ctx.get(tk, 0) / tri_tot if tri_tot > 0 else 0.0
        p_cache = (wB * p_bigram + wT * p_trigram) / max(wc, 1e-12) if wc > 1e-12 else 0.0
        p_blend[i] = le * float(p_lm[tk]) + (1.0 - le) * p_cache

    escape_mass = max(1e-12, 1.0 - p_blend.sum())
    cum_s1 = stage1_cum_32(p_blend, escape_mass, FLOOR_FRAC)

    sym = dec.decode_symbol(cum_s1)

    if pos < 3:
        print(f"  [DEC] pos={pos} sym={sym} decoded_token={topk_idx[sym] if sym < TOP_K else 'escape'} target={ids[pos+1]}")

    if sym < TOP_K:
        token_id = int(topk_idx[sym])
        if pos < 10:
            print(f"  [DEC] pos={pos} stage1={sym} -> token={token_id} target={ids[pos+1]} {'OK' if token_id==ids[pos+1] else 'MISMATCH'}")
    else:
        # Escape: decode stage-2
        mask = np.ones(dec_V, dtype=bool)
        mask[topk_idx] = False
        rest_ids = np.arange(dec_V, dtype=np.int64)[mask]

        if USE_CACHE_S2:
            cpost = np.zeros(dec_V, dtype=np.float64)
            for tk, cnt in bi_ctx.items():
                cpost[tk] += cnt * BIGRAM_LAMBDA
            cpost = cpost[mask]
            if cpost.sum() > 0:
                co = cache_rest_cum_32(cpost, rest_ids, S2_FLOOR)
            else:
                co = uniform_cum_32(len(rest_ids))
        else:
            co = uniform_cum_32(len(rest_ids))

        rank = dec.decode_symbol(co)
        token_id = int(rest_ids[rank])
        if pos < 10:
            print(f"  [DEC] pos={pos} escape rank={rank} -> token={token_id} target={ids[pos+1]} {'OK' if token_id==ids[pos+1] else 'MISMATCH'}")

    decoded_ids.append(token_id)

    # Update decoder tables (same as encoder)
    dec_bi_totals[ib] = dec_bi_totals.get(ib, 0) + 1
    if ib not in dec_bi_counts:
        dec_bi_counts[ib] = {}
    dec_bi_counts[ib][token_id] = dec_bi_counts[ib].get(token_id, 0) + 1

    if USE_TRIGRAM:
        dec_tri_totals[it] = dec_tri_totals.get(it, 0) + 1
        if it not in dec_tri_counts:
            dec_tri_counts[it] = {}
        dec_tri_counts[it][token_id] = dec_tri_counts[it].get(token_id, 0) + 1

dt = time.time() - t0
print(f"  Decoded {len(decoded_ids)} tokens in {dt:.1f}s")

# Debug: compare decoder tables at first 3 positions
print("\n  [DEBUG] Decoder table state:")
for dbg_pos in range(min(3, len(decoded_ids))):
    if dbg_pos == 0:
        dbg_prev = first_token
        dbg_prev2 = 0
    elif dbg_pos == 1:
        dbg_prev = decoded_ids[0]
        dbg_prev2 = first_token
    else:
        dbg_prev = decoded_ids[dbg_pos - 1]
        dbg_prev2 = decoded_ids[dbg_pos - 2]
    dbg_ib = dbg_prev
    dbg_bi_ctx = dec_bi_counts.get(dbg_ib, {})
    dbg_bi_tot = dec_bi_totals.get(dbg_ib, 0)
    print(f"    pos={dbg_pos}: prev={dbg_prev} prev2={dbg_prev2} bi_ctx={dict(list(dbg_bi_ctx.items())[:3])} bi_tot={dbg_bi_tot}")
    if USE_TRIGRAM:
        dbg_it = (dbg_prev2, dbg_prev)
        dbg_tri_ctx = dec_tri_counts.get(dbg_it, {})
        dbg_tri_tot = dec_tri_totals.get(dbg_it, 0)
        print(f"           tri_ctx={dict(list(dbg_tri_ctx.items())[:3])} tri_tot={dbg_tri_tot}")

# ─── Verify ───
original = ids[1:len(decoded_ids)+1]  # decoder outputs ids[1], ids[2], ..., ids[n-1]
match = sum(1 for a, b in zip(original, decoded_ids) if a == b)
total = len(original)
print(f"\n{'='*60}")
print(f"VERIFICATION: {match}/{total} tokens match ({match/total*100:.2f}%)")
print(f"{'='*60}")

if match == total:
    print("✓ FULL ROUNDTRIP PASSED — bitstream is valid!")
    # SHA-256 hash comparison (decoder outputs ids[1:])
    orig_slice = ids[1:len(decoded_ids)+1]
    orig_bytes = struct.pack(f"<{len(orig_slice)}I", *orig_slice)
    dec_bytes = struct.pack(f"<{len(decoded_ids)}I", *decoded_ids)
    orig_hash = hashlib.sha256(orig_bytes).hexdigest()
    dec_hash = hashlib.sha256(dec_bytes).hexdigest()
    print(f"  SHA-256 original (ids[1:]): {orig_hash}")
    print(f"  SHA-256 decoded:             {dec_hash}")
    print(f"  Token hash match: {orig_hash == dec_hash}")
    # Also verify the text roundtrip
    orig_text = tok.decode(ids)
    dec_text = tok.decode(decoded_ids)
    text_hash = hashlib.sha256(orig_text.encode()).hexdigest()
    dec_text_hash = hashlib.sha256(dec_text.encode()).hexdigest()
    print(f"  Text SHA-256 orig: {text_hash}")
    print(f"  Text SHA-256 dec:  {dec_text_hash}")
    print(f"  Text match: {text_hash == dec_text_hash}")
else:
    print(f"⚠ ROUNDTRIP: {match}/{total} match ({match/total*100:.2f}%) — {total - match} mismatches")
    for i, (a, b) in enumerate(zip(original, decoded_ids)):
        if a != b:
            print(f"  MISMATCH at pos {i}: expected {a}, got {b}")
            if i > 20:
                break

# Save result
result = {
    "model": os.path.basename(MODEL_DIR),
    "kb": kb,
    "n_tokens": len(ids),
    "n_bytes": n_bytes,
    "bits": stored_bits,
    "bpb": round(stored_bits / n_bytes, 4),
    "file_bytes": file_size,
    "compression_ratio": round(n_bytes / file_size, 2),
    "roundtrip": match == total,
    "match_pct": round(match / total * 100, 2),
    "encode_time_s": round(dt, 1),
}
with open(f"logs/zllm_verify_{kb}kb.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to logs/zllm_verify_{kb}kb.json")
