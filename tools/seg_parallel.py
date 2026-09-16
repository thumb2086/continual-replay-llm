"""Segment-parallel per-token compressor/decoder.

K segments, each processed independently with per-token KV-cache.
Each segment builds its own n-gram tables (segment-local context).
Encoder and decoder use identical forward method → verifiable roundtrip.
"""
import sys, os, time, json, hashlib, struct, multiprocessing as mp
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from ac32 import (
    ArithmeticEncoder32, ArithmeticDecoder32,
    stage1_cum_32, uniform_cum_32, cache_rest_cum_32, TOTAL32,
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
K_SEGMENTS = int(os.environ.get("K_SEGMENTS", "1"))

def forward_one(model, token_id, past_kv):
    x = torch.tensor([[token_id]], device="cuda")
    with torch.no_grad():
        out = model(x, use_cache=True, past_key_values=past_kv)
    logits = out.logits[0, -1].float().cpu().numpy()
    return logits, out.past_key_values

def encode_segment(model, token_ids, V, debug=False):
    """Encode a segment: per-token forward + arithmetic coding."""
    bi_counts, bi_totals = {}, {}
    tri_counts, tri_totals = {}, {}
    enc = ArithmeticEncoder32(store=True)
    past_kv = None
    n = len(token_ids)

    for i in range(n):
        logits, past_kv = forward_one(model, token_ids[i], past_kv)
        if debug and i < 3:
            print(f"    [ENC] i={i} token={token_ids[i]} logit_top3={np.argsort(logits)[-3:][::-1]}")

        p = np.exp(logits - logits.max())
        p = p / p.sum()

        topk = np.argpartition(p, -TOP_K)[-TOP_K:]
        topk = topk[np.argsort(-p[topk])]

        ib = token_ids[i]
        bi_ctx = bi_counts.get(ib, {})
        bi_tot = bi_totals.get(ib, 0)
        if USE_TRIGRAM and i > 0:
            it = (token_ids[i-1], token_ids[i])
            tri_ctx = tri_counts.get(it, {})
            tri_tot = tri_totals.get(it, 0)
        else:
            it = None
            tri_ctx, tri_tot = {}, 0

        wB = bi_tot / (bi_tot + BIGRAM_CONF) if bi_tot > 0 else 0.0
        wT = tri_tot / (tri_tot + TRIGRAM_CONF) if tri_tot > 0 else 0.0
        wc = 1.0 - (1.0 - wT) * (1.0 - wB)
        le = 1.0 - (1.0 - BIGRAM_LAMBDA) * wc

        p_blend = np.zeros(TOP_K, dtype=np.float64)
        for j, tk in enumerate(topk):
            pb = bi_ctx.get(tk, 0) / bi_tot if bi_tot > 0 else 0.0
            pt = tri_ctx.get(tk, 0) / tri_tot if tri_tot > 0 else 0.0
            pc = (wB * pb + wT * pt) / max(wc, 1e-12) if wc > 1e-12 else 0.0
            p_blend[j] = le * float(p[tk]) + (1.0 - le) * pc

        esc = max(1e-12, 1.0 - p_blend.sum())
        cum = stage1_cum_32(p_blend, esc, FLOOR_FRAC)

        if i < n - 1:
            target = token_ids[i + 1]
            hit = np.where(topk == target)[0]
            if len(hit) > 0:
                enc.encode_symbol(cum, int(hit[0]))
            else:
                enc.encode_symbol(cum, TOP_K)
                mask = np.ones(V, dtype=bool); mask[topk] = False
                rest = np.arange(V, dtype=np.int64)[mask]
                rank = int(np.where(rest == target)[0][0])
                co = uniform_cum_32(len(rest))
                enc.encode_symbol(co, rank)

        # Update tables
        bi_totals[ib] = bi_totals.get(ib, 0) + 1
        if ib not in bi_counts: bi_counts[ib] = {}
        if i < n - 1:
            bi_counts[ib][token_ids[i+1]] = bi_counts[ib].get(token_ids[i+1], 0) + 1
        if it is not None:
            tri_totals[it] = tri_totals.get(it, 0) + 1
            if it not in tri_counts: tri_counts[it] = {}
            if i < n - 1:
                tri_counts[it][token_ids[i+1]] = tri_counts[it].get(token_ids[i+1], 0) + 1

    bitstr, bits = enc.finish()
    del past_kv
    torch.cuda.empty_cache()
    return bitstr, bits

def decode_segment(model, bitstr, n_tokens, first_token, V, debug=False):
    """Decode a segment: per-token forward + arithmetic decoding."""
    bi_counts, bi_totals = {}, {}
    tri_counts, tri_totals = {}, {}
    dec = ArithmeticDecoder32(bitstr)
    past_kv = None
    decoded = []

    for i in range(n_tokens - 1):
        prev = decoded[-1] if decoded else first_token
        logits, past_kv = forward_one(model, prev, past_kv)
        if debug and i < 3:
            print(f"    [DEC] i={i} prev={prev} logit_top3={np.argsort(logits)[-3:][::-1]}")
        p = np.exp(logits - logits.max())
        p = p / p.sum()

        topk = np.argpartition(p, -TOP_K)[-TOP_K:]
        topk = topk[np.argsort(-p[topk])]

        ib = prev
        bi_ctx = bi_counts.get(ib, {})
        bi_tot = bi_totals.get(ib, 0)
        if USE_TRIGRAM and len(decoded) >= 1:
            it = (decoded[-1], prev) if decoded else (first_token, prev)
            tri_ctx = tri_counts.get(it, {})
            tri_tot = tri_totals.get(it, 0)
        else:
            it = None
            tri_ctx, tri_tot = {}, 0

        wB = bi_tot / (bi_tot + BIGRAM_CONF) if bi_tot > 0 else 0.0
        wT = tri_tot / (tri_tot + TRIGRAM_CONF) if tri_tot > 0 else 0.0
        wc = 1.0 - (1.0 - wT) * (1.0 - wB)
        le = 1.0 - (1.0 - BIGRAM_LAMBDA) * wc

        p_blend = np.zeros(TOP_K, dtype=np.float64)
        for j, tk in enumerate(topk):
            pb = bi_ctx.get(tk, 0) / bi_tot if bi_tot > 0 else 0.0
            pt = tri_ctx.get(tk, 0) / tri_tot if tri_tot > 0 else 0.0
            pc = (wB * pb + wT * pt) / max(wc, 1e-12) if wc > 1e-12 else 0.0
            p_blend[j] = le * float(p[tk]) + (1.0 - le) * pc

        esc = max(1e-12, 1.0 - p_blend.sum())
        cum = stage1_cum_32(p_blend, esc, FLOOR_FRAC)
        sym = dec.decode_symbol(cum)

        if sym < TOP_K:
            token_id = int(topk[sym])
        else:
            mask = np.ones(V, dtype=bool); mask[topk] = False
            rest = np.arange(V, dtype=np.int64)[mask]
            co = uniform_cum_32(len(rest))
            rank = dec.decode_symbol(co)
            token_id = int(rest[rank])

        decoded.append(token_id)
        bi_totals[ib] = bi_totals.get(ib, 0) + 1
        if ib not in bi_counts: bi_counts[ib] = {}
        bi_counts[ib][token_id] = bi_counts[ib].get(token_id, 0) + 1
        if it is not None:
            tri_totals[it] = tri_totals.get(it, 0) + 1
            if it not in tri_counts: tri_counts[it] = {}
            tri_counts[it][token_id] = tri_counts[it].get(token_id, 0) + 1

    del past_kv
    torch.cuda.empty_cache()
    return decoded

# ─── Main ───
if __name__ == "__main__":
    kb = int(os.environ.get("ENWIK8_KB", "100"))
    print(f"Segment-parallel per-token: K={K_SEGMENTS}, {kb}KB")

    print("[1] Loading model...")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, torch_dtype=torch.float16
    ).cuda().eval()
    V = tok.vocab_size

    enwik8_path = os.environ.get("ENWIK8_PATH", "./data/cloud/enwik8")
    offset_mb = int(os.environ.get("ENWIK8_OFFSET_MB", "50"))
    with open(enwik8_path, "rb") as f:
        f.seek(offset_mb * 1024 * 1024)
        raw = f.read(kb * 1024)
    ids = tok.encode(raw.decode("utf-8", errors="ignore"))
    n_bytes = len(raw)
    print(f"  {len(ids)} tokens, {n_bytes} bytes")

    # Split into K segments
    seg_len = len(ids) // K_SEGMENTS
    segments = [ids[i*seg_len:(i+1)*seg_len] for i in range(K_SEGMENTS)]
    if len(ids) % K_SEGMENTS:
        segments[-1] = ids[(K_SEGMENTS-1)*seg_len:]

    # Encode each segment
    print(f"\n[2] Encoding {K_SEGMENTS} segments...")
    t0 = time.time()
    bitstreams = []
    total_bits = 0
    for si, seg in enumerate(segments):
        print(f"  segment {si}/{K_SEGMENTS} ({len(seg)} tokens)...", flush=True)
        bs, bits = encode_segment(model, seg, V, debug=(si==0))
        bitstreams.append((bs, bits))
        total_bits += bits
    dt_enc = time.time() - t0
    bpb = total_bits / n_bytes
    print(f"  Encoded in {dt_enc:.1f}s, {total_bits} bits, bpb={bpb:.4f}")

    # Write .zllm
    zllm_path = f"test_{kb}kb_k{K_SEGMENTS}.zllm"
    with open(zllm_path, "wb") as f:
        f.write(b"ZLLM")
        f.write(struct.pack("<I", 2))  # version 2 (segment-parallel)
        f.write(struct.pack("<I", V))
        f.write(struct.pack("<I", len(ids)))
        f.write(struct.pack("<I", K_SEGMENTS))
        f.write(struct.pack("<I", ids[0]))
        for bs, bits in bitstreams:
            f.write(struct.pack("<I", bits))
            padded = bs + "0" * ((8 - len(bs) % 8) % 8)
            f.write(bytes(int(padded[i:i+8], 2) for i in range(0, len(padded), 8)))
    file_size = os.path.getsize(zllm_path)
    print(f"  .zllm: {file_size} bytes ({file_size/n_bytes:.2f}x)")

    # Decode each segment
    print(f"\n[3] Decoding {K_SEGMENTS} segments...")
    t0 = time.time()
    all_decoded = []
    with open(zllm_path, "rb") as f:
        magic = f.read(4)
        version = struct.unpack("<I", f.read(4))[0]
        dec_V = struct.unpack("<I", f.read(4))[0]
        dec_n = struct.unpack("<I", f.read(4))[0]
        dec_K = struct.unpack("<I", f.read(4))[0]
        first_token = struct.unpack("<I", f.read(4))[0]
        for si in range(dec_K):
            seg_bits = struct.unpack("<I", f.read(4))[0]
            seg_bytes = (seg_bits + 7) // 8
            seg_bitbytes = f.read(seg_bytes)
            seg_bs = "".join(f"{b:08b}" for b in seg_bitbytes)[:seg_bits]
            seg_n = len(segments[si])
            print(f"  segment {si}/{dec_K} ({seg_n} tokens, {seg_bits} bits)...", flush=True)
            decoded = decode_segment(model, seg_bs, seg_n, first_token, dec_V, debug=(si==0))
            all_decoded.extend(decoded)
    dt_dec = time.time() - t0
    print(f"  Decoded in {dt_dec:.1f}s")

    # Verify
    original = ids[1:len(all_decoded)+1]
    match = sum(1 for a, b in zip(original, all_decoded) if a == b)
    total = len(original)
    print(f"\n{'='*60}")
    print(f"VERIFICATION: {match}/{total} ({match/total*100:.2f}%)")
    print(f"{'='*60}")

    if match == total:
        print("FULL ROUNDTRIP PASSED!")
        ob = struct.pack(f"<{len(original)}I", *original)
        db = struct.pack(f"<{len(all_decoded)}I", *all_decoded)
        print(f"  SHA-256: {'match' if hashlib.sha256(ob).hexdigest()==hashlib.sha256(db).hexdigest() else 'MISMATCH'}")
    else:
        print(f"MISMATCH: {total-match} errors")

    result = {"model": os.path.basename(MODEL_DIR), "kb": kb, "k_segments": K_SEGMENTS,
              "n_tokens": len(ids), "n_bytes": n_bytes, "bits": total_bits,
              "bpb": round(bpb, 4), "file_bytes": file_size, "ratio": round(n_bytes/file_size, 2),
              "roundtrip": match == total, "match_pct": round(match/total*100, 2),
              "encode_s": round(dt_enc, 1), "decode_s": round(dt_dec, 1)}
    with open(f"data/zllm_k{K_SEGMENTS}_{kb}kb.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to data/zllm_k{K_SEGMENTS}_{kb}kb.json")
