"""SmolLM2 + bigram-cache ensemble (v2) on enwik8.

v1 (unigram cache) failed: hurt ratio + slowed 2.5x.
v2 fixes both causes:
1. Bigram (order-1) predictor: captures XML tag sequences, the dominant
   repeat structure in enwik8. Unigram was too weak.
2. Confidence gating: blend weight scales with bigram row evidence, so
   cold rows fall back to pure LLM (fixes v1's early uniform noise).
3. No giant float32 transfers: blend happens on small candidate unions
   on CPU; stage-1 rows are batch-counted with ONE numba call per block.

Causality: bigram counts update AFTER each block is coded; the decoder
mirrors exactly, so sync is preserved.
"""

import torch
import numpy as np
import sys
import time
import json
import os
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from transformers import AutoModelForCausalLM, AutoTokenizer
from real_compression import (
    ArithmeticEncoder, ArithmeticDecoder, nb_encode_count,
)
from bpe_compress import (
    MODEL_DIR, TOP_K, BLOCK_TOKENS, TAIL_GROUP,
    stage1_cum, uniform_cum, tail_group_rank, n_tail_groups,
)

# ---- v2 ensemble config ----
BIGRAM_LAMBDA = float(os.environ.get("BIGRAM_LAMBDA", "0.85"))
BIGRAM_CONF = 10.0     # row evidence needed for full weight
PREFILTER = 8192       # LLM prefilter size for candidate union


def main():
    print("=" * 70)
    print("SMOLLM2 + BIGRAM ENSEMBLE V2 ON ENWIK8")
    print("=" * 70)
    device = torch.device("cuda")
    t0 = time.time()

    print("\n[1/4] Loading model + tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, torch_dtype=torch.float16
    ).to(device).eval()
    V = model.config.vocab_size
    print(f"  Vocab: {V}")
    print(f"  Blend: bigram λ={BIGRAM_LAMBDA}, conf={BIGRAM_CONF}, prefilter={PREFILTER}")

    print("\n[2/4] Loading enwik8 sample...")
    import os as _os
    _off_mb = int(_os.environ.get("ENWIK8_OFFSET_MB", "0"))
    with open("./data/cloud/enwik8", "rb") as f:
        f.seek(_off_mb * 1024 * 1024)
        raw = f.read(100 * 1024)
    print(f"  Offset: {_off_mb}MB")
    text = raw.decode("utf-8", errors="ignore")
    ids = tok.encode(text)
    print(f"  Chars: {len(text)}, tokens: {len(ids)}")
    n_bytes = len(text.encode("utf-8"))

    # Precompute the 3 possible offset-stage cums (group sizes fixed)
    n_rest_global = V - TOP_K
    n_groups_global = n_tail_groups(n_rest_global)
    cum_g_global = uniform_cum(n_groups_global)
    cum_o_cache = {}
    for _g in range(n_groups_global):
        _gs = TAIL_GROUP if _g < n_groups_global - 1 else n_rest_global - _g * TAIL_GROUP
        cum_o_cache[_g] = uniform_cum(_gs)

    print("\n[3/4] Coding with bigram blend...")
    total_bits, n_escapes = 0, 0
    n_verify_ok, n_verify_fail = 0, 0
    all_ids = np.arange(V)
    bi_counts = {}   # prev_id -> {next_id: count}
    bi_totals = {}   # prev_id -> total
    prev_tail = None

    torch.cuda.synchronize()
    t = time.time()
    with torch.no_grad():
        for b0 in range(0, len(ids), BLOCK_TOKENS):
            block = ids[b0:b0 + BLOCK_TOKENS]
            if len(block) < 2:
                if block:
                    prev_tail = block[-1]
                continue
            x = torch.tensor([block], device=device)
            logits = model(x).logits[0].float().cpu().numpy()
            logits = logits - logits.max(axis=1, keepdims=True)
            e = np.exp(logits, dtype=np.float64)
            probs = e / e.sum(axis=1, keepdims=True)

            # Per-position blended top-k; accumulate stage-1 rows for
            # ONE batched numba count per block.
            s1_cums, s1_syms = [], []
            esc_infos = []  # (g, off, gsize_idx) per escape, in order
            for t_idx in range(len(block) - 1):
                p = probs[t_idx]
                prev = block[t_idx]
                target = block[t_idx + 1]
                row = bi_counts.get(prev)
                if row:
                    rtot = bi_totals[prev]
                    w = rtot / (rtot + BIGRAM_CONF)
                    pre = np.argpartition(p, -PREFILTER)[-PREFILTER:]
                    rk = np.fromiter(row.keys(), dtype=np.int64, count=len(row))
                    rv = np.fromiter(row.values(), dtype=np.float64, count=len(row))
                    cand = np.union1d(pre, rk)
                    pos = np.searchsorted(cand, rk)
                    pb = np.zeros(len(cand))
                    pb[pos] = rv / rtot
                    lam_eff = 1.0 - (1.0 - BIGRAM_LAMBDA) * w
                    score = lam_eff * p[cand] + (1.0 - lam_eff) * pb
                    o = np.argpartition(score, -TOP_K)[-TOP_K:]
                    o = o[np.argsort(-score[o])]
                    topk_idx, topk_p = cand[o], score[o]
                else:
                    topk_idx = np.argpartition(p, -TOP_K)[-TOP_K:]
                    topk_idx = topk_idx[np.argsort(-p[topk_idx])]
                    topk_p = p[topk_idx]
                esc_mass = max(1e-12, 1.0 - topk_p.sum())
                cum1 = stage1_cum(topk_p, esc_mass)

                hit = np.where(topk_idx == target)[0]
                if len(hit):
                    s1_cums.append(cum1)
                    s1_syms.append(int(hit[0]))
                    esc_infos.append(None)
                else:
                    n_escapes += 1
                    s1_cums.append(cum1)
                    s1_syms.append(TOP_K)
                    mask = np.ones(V, dtype=bool)
                    mask[topk_idx] = False
                    rest = all_ids[mask]
                    rank = int(np.where(rest == target)[0][0])
                    g, off = tail_group_rank(rank)
                    esc_infos.append((g, off))

            # ONE numba call for all stage-1 rows in this block
            C = np.ascontiguousarray(np.stack(s1_cums))
            S = np.array(s1_syms, dtype=np.int64)
            n1 = int(nb_encode_count(C, S))
            if n1 < 0:
                raise ArithmeticError(f"stage-1 encoder error {n1} at block {b0}")
            total_bits += n1
            # Escape stage-2 rows (rare): tiny individual counts
            n2_total = 0
            for (g, off) in [e for e in esc_infos if e is not None]:
                cg = cum_g_global
                n2_total += int(nb_encode_count(
                    np.ascontiguousarray(cg).reshape(1, -1),
                    np.array([g], dtype=np.int64)))
                co = cum_o_cache[g]
                n2_total += int(nb_encode_count(
                    np.ascontiguousarray(co).reshape(1, -1),
                    np.array([off], dtype=np.int64)))
            total_bits += n2_total

            # Verify 200 positions SPREAD across the whole file
            # (every ~130th token), so the hot-cache blend path is exercised.
            # (recompute the needed rows with store=True encoders)
            for t_idx in range(len(block) - 1):
                gpos = b0 + t_idx
                if gpos % 130 != 0 or n_verify_ok + n_verify_fail >= 200:
                    continue
                p = probs[t_idx]
                prev = block[t_idx]
                target = block[t_idx + 1]
                row = bi_counts.get(prev)
                if row:
                    rtot = bi_totals[prev]
                    w = rtot / (rtot + BIGRAM_CONF)
                    pre = np.argpartition(p, -PREFILTER)[-PREFILTER:]
                    rk = np.fromiter(row.keys(), dtype=np.int64, count=len(row))
                    rv = np.fromiter(row.values(), dtype=np.float64, count=len(row))
                    cand = np.union1d(pre, rk)
                    pos = np.searchsorted(cand, rk)
                    pb = np.zeros(len(cand))
                    pb[pos] = rv / rtot
                    lam_eff = 1.0 - (1.0 - BIGRAM_LAMBDA) * w
                    score = lam_eff * p[cand] + (1.0 - lam_eff) * pb
                    o = np.argpartition(score, -TOP_K)[-TOP_K:]
                    o = o[np.argsort(-score[o])]
                    v_topk, v_p = cand[o], score[o]
                else:
                    v_topk = np.argpartition(p, -TOP_K)[-TOP_K:]
                    v_topk = v_topk[np.argsort(-p[v_topk])]
                    v_p = p[v_topk]
                v_esc = max(1e-12, 1.0 - v_p.sum())
                v_cum = stage1_cum(v_p, v_esc)
                hit = np.where(v_topk == target)[0]
                if len(hit):
                    enc = ArithmeticEncoder(store=True)
                    enc.encode_symbol(v_cum, int(hit[0]))
                    bs, _ = enc.finish()
                    dec = ArithmeticDecoder(bs)
                    got = int(v_topk[dec.decode_symbol(v_cum)])
                else:
                    enc = ArithmeticEncoder(store=True)
                    enc.encode_symbol(v_cum, TOP_K)
                    bs, _ = enc.finish()
                    dec = ArithmeticDecoder(bs)
                    assert dec.decode_symbol(v_cum) == TOP_K
                    mask = np.ones(V, dtype=bool)
                    mask[v_topk] = False
                    rest = all_ids[mask]
                    rank = int(np.where(rest == target)[0][0])
                    ng = n_tail_groups(len(rest))
                    gg, oo = tail_group_rank(rank)
                    cg2 = uniform_cum(ng)
                    en2 = ArithmeticEncoder(store=True)
                    en2.encode_symbol(cg2, gg)
                    bs2, _ = en2.finish()
                    gs = TAIL_GROUP if gg < ng - 1 else len(rest) - gg * TAIL_GROUP
                    co2 = uniform_cum(gs)
                    en3 = ArithmeticEncoder(store=True)
                    en3.encode_symbol(co2, oo)
                    bs3, _ = en3.finish()
                    dg = ArithmeticDecoder(bs2).decode_symbol(cg2)
                    gs = TAIL_GROUP if dg < ng - 1 else len(rest) - dg * TAIL_GROUP
                    co3 = uniform_cum(gs)
                    got = int(rest[dg * TAIL_GROUP + ArithmeticDecoder(bs3).decode_symbol(co3)])
                if got == target:
                    n_verify_ok += 1
                else:
                    n_verify_fail += 1
                    print(f"  FAIL at block {b0}, pos {t_idx}")

            # Causal bigram update AFTER this block is coded
            seq = ([prev_tail] if prev_tail is not None else []) + block
            for a, b in zip(seq, seq[1:]):
                d = bi_counts.get(a)
                if d is None:
                    d = {}
                    bi_counts[a] = d
                d[b] = d.get(b, 0) + 1
                bi_totals[a] = bi_totals.get(a, 0) + 1
            prev_tail = block[-1]

            if (b0 // BLOCK_TOKENS + 1) % 2 == 0:
                print(f"  ... block {b0//BLOCK_TOKENS+1}/{(len(ids)+BLOCK_TOKENS-1)//BLOCK_TOKENS}, "
                      f"escapes: {n_escapes}, bigram keys: {len(bi_counts)}", flush=True)

    torch.cuda.synchronize()
    dt = time.time() - t
    print(f"\n  Total bits: {total_bits}")
    print(f"  bits/byte: {total_bits/n_bytes:.4f}")
    print(f"  escapes: {n_escapes}")
    print(f"  Verified: {n_verify_ok} lossless, fails: {n_verify_fail}")
    print(f"  Time: {dt:.1f}s")

    results = {
        "model": "SmolLM2-135M-bigram-ensemble-v2",
        "bigram_lambda": BIGRAM_LAMBDA,
        "bigram_conf": BIGRAM_CONF,
        "top_k": TOP_K,
        "n_bytes": n_bytes,
        "total_bits": total_bits,
        "bpb": total_bits / n_bytes,
        "escapes": n_escapes,
        "verified_ok": n_verify_ok,
        "verified_fail": n_verify_fail,
        "elapsed_s": time.time() - t0,
    }
    with open(f"./data/smollm2_ensemble_v2_off{_off_mb}_lam{BIGRAM_LAMBDA}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to data/smollm2_ensemble_v2_off{_off_mb}_lam{BIGRAM_LAMBDA}.json")


if __name__ == "__main__":
    main()
