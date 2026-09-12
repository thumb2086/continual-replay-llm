"""SmolLM2 + multi-order PPM-style ensemble (v3) on enwik8.

v2 (bigram only) reached 0.9706 mid-slice. v3 adds:
1. Trigram cache (order-2): sharper repetition capture.
2. Backoff chain: trigram -> bigram -> pure LLM, by row evidence.
3. Recency decay: halve all counts each block (topics drift).

Shared blend helper is used by BOTH the fast counting path and the
reference verify path, so they cannot diverge.
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

# ---- v3 config ----
LAMBDA = 0.75        # LLM weight when a backed-off order is active
THRESH_TRI = 3       # trigram row evidence needed
THRESH_BI = 5        # bigram row evidence needed
PREFILTER = 8192
DECAY = 0.5          # recency halving per block


class OrderCaches:
    """Causal bigram/trigram caches with recency decay."""

    def __init__(self):
        self.bi = {}   # prev -> {next: count}
        self.tri = {}  # (prev2, prev) -> {next: count}

    def query(self, prev2, prev):
        """Return (order, row_dict, row_total): highest order with evidence."""
        if prev2 is not None:
            row = self.tri.get((prev2, prev))
            if row:
                tot = 0
                for v in row.values():
                    tot += v
                if tot >= THRESH_TRI:
                    return 2, row, tot
        if prev is not None:
            row = self.bi.get(prev)
            if row:
                tot = 0
                for v in row.values():
                    tot += v
                if tot >= THRESH_BI:
                    return 1, row, tot
        return 0, None, 0

    def update_block(self, tokens, prev_tail):
        """Add all consecutive pairs/triples. Called AFTER block coded."""
        seq = ([prev_tail] if prev_tail is not None else []) + list(tokens)
        for a, b in zip(seq, seq[1:]):
            d = self.bi.get(a)
            if d is None:
                d = {}
                self.bi[a] = d
            d[b] = d.get(b, 0) + 1
        for a, b, c in zip(seq, seq[1:], seq[2:]):
            key = (a, b)
            d = self.tri.get(key)
            if d is None:
                d = {}
                self.tri[key] = d
            d[c] = d.get(c, 0) + 1
        # Recency decay (deterministic; decoder mirrors identically)
        for d in self.bi.values():
            for k in list(d.keys()):
                d[k] //= 2
                if d[k] <= 0:
                    del d[k]
        for d in self.tri.values():
            for k in list(d.keys()):
                d[k] //= 2
                if d[k] <= 0:
                    del d[k]
        return tokens[-1] if tokens else prev_tail


def blend_topk(p, prev2, prev, caches):
    """Shared helper: returns (topk_idx, topk_scores, order) via backoff chain.

    Used by both the fast counting path and the reference verify path.
    """
    order, row, rtot = caches.query(prev2, prev)
    if order == 0:
        topk_idx = np.argpartition(p, -TOP_K)[-TOP_K:]
        topk_idx = topk_idx[np.argsort(-p[topk_idx])]
        return topk_idx, p[topk_idx], 0
    # Candidate union: LLM prefilter + row keys
    pre = np.argpartition(p, -PREFILTER)[-PREFILTER:]
    rk = np.fromiter(row.keys(), dtype=np.int64, count=len(row))
    rv = np.fromiter(row.values(), dtype=np.float64, count=len(row))
    cand = np.union1d(pre, rk)
    pos = np.searchsorted(cand, rk)
    pb = np.zeros(len(cand))
    pb[pos] = rv / rtot
    w = rtot / (rtot + 10.0)
    lam_eff = 1.0 - (1.0 - LAMBDA) * w
    score = lam_eff * p[cand] + (1.0 - lam_eff) * pb
    o = np.argpartition(score, -TOP_K)[-TOP_K:]
    o = o[np.argsort(-score[o])]
    return cand[o], score[o], order


def main():
    print("=" * 70)
    print("SMOLLM2 + MULTI-ORDER ENSEMBLE V3 ON ENWIK8")
    print("=" * 70)
    device = torch.device("cuda")
    t0 = time.time()
    off_mb = int(os.environ.get("ENWIK8_OFFSET_MB", "50"))

    print("\n[1/4] Loading model + tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, torch_dtype=torch.float16
    ).to(device).eval()
    V = model.config.vocab_size
    print(f"  Vocab: {V}, backoff tri({THRESH_TRI})/bi({THRESH_BI}), λ={LAMBDA}")

    print("\n[2/4] Loading enwik8 sample...")
    with open("./data/cloud/enwik8", "rb") as f:
        f.seek(off_mb * 1024 * 1024)
        raw = f.read(100 * 1024)
    print(f"  Offset: {off_mb}MB")
    text = raw.decode("utf-8", errors="ignore")
    ids = tok.encode(text)
    print(f"  Chars: {len(text)}, tokens: {len(ids)}")
    n_bytes = len(text.encode("utf-8"))

    n_rest_global = V - TOP_K
    n_groups_global = n_tail_groups(n_rest_global)
    cum_g_global = uniform_cum(n_groups_global)
    cum_o_cache = {}
    for _g in range(n_groups_global):
        _gs = TAIL_GROUP if _g < n_groups_global - 1 else n_rest_global - _g * TAIL_GROUP
        cum_o_cache[_g] = uniform_cum(_gs)
    all_ids = np.arange(V)

    print("\n[3/4] Coding with backoff blend...")
    total_bits, n_escapes = 0, 0
    n_verify_ok, n_verify_fail = 0, 0
    n_hot_tri, n_hot_bi, n_cold = 0, 0, 0
    caches = OrderCaches()
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

            s1_cums, s1_syms = [], []
            esc_list = []  # (g, off) per escape
            # per-position context needs running prev2/prev INCLUDING
            # within-block history (causal: uses only already-coded tokens).
            # ctx layout: ctx[0]=prev_tail (or None), ctx[1:]=block.
            # Block position t_idx <=> ctx index t_idx+1, so its
            # previous token is ctx[t_idx] and prev2 is ctx[t_idx-1].
            ctx = [prev_tail] + block
            for t_idx in range(len(block) - 1):
                p = probs[t_idx]
                target = block[t_idx + 1]
                prev = ctx[t_idx]
                prev2 = ctx[t_idx - 1] if t_idx >= 1 else None
                topk_idx, topk_p, order = blend_topk(p, prev2, prev, caches)
                if order == 2:
                    n_hot_tri += 1
                elif order == 1:
                    n_hot_bi += 1
                else:
                    n_cold += 1
                esc_mass = max(1e-12, 1.0 - topk_p.sum())
                cum1 = stage1_cum(topk_p, esc_mass)
                hit = np.where(topk_idx == target)[0]
                if len(hit):
                    s1_cums.append(cum1)
                    s1_syms.append(int(hit[0]))
                    esc_list.append(None)
                else:
                    n_escapes += 1
                    s1_cums.append(cum1)
                    s1_syms.append(TOP_K)
                    mask = np.ones(V, dtype=bool)
                    mask[topk_idx] = False
                    rest = all_ids[mask]
                    rank = int(np.where(rest == target)[0][0])
                    g, off = tail_group_rank(rank)
                    esc_list.append((g, off))

            C = np.ascontiguousarray(np.stack(s1_cums))
            S = np.array(s1_syms, dtype=np.int64)
            n1 = int(nb_encode_count(C, S))
            if n1 < 0:
                raise ArithmeticError(f"stage-1 error {n1} at block {b0}")
            total_bits += n1
            for (g, off) in [e for e in esc_list if e is not None]:
                total_bits += int(nb_encode_count(
                    np.ascontiguousarray(cum_g_global).reshape(1, -1),
                    np.array([g], dtype=np.int64)))
                total_bits += int(nb_encode_count(
                    np.ascontiguousarray(cum_o_cache[g]).reshape(1, -1),
                    np.array([off], dtype=np.int64)))

            # Spread verification (every ~130th global position, 200 total)
            # using the SHARED blend helper + reference Python codec.
            # Runs BEFORE update_block so cache state matches coding time.
            vctx = [prev_tail] + block
            for t_idx in range(len(block) - 1):
                gpos = b0 + t_idx
                if gpos % 130 != 0 or n_verify_ok + n_verify_fail >= 200:
                    continue
                p = probs[t_idx]
                target = block[t_idx + 1]
                v_prev = vctx[t_idx]
                v_prev2 = vctx[t_idx - 1] if t_idx >= 1 else None
                v_topk, v_p, _ = blend_topk(p, v_prev2, v_prev, caches)
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

            prev_tail = caches.update_block(block, prev_tail)

            if (b0 // BLOCK_TOKENS + 1) % 2 == 0:
                print(f"  ... block {b0//BLOCK_TOKENS+1}, escapes: {n_escapes}", flush=True)

    torch.cuda.synchronize()
    dt = time.time() - t
    print(f"\n  Total bits: {total_bits}")
    print(f"  bits/byte: {total_bits/n_bytes:.4f}")
    print(f"  escapes: {n_escapes}")
    print(f"  Verified: {n_verify_ok} lossless, fails: {n_verify_fail}")
    print(f"  Order usage: tri={n_hot_tri}, bi={n_hot_bi}, cold={n_cold}")
    print(f"  Time: {dt:.1f}s")

    results = {
        "model": "SmolLM2-135M-ensemble-v3",
        "bigram_lambda": LAMBDA,
        "thresh_tri": THRESH_TRI,
        "thresh_bi": THRESH_BI,
        "decay": DECAY,
        "top_k": TOP_K,
        "n_bytes": n_bytes,
        "total_bits": total_bits,
        "bpb": total_bits / n_bytes,
        "escapes": n_escapes,
        "order_tri": n_hot_tri,
        "order_bi": n_hot_bi,
        "order_cold": n_cold,
        "verified_ok": n_verify_ok,
        "verified_fail": n_verify_fail,
        "elapsed_s": time.time() - t0,
    }
    with open(f"./data/smollm2_v3_off{off_mb}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved.")


if __name__ == "__main__":
    main()
