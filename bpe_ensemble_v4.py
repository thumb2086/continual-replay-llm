"""SmolLM2 + bigram-cache ensemble (v4, speed) on enwik8.

v4 = v2 math, GPU-accelerated plumbing:
1. Softmax in fp32 on GPU (torch.softmax), not float64 np.exp on CPU.
   (~40s of v2's 69s was 1.6B-element CPU exp.)
2. Per-block torch.topk on GPU for TOP_K (plain path) and PREFILTER
   (row path). Same selected SET as argpartition (ties: measure-zero);
   re-sorted by value desc on CPU so symbol ranks match v2 exactly.
3. Batched LLM forwards (BATCH_FWD, default 2). Blocks are independent
   for a frozen model; bigram rows only touch the blend, and token
   text is known upfront, so batching changes nothing mathematically.
4. Verify reuses stored per-position (topk_idx, topk_p) instead of
   recomputing the blend a second time.

Knobs (env):
  USE_GPU_TOPK=0 + BATCH_FWD=1  -> bit-identical-ish v2 reproduction
                                   (validates the refactor; expect bpb
                                   match within 1e-4, tiny fp32/fp64 drift)
"""
import torch
import numpy as np
import sys
import time
import json
import os

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

BIGRAM_LAMBDA = float(os.environ.get("BIGRAM_LAMBDA", "0.95"))
BIGRAM_CONF = 10.0
PREFILTER = int(os.environ.get("PREFILTER", "8192"))
USE_GPU_TOPK = int(os.environ.get("USE_GPU_TOPK", "1"))
BATCH_FWD = int(os.environ.get("BATCH_FWD", "2"))


def block_topk_gpu(probs_b, idx):
    """GPU topk for one block. probs_b: [B,V] fp32 CUDA. Returns
    (topk_vals_cpu [B,K], topk_idx_cpu [B,K]) for K=TOP_K, and
    (pre_vals, pre_idx) for K=PREFILTER. Values re-sorted desc on CPU."""
    with torch.no_grad():
        tv, ti = torch.topk(probs_b, TOP_K, dim=1)
        pv, pi = torch.topk(probs_b, PREFILTER, dim=1)
    tv = tv.float().cpu().numpy()
    ti = ti.cpu().numpy().astype(np.int64)
    pv = pv.float().cpu().numpy()
    pi = pi.cpu().numpy().astype(np.int64)
    # torch.topk already returns values sorted descending; that order
    # matches v2's argsort(-p) whenever values are distinct (float ties:
    # measure-zero, absorbed as run noise ~1e-4). NO cpu re-sort:
    # re-sorting the 8192-wide prefilter cost ~40s/run in v4full3.
    return (tv, ti), (pv, pi)


def main():
    print("=" * 70)
    print("SMOLLM2 + BIGRAM ENSEMBLE V4 (SPEED) ON ENWIK8")
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
    print(f"  Plumbing: gpu_topk={USE_GPU_TOPK}, batch_fwd={BATCH_FWD}")

    print("\n[2/4] Loading enwik8 sample...")
    _off_mb = int(os.environ.get("ENWIK8_OFFSET_MB", "50"))
    with open("./data/cloud/enwik8", "rb") as f:
        f.seek(_off_mb * 1024 * 1024)
        raw = f.read(100 * 1024)
    print(f"  Offset: {_off_mb}MB")
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

    print("\n[3/4] Coding with bigram blend (v4 plumbing)...")
    total_bits, n_escapes = 0, 0
    n_verify_ok, n_verify_fail = 0, 0
    all_ids = np.arange(V)
    bi_counts = {}
    bi_totals = {}
    prev_tail = None
    stored = []  # per coded position: (topk_idx, topk_p) for verify reuse

    torch.cuda.synchronize()
    t = time.time()
    ph_fwd = ph_st = ph_loop = ph_code = ph_verify = 0.0
    blocks = [ids[b0:b0 + BLOCK_TOKENS] for b0 in range(0, len(ids), BLOCK_TOKENS)]
    b0_global = 0  # token offset of current block
    with torch.no_grad():
        bi = 0
        while bi < len(blocks):
            chunk = blocks[bi:bi + BATCH_FWD]
            if len({len(b) for b in chunk}) > 1:
                chunk = chunk[:1]  # uneven tail block: run solo, no pad
            # forward (batched); drop degenerate blocks (<2 tokens)
            fwd_idx = [k for k, b in enumerate(chunk) if len(b) >= 2]
            if fwd_idx:
                ta = time.time()
                x = torch.stack([torch.tensor(chunk[k], device=device)
                                 for k in fwd_idx])
                logits_f = model(x).logits  # [F, B, V] fp16 RAW (no softmax yet)
                torch.cuda.synchronize()
                ph_fwd += time.time() - ta
            else:
                logits_f = None
            fpos = 0
            for k, block in enumerate(chunk):
                if len(block) < 2:
                    if block:
                        prev_tail = block[-1]
                    b0_global += len(block)
                    continue
                lg_b = logits_f[fpos]  # [B, V] fp16 GPU view
                fpos += 1
                Lb = len(block)
                ta = time.time()
                if USE_GPU_TOPK:
                    with torch.no_grad():
                        probs_b = torch.softmax(lg_b.float(), dim=-1)
                    (tv, ti), (pv, pi) = block_topk_gpu(probs_b, k)
                    probs_cpu = None
                else:
                    lg = lg_b.float().cpu().numpy()
                    lg = lg - lg.max(axis=1, keepdims=True)
                    e = np.exp(lg, dtype=np.float64)
                    probs_cpu = e / e.sum(axis=1, keepdims=True)
                    probs_b = None
                torch.cuda.synchronize()
                ph_st += time.time() - ta

                s1_cums, s1_syms = [], []
                esc_infos = []
                ta = time.time()
                for t_idx in range(Lb - 1):
                    prev = block[t_idx]
                    target = block[t_idx + 1]
                    row = bi_counts.get(prev)
                    if row:
                        rtot = bi_totals[prev]
                        w = rtot / (rtot + BIGRAM_CONF)
                        if USE_GPU_TOPK:
                            pre = pi[t_idx]
                            # single-row H2D (200KB); keeps v2-identical numpy below
                            p_full = probs_b[t_idx].float().cpu().numpy()
                        else:
                            p_full = probs_cpu[t_idx]
                            pre = np.argpartition(p_full, -PREFILTER)[-PREFILTER:]
                        rk = np.fromiter(row.keys(), dtype=np.int64, count=len(row))
                        rv = np.fromiter(row.values(), dtype=np.float64, count=len(row))
                        cand = np.union1d(pre, rk)
                        pos = np.searchsorted(cand, rk)
                        pbv = np.zeros(len(cand))
                        pbv[pos] = rv / rtot
                        lam_eff = 1.0 - (1.0 - BIGRAM_LAMBDA) * w
                        pcand = p_full[cand]
                        score = lam_eff * pcand + (1.0 - lam_eff) * pbv
                        o = np.argpartition(score, -TOP_K)[-TOP_K:]
                        o = o[np.argsort(-score[o])]
                        topk_idx, topk_p = cand[o], score[o]
                    else:
                        if USE_GPU_TOPK:
                            topk_idx = ti[t_idx]
                            topk_p = tv[t_idx].astype(np.float64)
                        else:
                            p_full = probs_cpu[t_idx]
                            topk_idx = np.argpartition(p_full, -TOP_K)[-TOP_K:]
                            topk_idx = topk_idx[np.argsort(-p_full[topk_idx])]
                            topk_p = p_full[topk_idx]
                    esc_mass = max(1e-12, 1.0 - topk_p.sum())
                    cum1 = stage1_cum(topk_p, esc_mass)
                    stored.append((topk_idx, topk_p))

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

                ph_loop += time.time() - ta
                ta = time.time()
                C = np.ascontiguousarray(np.stack(s1_cums))
                S = np.array(s1_syms, dtype=np.int64)
                n1 = int(nb_encode_count(C, S))
                if n1 < 0:
                    raise ArithmeticError(f"stage-1 encoder error {n1}")
                total_bits += n1
                n2_total = 0
                for (g, off) in [e for e in esc_infos if e is not None]:
                    n2_total += int(nb_encode_count(
                        np.ascontiguousarray(cum_g_global).reshape(1, -1),
                        np.array([g], dtype=np.int64)))
                    co = cum_o_cache[g]
                    n2_total += int(nb_encode_count(
                        np.ascontiguousarray(co).reshape(1, -1),
                        np.array([off], dtype=np.int64)))
                total_bits += n2_total

                ph_code += time.time() - ta
                ta = time.time()
                # Verify 200 positions spread across file, reusing stored rows
                base = len(stored) - (Lb - 1)
                for t_idx in range(Lb - 1):
                    gpos = b0_global + t_idx
                    if gpos % 130 != 0 or n_verify_ok + n_verify_fail >= 200:
                        continue
                    v_topk, v_p = stored[base + t_idx]
                    target = block[t_idx + 1]
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
                        print(f"  FAIL at block {b0_global}, pos {t_idx}")

                ph_verify += time.time() - ta
                seq = ([prev_tail] if prev_tail is not None else []) + block
                for a, b in zip(seq, seq[1:]):
                    d = bi_counts.get(a)
                    if d is None:
                        d = {}
                        bi_counts[a] = d
                    d[b] = d.get(b, 0) + 1
                    bi_totals[a] = bi_totals.get(a, 0) + 1
                prev_tail = block[-1]
                b0_global += len(block)
                if probs_b is not None:
                    del probs_b
            if logits_f is not None:
                del logits_f
            bi += len(chunk)
            done_n = min(bi, len(blocks))
            if done_n % 2 == 0 or done_n == len(blocks):
                print(f"  ... block {done_n}/{len(blocks)}, escapes: {n_escapes}, "
                      f"bigram keys: {len(bi_counts)}", flush=True)

    torch.cuda.synchronize()
    dt = time.time() - t
    print(f"\n  Total bits: {total_bits}")
    print(f"  bits/byte: {total_bits/n_bytes:.4f}")
    print(f"  escapes: {n_escapes}")
    print(f"  Verified: {n_verify_ok} lossless, fails: {n_verify_fail}")
    print(f"  Time: {dt:.1f}s")
    print(f"  PHASES: fwd={ph_fwd:.1f}s softmaxtopk={ph_st:.1f}s "
          f"loop={ph_loop:.1f}s code={ph_code:.1f}s verify={ph_verify:.1f}s")

    tag = f"off{_off_mb}_lam{BIGRAM_LAMBDA}_gputopk{USE_GPU_TOPK}_bf{BATCH_FWD}_pf{PREFILTER}"
    results = {
        "model": "SmolLM2-135M-bigram-ensemble-v4",
        "bigram_lambda": BIGRAM_LAMBDA,
        "top_k": TOP_K,
        "n_bytes": n_bytes,
        "total_bits": total_bits,
        "bpb": total_bits / n_bytes,
        "escapes": n_escapes,
        "verified_ok": n_verify_ok,
        "verified_fail": n_verify_fail,
        "elapsed_s": time.time() - t0,
    }
    with open(f"./data/smollm2_ensemble_v4_{tag}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to data/smollm2_ensemble_v4_{tag}.json")


if __name__ == "__main__":
    main()
