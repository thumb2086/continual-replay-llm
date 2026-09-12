"""SmolLM2 overlap-KN ensemble (v7: floor-tuned 32-bit coder) on enwik8.

v7 = v6 + ac32 throughout:
1. Stage-1 via stage1_cum_32 with tunable FLOOR_FRAC. The 16-bit core's
   +1-count floor steals ~12.5% (K=2048) / ~6% (K=1024) of mass to smooth
   the head -- that tax, not raw precision, is what 32-bit TOTAL buys
   the range to tune (validated: smaller floor wins when tails are rare).
2. Stage-2 via cache-informed distribution over rest (single count),
   built from shared bigram/trigram tables both sides can reproduce --
   replaces uniform group+offset (two counts). USE_CACHE_S2=0 falls back
   to uniform_cum_32 rest (isolates the stage-2 gain).
Knobs: FLOOR_FRAC (stage-1 floor, default parity 6.1035e-5),
S2_FLOOR (stage-2 floor, default 2.2e-5 ~= uniform over 47K),
USE_CACHE_S2=0/1.
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
from ac32 import (
    uniform_cum_32, nb_encode_count_32, stage1_cum_32, cache_rest_cum_32,
    ArithmeticEncoder32, ArithmeticDecoder32, TOTAL32, nb_blend_row,
)

BIGRAM_LAMBDA = float(os.environ.get("BIGRAM_LAMBDA", "0.99"))
BIGRAM_CONF = float(os.environ.get("BIGRAM_CONF", "10.0"))
TRIGRAM_CONF = float(os.environ.get("TRIGRAM_CONF", "7.0"))
USE_TRIGRAM = int(os.environ.get("USE_TRIGRAM", "1"))
OVERLAP = int(os.environ.get("OVERLAP", "2048"))
assert 0 <= OVERLAP <= 4096 and OVERLAP < BLOCK_TOKENS
BLOCK_NEW = BLOCK_TOKENS - OVERLAP
FLOOR_FRAC = float(os.environ.get("FLOOR_FRAC", "6.1035e-5"))
S2_FLOOR = float(os.environ.get("S2_FLOOR", "1.5e-5"))
USE_CACHE_S2 = int(os.environ.get("USE_CACHE_S2", "1"))
USE_NUMBA_LOOP = int(os.environ.get("USE_NUMBA_LOOP", "1"))
USE_FP16_XFER = int(os.environ.get("USE_FP16_XFER", "0"))
PREFILTER = int(os.environ.get("PREFILTER", "8192"))
USE_GPU_TOPK = int(os.environ.get("USE_GPU_TOPK", "1"))
BATCH_FWD = int(os.environ.get("BATCH_FWD", "1"))
# (v8-vram: default 1. Batch-2 forward saves ~10% wall but pins 1.6GB
# logits; headroom beats speed until the loop is the bottleneck.)


def block_topk_gpu(probs_b, idx):
    """GPU topk for one block. probs_b: [B,V] fp32 CUDA. Returns
    (topk_vals_cpu [B,K], topk_idx_cpu [B,K]) for K=TOP_K, and
    (pre_vals, pre_idx) for K=PREFILTER. Values re-sorted desc on CPU."""
    with torch.no_grad():
        tvs, tis, pvs, pis = [], [], [], []
        # v8-vram: row-chunked topk (2048 rows at a time). Per-row op =>
        # identical results, ~4x smaller sort workspace. (Unchunked
        # topk(k=8192) over [8192,49K] spiked VRAM -> WDDM paging -> 12GB
        # host RSS + GPU 100% thrash. See memdiag notes.)
        for c0 in range(0, probs_b.shape[0], 2048):
            pc = probs_b[c0:c0 + 2048]
            _tv, _ti = torch.topk(pc, TOP_K, dim=1)
            _pv, _pi = torch.topk(pc, PREFILTER, dim=1)
            tvs.append(_tv)
            tis.append(_ti)
            pvs.append(_pv)
            pis.append(_pi)
        tv, ti = torch.cat(tvs), torch.cat(tis)
        pv, pi = torch.cat(pvs), torch.cat(pis)
    tv = tv.float().cpu().numpy()
    ti = ti.cpu().numpy().astype(np.int64)
    pv = pv.float().cpu().numpy()
    pi = pi.cpu().numpy().astype(np.int64)
    # torch.topk already returns values sorted descending; that order
    # matches v2's argsort(-p) whenever values are distinct (float ties:
    # measure-zero, absorbed as run noise ~1e-4). NO cpu re-sort:
    # re-sorting the 8192-wide prefilter cost ~40s/run in v4full3.
    return (tv, ti), (pv, pi)


def cache_posterior_dense(V, prev, prev2, bi_counts, bi_totals,
                          tri_counts, tri_totals):
    """Dense cache posterior over V from shared tables (KN weights).

    c = wT*T + (1-wT)*wB*B, unnormalized counts (zeros where unseen).
    Deterministic given tables: encoder and decoder build it identically.
    Returns None when neither row exists (caller falls back to uniform).
    """
    brow = bi_counts.get(prev)
    tkey = (prev2, prev) if (USE_TRIGRAM and prev2 is not None) else None
    trow = tri_counts.get(tkey) if tkey is not None else None
    if not brow and not trow:
        return None
    c = np.zeros(V, dtype=np.float64)
    if brow:
        btot = bi_totals[prev]
        wB = btot / (btot + BIGRAM_CONF)
    else:
        wB = 0.0
    if trow:
        ttot = tri_totals[tkey]
        wT = ttot / (ttot + TRIGRAM_CONF)
    else:
        wT = 0.0
    if brow:
        brk = np.fromiter(brow.keys(), dtype=np.int64, count=len(brow))
        brv = np.fromiter(brow.values(), dtype=np.float64, count=len(brow))
        c[brk] += (1.0 - wT) * wB * brv
    if trow:
        trk = np.fromiter(trow.keys(), dtype=np.int64, count=len(trow))
        trv = np.fromiter(trow.values(), dtype=np.float64, count=len(trow))
        c[trk] += wT * trv
    return c


def main():
    print("=" * 70)
    print("SMOLLM2 SPEED ENSEMBLE V8 ON ENWIK8")
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
    print(f"  AC32: floor_frac={FLOOR_FRAC:g}, cache_s2={USE_CACHE_S2}, s2_floor={S2_FLOOR:g}")

    print("\n[2/4] Loading enwik8 sample...")
    _off_mb = int(os.environ.get("ENWIK8_OFFSET_MB", "50"))
    with open("./data/cloud/enwik8", "rb") as f:
        f.seek(_off_mb * 1024 * 1024)
        raw = f.read(int(os.environ.get("ENWIK8_KB", "100")) * 1024)
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

    print("\n[3/4] Coding with overlap-KN blend (v6)...")
    total_bits, n_escapes = 0, 0
    n_verify_ok, n_verify_fail = 0, 0
    # v6: file token 0 coded once via uniform(V) with the 32-bit coder
    # (49K symbols can't fit 14-bit TOTAL) -- makes the stream
    # self-contained (v2-v5 left it + all block-boundary tokens uncoded).
    cum0 = uniform_cum_32(V)
    total_bits += int(nb_encode_count_32(np.ascontiguousarray(cum0).reshape(1, -1),
                                         np.array([ids[0]], dtype=np.int64)))
    _e0 = ArithmeticEncoder32(store=True)
    _e0.encode_symbol(cum0, ids[0])
    _bs0, _ = _e0.finish()
    if int(ArithmeticDecoder32(_bs0).decode_symbol(cum0)) == ids[0]:
        n_verify_ok += 1
    else:
        n_verify_fail += 1
        print("  FAIL at file token 0")
    all_ids = np.arange(V)
    bi_counts = {}
    bi_totals = {}
    tri_counts = {}  # (a,b) -> {c: n}
    tri_totals = {}  # (a,b) -> n
    prev_tail = None
    prev2_tail = None
    stored = []  # per coded position: (topk_idx, topk_p) for verify reuse
    # v8-memdiag: RSS census per segment (find the 12GB grower)
    import tracemalloc as _tm
    _tm.start()
    import ctypes as _ct
    import struct as _st

    def _rss_gb():
        try:
            pmc = _ct.create_string_buffer(72)
            h = _ct.windll.kernel32.GetCurrentProcess()
            if _ct.windll.psapi.GetProcessMemoryInfo(h, pmc, 72):
                ws = _st.unpack("P", pmc[8:8 + _st.calcsize("P")])[0]
                return ws / (1024 ** 3)
        except Exception:
            pass
        return -1.0
    # v8 numba blend scratch (allocated once, reused per position)
    _seen_pre = np.zeros(V, dtype=np.int32)
    _seen_row = np.zeros(V, dtype=np.int32)
    _work = np.zeros(V, dtype=np.float64)
    _heap_s = np.empty(TOP_K, dtype=np.float64)
    _heap_i = np.empty(TOP_K, dtype=np.int64)
    _out_idx = np.empty(TOP_K, dtype=np.int64)
    _out_p = np.empty(TOP_K, dtype=np.float64)
    _E64 = np.empty(0, dtype=np.int64)
    _E64f = np.empty(0, dtype=np.float64)
    _tagc = [0]

    torch.cuda.synchronize()
    t = time.time()
    ph_fwd = ph_st = ph_xfer = ph_loop = ph_code = ph_esc = ph_verify = 0.0
    # ---- v6 segments: (input ids, T0, NC, n_new) ----
    # seg0: inp = ids[0:BT], pairs t in [0, len-1)
    # segk: inp = tail(O) + new(NB), pairs t in [O-1, O-1+n)
    # (OV=0 degenerates to v5 blocks exactly.)
    segs = []
    _pos, _first = 0, True
    while _pos < len(ids):
        if _first:
            _new = ids[_pos:_pos + BLOCK_TOKENS]
            _inp = _new
            _T0, _NC = 0, len(_inp) - 1
            _pos += len(_new)
            _first = False
        else:
            _tail = ids[_pos - OVERLAP:_pos] if OVERLAP > 0 else []
            _new = ids[_pos:_pos + BLOCK_NEW]
            _inp = _tail + _new
            if _tail:
                _T0, _NC = len(_tail) - 1, len(_new)
            else:
                _T0, _NC = 0, len(_inp) - 1
            _pos += len(_new)
        segs.append((_inp, _T0, _NC, len(_new)))
    print(f"  Segments: {len(segs)}, overlap={OVERLAP}, new/block={BLOCK_NEW}")
    coded_count = 0  # coded-pair counter (verify spread)
    with torch.no_grad():
        si = 0
        while si < len(segs):
            chunk = segs[si:si + BATCH_FWD]
            if len({len(s[0]) for s in chunk}) > 1:
                chunk = chunk[:1]  # uneven tail input: run solo, no pad
            # forward (batched); drop degenerate segments (no pairs to code)
            fwd_idx = [k for k, s in enumerate(chunk) if s[2] >= 1 and len(s[0]) >= 2]
            if fwd_idx:
                _ta = time.time()
                x = torch.stack([torch.tensor(chunk[k][0], device=device)
                                 for k in fwd_idx])
                logits_f = model(x).logits  # [F, B, V] fp16 RAW (no softmax yet)
                torch.cuda.synchronize()
                ph_fwd += time.time() - _ta
            else:
                logits_f = None
            fpos = 0
            for k, (block, T0, NC, n_new) in enumerate(chunk):
                if NC < 1 or len(block) < 2:
                    continue
                lg_b = logits_f[fpos]  # [B, V] fp16 GPU view
                fpos += 1
                if USE_GPU_TOPK:
                    _ta = time.time()
                    with torch.no_grad():
                        probs_b = torch.softmax(lg_b.float(), dim=-1)
                    (tv, ti), (pv, pi) = block_topk_gpu(probs_b, k)
                    torch.cuda.synchronize()
                    ph_st += time.time() - _ta
                    _ta = time.time()
                    # ONE block H2D (fp32) instead of per-row syncs (v8 speed)
                    if USE_FP16_XFER:
                        # v8-mem: halve transfer (0.8GB). Rows upcast on
                        # use; tail probs < 6e-5 flush to 0 -- ratio TBD.
                        blk_probs = probs_b.half().cpu().numpy().astype(np.float32)
                    else:
                        blk_probs = probs_b.float().cpu().numpy()
                    torch.cuda.synchronize()
                    ph_xfer += time.time() - _ta
                    # v8-lean: drop the 1.6GB GPU copy immediately; the CPU
                    # copy serves all rows. (Was pinned through the loop.)
                    probs_b = None
                    probs_cpu = None
                else:
                    lg = lg_b.float().cpu().numpy()
                    lg = lg - lg.max(axis=1, keepdims=True)
                    e = np.exp(lg, dtype=np.float64)
                    probs_cpu = e / e.sum(axis=1, keepdims=True)
                    blk_probs = probs_cpu
                    probs_b = None

                s1_cums, s1_syms = [], []
                esc_infos = []
                sbase = len(stored)
                _ta = time.time()
                for t_idx in range(T0, T0 + NC):
                    prev = block[t_idx]
                    target = block[t_idx + 1]
                    prev2 = block[t_idx - 1] if t_idx > 0 else prev_tail
                    brow = bi_counts.get(prev)
                    tkey = (prev2, prev) if (USE_TRIGRAM and prev2 is not None) else None
                    trow = tri_counts.get(tkey) if tkey is not None else None
                    if brow or trow:
                        if brow:
                            btot = bi_totals[prev]
                            wB = btot / (btot + BIGRAM_CONF)
                            brk = np.fromiter(brow.keys(), dtype=np.int64, count=len(brow))
                            brv = np.fromiter(brow.values(), dtype=np.float64, count=len(brow))
                        else:
                            wB, brk, brv, btot = 0.0, None, None, 1
                        if trow:
                            ttot = tri_totals[tkey]
                            wT = ttot / (ttot + TRIGRAM_CONF)
                            trk = np.fromiter(trow.keys(), dtype=np.int64, count=len(trow))
                            trv = np.fromiter(trow.values(), dtype=np.float64, count=len(trow))
                        else:
                            wT, trk, trv, ttot = 0.0, None, None, 1
                        if USE_GPU_TOPK:
                            pre = pi[t_idx]
                            p_full = blk_probs[t_idx]
                        else:
                            p_full = blk_probs[t_idx]
                            pre = np.argpartition(p_full, -PREFILTER)[-PREFILTER:]
                        w_cache = 1.0 - (1.0 - wT) * (1.0 - wB)
                        lam_eff = 1.0 - (1.0 - BIGRAM_LAMBDA) * w_cache
                        if USE_NUMBA_LOOP:
                            sb = ((1.0 - wT) * wB / btot) if brk is not None else 0.0
                            st = (wT / ttot) if trk is not None else 0.0
                            _tagc[0] += 1
                            _n = nb_blend_row(
                                p_full, pre,
                                brk if brk is not None else _E64,
                                brv if brv is not None else _E64f,
                                trk if trk is not None else _E64,
                                trv if trv is not None else _E64f,
                                sb, st, lam_eff, TOP_K,
                                _seen_pre, _seen_row, _work, _tagc[0],
                                _heap_s, _heap_i, _out_idx, _out_p)
                            topk_idx, topk_p = _out_idx[:_n].copy(), _out_p[:_n].copy()
                        else:
                            cand = pre
                            if brk is not None:
                                cand = np.union1d(cand, brk)
                            if trk is not None:
                                cand = np.union1d(cand, trk)
                            pbv = np.zeros(len(cand))
                            if brk is not None:
                                bpos = np.searchsorted(cand, brk)
                                pbv[bpos] += (1.0 - wT) * wB * (brv / btot)
                            if trk is not None:
                                tpos = np.searchsorted(cand, trk)
                                pbv[tpos] += wT * (trv / ttot)
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
                    cum1 = stage1_cum_32(topk_p, esc_mass, FLOOR_FRAC)
                    # v8-lean: store ONLY verify candidates (was: all 30K
                    # positions ~= 500MB). Verify iterates stored[sbase:].
                    _gpos = coded_count + (t_idx - T0)
                    if ((OVERLAP > 0 and t_idx == T0) or (_gpos % 130 == 0)) and len(stored) < 300:
                        stored.append((t_idx, topk_idx, topk_p))

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
                        esc_infos.append((rank, prev, prev2, topk_idx))

                ph_loop += time.time() - _ta
                _ta = time.time()
                C = np.ascontiguousarray(np.stack(s1_cums))
                S = np.array(s1_syms, dtype=np.int64)
                n1 = int(nb_encode_count_32(C, S))
                if n1 < 0:
                    raise ArithmeticError(f"stage-1 encoder error {n1}")
                total_bits += n1
                ph_code += time.time() - _ta
                _ta = time.time()
                # Escape stage-2: ONE 32-bit count over rest[] with a
                # cache-informed distribution both sides reproduce.
                n2_total = 0
                for (rank, eprev, eprev2, etopk) in [e for e in esc_infos if e is not None]:
                    emask = np.ones(V, dtype=bool)
                    emask[etopk] = False
                    erest = all_ids[emask]
                    if USE_CACHE_S2:
                        cpost = cache_posterior_dense(
                            V, eprev, eprev2, bi_counts, bi_totals,
                            tri_counts, tri_totals)
                        co = (uniform_cum_32(len(erest)) if cpost is None
                              else cache_rest_cum_32(cpost, erest, S2_FLOOR))
                    else:
                        co = uniform_cum_32(len(erest))
                    n2_total += int(nb_encode_count_32(
                        np.ascontiguousarray(co).reshape(1, -1),
                        np.array([rank], dtype=np.int64)))
                total_bits += n2_total
                ph_esc += time.time() - _ta
                _ta = time.time()

                # Verify sampled positions (v8-lean: stored[sbase:] holds
                # ONLY verify candidates as (t_idx, topk, probs) tuples).
                for (v_tidx, v_topk, v_p) in stored[sbase:]:
                    if n_verify_ok + n_verify_fail >= 220:
                        break
                    target = block[v_tidx + 1]
                    v_esc = max(1e-12, 1.0 - v_p.sum())
                    v_cum = stage1_cum_32(v_p, v_esc, FLOOR_FRAC)
                    hit = np.where(v_topk == target)[0]
                    if len(hit):
                        enc = ArithmeticEncoder32(store=True)
                        enc.encode_symbol(v_cum, int(hit[0]))
                        bs, _ = enc.finish()
                        dec = ArithmeticDecoder32(bs)
                        got = int(v_topk[dec.decode_symbol(v_cum)])
                    else:
                        enc = ArithmeticEncoder32(store=True)
                        enc.encode_symbol(v_cum, TOP_K)
                        bs, _ = enc.finish()
                        dec = ArithmeticDecoder32(bs)
                        assert dec.decode_symbol(v_cum) == TOP_K
                        mask = np.ones(V, dtype=bool)
                        mask[v_topk] = False
                        rest = all_ids[mask]
                        rank = int(np.where(rest == target)[0][0])
                        vprev = block[v_tidx]
                        vprev2 = block[v_tidx - 1] if v_tidx > 0 else prev_tail
                        if USE_CACHE_S2:
                            vcpost = cache_posterior_dense(
                                V, vprev, vprev2, bi_counts, bi_totals,
                                tri_counts, tri_totals)
                            vco = (uniform_cum_32(len(rest)) if vcpost is None
                                   else cache_rest_cum_32(vcpost, rest, S2_FLOOR))
                        else:
                            vco = uniform_cum_32(len(rest))
                        en2 = ArithmeticEncoder32(store=True)
                        en2.encode_symbol(vco, rank)
                        bs2, _ = en2.finish()
                        got = int(rest[ArithmeticDecoder32(bs2).decode_symbol(vco)])
                    if got == target:
                        n_verify_ok += 1
                    else:
                        n_verify_fail += 1
                        print(f"  FAIL at coded-pos {coded_count + (v_tidx - T0)}, pos {v_tidx}")

                new = block[len(block) - n_new:]
                ph_verify += time.time() - _ta
                seq = ([prev_tail] if prev_tail is not None else []) + new
                for a, b in zip(seq, seq[1:]):
                    d = bi_counts.get(a)
                    if d is None:
                        d = {}
                        bi_counts[a] = d
                    d[b] = d.get(b, 0) + 1
                    bi_totals[a] = bi_totals.get(a, 0) + 1
                if USE_TRIGRAM:
                    tseq = ([prev2_tail] if prev2_tail is not None else []) + seq
                    for a, b, c in zip(tseq, tseq[1:], tseq[2:]):
                        key = (a, b)
                        d = tri_counts.get(key)
                        if d is None:
                            d = {}
                            tri_counts[key] = d
                        d[c] = d.get(c, 0) + 1
                        tri_totals[key] = tri_totals.get(key, 0) + 1
                    if len(seq) >= 2:
                        prev2_tail = seq[-2]
                prev_tail = new[-1]
                coded_count += NC
                # v8-lean: release segment scratch (blk 1.6GB, topk ~0.8GB)
                del blk_probs
                if USE_GPU_TOPK:
                    del tv, ti, pv, pi
                if probs_b is not None:
                    del probs_b
                torch.cuda.empty_cache()
            if logits_f is not None:
                del logits_f
                torch.cuda.empty_cache()
            si += len(chunk)
            done_n = min(si, len(segs))
            _snap = _tm.take_snapshot()
            _top = _snap.statistics("filename")[:3]
            _top_s = "; ".join(f"{s.traceback[0].filename.split(chr(92))[-1]}:{s.traceback[0].lineno} {s.size/1e6:.0f}MB" for s in _top)
            print(f"  ... seg {done_n}/{len(segs)}, escapes: {n_escapes}, "
                  f"bigram keys: {len(bi_counts)}, coded: {coded_count}, "
                  f"stored={len(stored)} trikeys={len(tri_counts)} | {_top_s}", flush=True)

    torch.cuda.synchronize()
    dt = time.time() - t
    print(f"\n  Total bits: {total_bits}")
    print(f"  bits/byte: {total_bits/n_bytes:.4f}")
    print(f"  escapes: {n_escapes}")
    print(f"  Verified: {n_verify_ok} lossless, fails: {n_verify_fail}")
    print(f"  Time: {dt:.1f}s")
    print(f"  PHASES: fwd={ph_fwd:.1f}s topk={ph_st:.1f}s xfer={ph_xfer:.1f}s "
          f"loop={ph_loop:.1f}s code={ph_code:.1f}s esc={ph_esc:.1f}s verify={ph_verify:.1f}s")
    print(f"  PEAK_VRAM={torch.cuda.max_memory_allocated()/1024**3:.2f}GB "
          f"(bf={BATCH_FWD} ov={OVERLAP})")

    _kb = int(os.environ.get("ENWIK8_KB", "100"))
    tag = (f"off{_off_mb}_lam{BIGRAM_LAMBDA}_gputopk{USE_GPU_TOPK}_bf{BATCH_FWD}"
           f"_pf{PREFILTER}_tri{USE_TRIGRAM}_bc{BIGRAM_CONF}_tc{TRIGRAM_CONF}_k{TOP_K}_ov{OVERLAP}"
           f"_f{FLOOR_FRAC:g}_s2{USE_CACHE_S2}_nb{USE_NUMBA_LOOP}_kb{_kb}")
    results = {
        "model": "SmolLM2-135M-speed-ensemble-v8",
        "floor_frac": FLOOR_FRAC,
        "use_cache_s2": USE_CACHE_S2,
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
    with open(f"./data/smollm2_ensemble_v8_{tag}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to data/smollm2_ensemble_v8_{tag}.json")


if __name__ == "__main__":
    main()
