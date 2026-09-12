"""SmolLM2 overlap-KN ensemble (v11: threaded loop) on enwik8.

v11 = v10 + two-phase loop (parity gate: 0.9213 exact):
Phase A (4 threads, pure): per-position blend/topk/rank. Kernels are
nogil; tables/blk_probs read-only; per-thread scratch. Phase B
(serial): esc_infos/stored/n_escapes bookkeeping in order, so the
bitstream is identical to the serial loop by construction.

v10 = v9 + three ratio-neutral speedups (parity gate: 0.9213 exact):
1. PREFILTER default 8192 -> 2048 (ledger-proven identical bpb).
2. Sliced lm_head: trunk-only forward, lm_head+softmax+topk per
   2048-row hidden slice, so full [B,V] logits never materialize
   (~0.8GB/frag saved). Slice grid matches block_topk_gpu's internal
   chunking -> bitwise identical tv/ti (same rows, same kernels).
3. BATCH_FWD default 1 -> 2 (spends the saved memory; math-neutral).

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
from concurrent.futures import ThreadPoolExecutor

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
    nb_find_rank, nb_batch_cum_32,
)

BIGRAM_LAMBDA = float(os.environ.get("BIGRAM_LAMBDA", "0.99"))
BIGRAM_CONF = float(os.environ.get("BIGRAM_CONF", "10.0"))
TRIGRAM_CONF = float(os.environ.get("TRIGRAM_CONF", "7.0"))
USE_TRIGRAM = int(os.environ.get("USE_TRIGRAM", "1"))
OVERLAP = int(os.environ.get("OVERLAP", "2048"))
assert 0 <= OVERLAP < BLOCK_TOKENS  # (v11-sweep: lifted 4096 cap; stride=BT-OV, cost ~BT/stride x forward)
BLOCK_NEW = BLOCK_TOKENS - OVERLAP
FLOOR_FRAC = float(os.environ.get("FLOOR_FRAC", "6.1035e-5"))
S2_FLOOR = float(os.environ.get("S2_FLOOR", "1.5e-5"))
USE_CACHE_S2 = int(os.environ.get("USE_CACHE_S2", "1"))
USE_NUMBA_LOOP = int(os.environ.get("USE_NUMBA_LOOP", "1"))
USE_FP16_XFER = int(os.environ.get("USE_FP16_XFER", "0"))
USE_CUDA_GRAPH = int(os.environ.get("USE_CUDA_GRAPH", "0"))
PREFILTER = int(os.environ.get("PREFILTER", "2048"))
USE_GPU_TOPK = int(os.environ.get("USE_GPU_TOPK", "1"))
BATCH_FWD = int(os.environ.get("BATCH_FWD", "1"))
N_LOOP_WORKERS = int(os.environ.get("N_LOOP_WORKERS", "8"))
AFF_BASE = int(os.environ.get("AFF_BASE", "24"))  # first dedicated E-core
# (v10-validated: bf=2 still pages at 11.22GB -- trunk activations, not
# logits, dominate. Slicing only removed the logits tip. Default stays 1;
# the win is peak 6.62 -> ~5.8GB headroom, not batch.)


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


_DIRTY_BI = set()   # v11-incr (ported from v13): rows touched by last tables-update
_DIRTY_TRI = set()


def _freeze_update(counts, totals, id_of, keys, vals, tots, dirty):
    """Incremental freeze: only new + dirty rows rebuilt (same values as
    full rebuild; untouched rows can't gain keys or counts). O(dirty)."""
    for a in dirty:
        row = counts.get(a)
        if row is None:
            continue
        i = id_of.get(a)
        if i is None:
            i = len(keys)
            id_of[a] = i
            keys.append(None)
            vals.append(None)
            tots.append(0.0)
        if keys[i] is None or len(keys[i]) != len(row):
            keys[i] = np.ascontiguousarray(
                np.fromiter(row.keys(), dtype=np.int64, count=len(row)))
        vals[i] = np.ascontiguousarray(
            np.fromiter(row.values(), dtype=np.float64, count=len(row)))
        tots[i] = totals[a]
    return id_of, keys, vals, np.asarray(tots, dtype=np.float64)


def _freeze_rows(counts, totals):
    """Freeze {key: {tok: n}} dicts into flat arrays, once per segment.

    Rows change only at segment end, so per-position fromiter (30K x
    small-alloc) becomes one O(total-pairs) pass. Returns
    (id_of, keys[], vals[], tots[]) with list-indexed rows.
    """
    id_of, keys, vals, tots = {}, [], [], []
    for a, row in counts.items():
        id_of[a] = len(keys)
        keys.append(np.ascontiguousarray(
            np.fromiter(row.keys(), dtype=np.int64, count=len(row))))
        vals.append(np.ascontiguousarray(
            np.fromiter(row.values(), dtype=np.float64, count=len(row))))
        tots.append(totals[a])
    return id_of, keys, vals, np.array(tots, dtype=np.float64)


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
    print("SMOLLM2 PRACTICAL ENSEMBLE V11 ON ENWIK8")
    print("=" * 70)
    device = torch.device("cuda")
    t0 = time.time()
    try:
        _hi_stream = torch.cuda.Stream(priority=-5)
        print("  [stream] high-priority CUDA stream (-5): forward queue-jumps")
    except Exception as _e:
        print(f"  [stream] default stream ({_e})")
        _hi_stream = None
    try:
        import ctypes as _ct2
        _ct2.windll.kernel32.SetPriorityClass(
            _ct2.windll.kernel32.GetCurrentProcess(), 0x80)  # HIGH_PRIORITY_CLASS
        print("  [prio] HIGH_PRIORITY_CLASS set (our threads first, video still plays)")
    except Exception as _e:
        print(f"  [prio] skip ({_e})")

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
    # v11-fast: uniform stage-2 rest cum is constant (V-TOP_K symbols);
    # share one read-only copy across escapes (nb_encode_count_32 never
    # mutates cums -- verified by inspection, parity gate watches).
    uni_rest_cum = np.ascontiguousarray(uniform_cum_32(V - TOP_K))
    bi_counts = {}
    bi_totals = {}
    tri_counts = {}  # (a,b) -> {c: n}
    tri_totals = {}  # (a,b) -> n
    _fz_bi = [{}, [], [], []]  # v11-incr: persistent frozen bigram state
    _fz_tri = [{}, [], [], []]  # v11-incr: persistent frozen trigram state
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
    # v11 threaded loop: 4 scratch sets (seen_pre/seen_row/work,
    # heap_s/heap_i/out_idx/out_p/rank_out/tagc), one pool for the run,
    # one reusable topk-idx buffer (rows written sparsely in Phase A).
    # v11-thread pinning: each pool worker locks to a dedicated E-core
    # (13900K cpus 16-31 are E-cores; default 24-31 for 8 workers).
    # Best-effort: failure leaves default scheduling (still correct).
    import threading as _th
    _aff_next = [0]
    _aff_lock = _th.Lock()

    def _pin_worker():
        with _aff_lock:
            _i = _aff_next[0]
            _aff_next[0] += 1
        try:
            import ctypes as _c3
            _c3.windll.kernel32.SetThreadAffinityMask(
                _c3.windll.kernel32.GetCurrentThread(),
                1 << (AFF_BASE + (_i % N_LOOP_WORKERS)))
        except Exception:
            pass

    _thr_scratch = []
    for _w in range(N_LOOP_WORKERS):
        _thr_scratch.append([
            np.zeros(V, dtype=np.int32), np.zeros(V, dtype=np.int32),
            np.zeros(V, dtype=np.float64), np.empty(TOP_K, dtype=np.float64),
            np.empty(TOP_K, dtype=np.int64), np.empty(TOP_K, dtype=np.int64),
            np.empty(TOP_K, dtype=np.float64), np.zeros(1, dtype=np.int64),
            [0],
        ])
    _pool = ThreadPoolExecutor(max_workers=N_LOOP_WORKERS,
                               initializer=_pin_worker)
    tidx_buf = np.empty((BLOCK_TOKENS, TOP_K), dtype=np.int64)
    _graphs = {}  # CUDA-graph cache: shape -> (graph, static_in, static_out)

    def _trunk_forward(x):
        """Trunk forward with CUDA-graph replay per input shape.

        Replay runs the identical kernels (bitwise identical output),
        with zero Python per-layer dispatch. Falls back to eager on
        any capture error. use_cache=False: hidden states are
        unaffected, KV work/memory skipped.
        """
        key = tuple(x.shape)
        g = _graphs.get(key, "missing")
        if g == "missing":
            try:
                static_in = torch.empty_like(x)
                static_in.copy_(x)
                for _ in range(3):  # warmup outside capture
                    _ = model.model(static_in, use_cache=False).last_hidden_state
                torch.cuda.synchronize()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    static_out = model.model(
                        static_in, use_cache=False).last_hidden_state
                _graphs[key] = (gr, static_in, static_out)
                return static_out.clone()
            except Exception as _e:
                print(f"  [cudagraph] capture failed {key}: {_e}; eager fallback")
                _graphs[key] = None
                return model.model(x, use_cache=False).last_hidden_state.clone()
        if g is None:
            return model.model(x, use_cache=False).last_hidden_state.clone()
        gr, static_in, static_out = g
        static_in.copy_(x)
        gr.replay()
        return static_out.clone()
    _E64 = np.empty(0, dtype=np.int64)
    _E64f = np.empty(0, dtype=np.float64)
    _n_plain = _n_blend = 0
    ph_loopA = ph_loopB = 0.0

    torch.cuda.synchronize()
    t = time.time()
    ph_fwd = ph_st = ph_xfer = ph_loop = ph_code = ph_esc = ph_verify = ph_frz = 0.0
    ph_fwd_gpu = 0.0  # CUDA-event pure-GPU forward (immune to CPU contention)
    _cpu = time.process_time  # CPU-time clock: time stolen by other apps doesn't count
    _FLOOR_INT = max(1, int(round(FLOOR_FRAC * TOTAL32)))
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
                _prev_stream = torch.cuda.current_stream()
                if _hi_stream is not None:
                    torch.cuda.set_stream(_hi_stream)
                _ev0 = torch.cuda.Event(enable_timing=True)
                _ev1 = torch.cuda.Event(enable_timing=True)
                _ev0.record()
                x = torch.stack([torch.tensor(chunk[k][0], device=device)
                                 for k in fwd_idx])
                # v12a-retest: graph replay gated by USE_CUDA_GRAPH, iff
                # set the per-chunk empty_cache calls are skipped (graph
                # pool safety). Parity gate: 0.9139 exact.
                if USE_CUDA_GRAPH:
                    hidden_all = _trunk_forward(x)
                else:
                    hidden_all = model.model(x, use_cache=False).last_hidden_state
                # v10: sliced lm_head + softmax + topk, per fragment.
                # Slice grid (0, 2048, ...) matches block_topk_gpu's
                # internal chunking, so tv/ti are bitwise identical to
                # the full-logits path (same rows, same kernels).
                frag_cache = []
                for _j in range(len(fwd_idx)):
                    _Hf = hidden_all[_j]
                    _B = _Hf.shape[0]
                    _tvs, _tis, _pvs, _pis, _bps, _lgs = [], [], [], [], [], []
                    for _s in range(0, _B, 2048):
                        _lg = model.lm_head(_Hf[_s:_s + 2048])
                        with torch.no_grad():
                            _pb = torch.softmax(_lg.float(), dim=-1)
                        if USE_GPU_TOPK:
                            (_tv, _ti), (_pv, _pi) = block_topk_gpu(_pb, 0)
                            _tvs.append(_tv)
                            _tis.append(_ti)
                            _pvs.append(_pv)
                            _pis.append(_pi)
                            if USE_FP16_XFER:
                                _bps.append(_pb.half().cpu().numpy().astype(np.float32))
                            else:
                                _bps.append(_pb.cpu().numpy())
                        else:
                            _lgs.append(_lg.float().cpu().numpy())
                        del _lg, _pb
                    if USE_GPU_TOPK:
                        _frag = (np.concatenate(_tvs), np.concatenate(_tis),
                                 np.concatenate(_pvs), np.concatenate(_pis),
                                 np.concatenate(_bps), None)
                    else:
                        _frag = (None, None, None, None, None,
                                 np.concatenate(_lgs))
                    frag_cache.append(_frag)
                    del _frag, _Hf
                del hidden_all, x
                if not USE_CUDA_GRAPH:
                    torch.cuda.empty_cache()
                _ev1.record()
                torch.cuda.synchronize()
                if _hi_stream is not None:
                    torch.cuda.set_stream(_prev_stream)
                ph_fwd += time.time() - _ta
                ph_fwd_gpu += _ev0.elapsed_time(_ev1) / 1000.0
            else:
                frag_cache = []
            fpos = 0
            for k, (block, T0, NC, n_new) in enumerate(chunk):
                if NC < 1 or len(block) < 2:
                    continue
                tv, ti, pv, pi, blk_probs, lg_full = frag_cache[fpos]
                fpos += 1
                probs_b = None
                if not USE_GPU_TOPK:
                    lg = lg_full - lg_full.max(axis=1, keepdims=True)
                    e = np.exp(lg, dtype=np.float64)
                    probs_cpu = e / e.sum(axis=1, keepdims=True)
                    blk_probs = probs_cpu
                    del lg, e

                s1_cums = None  # v9: cum built batched (no per-pos list)
                esc_infos = []
                sbase = len(stored)
                # ---- v9 freeze: tables -> flat arrays once per segment ----
                _ta = _cpu()
                bi_id_of, bi_keys, bi_vals, bi_tot = _freeze_update(
                    bi_counts, bi_totals,
                    _fz_bi[0], _fz_bi[1], _fz_bi[2], _fz_bi[3], _DIRTY_BI)
                if USE_TRIGRAM:
                    tri_id_of, tri_keys, tri_vals, tri_tot = _freeze_update(
                        tri_counts, tri_totals,
                        _fz_tri[0], _fz_tri[1], _fz_tri[2], _fz_tri[3], _DIRTY_TRI)
                else:
                    tri_id_of, tri_keys, tri_vals, tri_tot = {}, [], [], []
                ph_frz += _cpu() - _ta
                # ---- v9 phase-1: per-position blend+rank, batched cum ----
                topk_batch = np.empty((NC, TOP_K), dtype=np.float64)
                esc_vec = np.empty(NC, dtype=np.float64)
                sym_arr = np.empty(NC, dtype=np.int64)
                _ta = time.time()
                rank_arr = np.empty(NC, dtype=np.int64)
                path_arr = np.empty(NC, dtype=np.int8)
                tidx_batch = tidx_buf[:NC]
                # v11 Phase A: pure per-position work, 4 threads. Reads:
                # block, frozen tables, blk_probs, ti/tv/pi. Writes:
                # distinct rows of topk_batch/rank_arr/path_arr/tidx_batch.
                _T0 = T0
                _coded = coded_count
                _taA = _cpu()

                def _phaseA(_lo, _hi, _S):
                    _sp, _sr, _wk = _S[0], _S[1], _S[2]
                    _hs, _his, _oi, _op, _ro, _tg = (
                        _S[3], _S[4], _S[5], _S[6], _S[7], _S[8])
                    for _t in range(_lo, _hi):
                        _pos = _t - _T0
                        _prev = block[_t]
                        _tgt = block[_t + 1]
                        _prev2 = block[_t - 1] if _t > 0 else prev_tail
                        _ib = bi_id_of.get(_prev, -1)
                        if USE_TRIGRAM and _prev2 is not None:
                            _it = tri_id_of.get((_prev2, _prev), -1)
                        else:
                            _it = -1
                        _ps = ((OVERLAP > 0 and _t == _T0)
                               or ((_coded + (_t - _T0)) % 130 == 0))
                        if _ib < 0 and _it < 0:
                            path_arr[_pos] = 0
                            if USE_GPU_TOPK:
                                _tidx = ti[_t]
                                _tp = tv[_t].astype(np.float64)
                            else:
                                _pf = blk_probs[_t]
                                _tidx = np.argpartition(_pf, -TOP_K)[-TOP_K:]
                                _tidx = _tidx[np.argsort(-_pf[_tidx])]
                                _tp = _pf[_tidx]
                            _hit = np.where(_tidx == _tgt)[0]
                            _rk = int(_hit[0]) if len(_hit) else -1
                            topk_batch[_pos] = _tp
                            if _rk < 0 or _ps:
                                tidx_batch[_pos] = _tidx
                        else:
                            path_arr[_pos] = 1
                            if _ib >= 0:
                                _brk, _brv = bi_keys[_ib], bi_vals[_ib]
                                _bt = bi_tot[_ib]
                                _wB = _bt / (_bt + BIGRAM_CONF)
                            else:
                                _brk, _brv, _bt, _wB = _E64, _E64f, 1, 0.0
                            if _it >= 0:
                                _trk, _trv = tri_keys[_it], tri_vals[_it]
                                _tt = tri_tot[_it]
                                _wT = _tt / (_tt + TRIGRAM_CONF)
                            else:
                                _trk, _trv, _tt, _wT = _E64, _E64f, 1, 0.0
                            _pfull = blk_probs[_t]
                            if USE_GPU_TOPK:
                                _pre = pi[_t]
                            else:
                                _pre = np.argpartition(
                                    _pfull, -PREFILTER)[-PREFILTER:]
                            _wc = 1.0 - (1.0 - _wT) * (1.0 - _wB)
                            _le = 1.0 - (1.0 - BIGRAM_LAMBDA) * _wc
                            _sb = ((1.0 - _wT) * _wB / _bt) if _ib >= 0 else 0.0
                            _st = (_wT / _tt) if _it >= 0 else 0.0
                            _tg[0] += 1
                            _n = nb_blend_row(
                                _pfull, _pre, _brk, _brv, _trk, _trv,
                                _sb, _st, _le, TOP_K,
                                _sp, _sr, _wk, _tg[0],
                                _hs, _his, _oi, _op,
                                int(_tgt), _ro)
                            if _n < TOP_K:
                                raise ArithmeticError(
                                    f"candidate shortfall {_n} < {TOP_K}")
                            _rk = int(_ro[0])
                            topk_batch[_pos] = _op[:_n]
                            if _rk < 0 or _ps:
                                tidx_batch[_pos] = _oi[:_n]
                        rank_arr[_pos] = _rk

                _step = (NC + N_LOOP_WORKERS - 1) // N_LOOP_WORKERS
                _futs = []
                for _w in range(N_LOOP_WORKERS):
                    _lo = _T0 + min(_w * _step, NC)
                    _hi = _T0 + min((_w + 1) * _step, NC)
                    if _hi > _lo:
                        _futs.append(_pool.submit(
                            _phaseA, _lo, _hi, _thr_scratch[_w]))
                for _f in _futs:
                    _f.result()
                ph_loopA += _cpu() - _taA
                # v11 Phase B: serial bookkeeping in position order, so
                # esc_infos/stored/n_escapes are identical to serial.
                _taB = _cpu()
                for _pos in range(NC):
                    _t = _T0 + _pos
                    _rk = int(rank_arr[_pos])
                    _tp = topk_batch[_pos]
                    esc_vec[_pos] = max(1e-12, 1.0 - _tp.sum())
                    _sample = ((OVERLAP > 0 and _t == _T0)
                               or ((_coded + _pos) % 130 == 0)
                               ) and len(stored) < 300
                    if _rk >= 0:
                        sym_arr[_pos] = _rk
                        esc_infos.append(None)
                        if _sample:
                            stored.append((_t, tidx_batch[_pos].copy(), _tp))
                    else:
                        n_escapes += 1
                        sym_arr[_pos] = TOP_K
                        _tidx = tidx_batch[_pos]
                        # rank of target in V-minus-topk, sorted: exactly
                        # `target` ints in [0,target), minus those in topk.
                        # (Same integer the mask+where path computes; the
                        # verify block still uses mask+where as a cross-check.)
                        _tgt = block[_t + 1]
                        rank_rest = int(_tgt) - len(np.unique(_tidx[_tidx < _tgt]))
                        _pv = block[_t]
                        _pv2 = block[_t - 1] if _t > 0 else prev_tail
                        esc_infos.append((rank_rest, _pv, _pv2, _tidx))
                _n_plain += int((path_arr == 0).sum())
                _n_blend += int((path_arr == 1).sum())
                ph_loopB += _cpu() - _taB

                ph_loop += time.time() - _ta
                _ta = _cpu()
                # K topk + 1 escape = K+1 symbols -> K+2 cum cols
                C = np.empty((NC, TOP_K + 2), dtype=np.int64)
                nb_batch_cum_32(topk_batch, esc_vec, _FLOOR_INT, C)
                n1 = int(nb_encode_count_32(C, sym_arr))
                if n1 < 0:
                    raise ArithmeticError(f"stage-1 encoder error {n1}")
                total_bits += n1
                ph_code += _cpu() - _ta
                _ta = _cpu()
                # Escape stage-2: ONE 32-bit count over rest[] with a
                # cache-informed distribution both sides reproduce.
                n2_total = 0
                for (rank, eprev, eprev2, etopk) in [e for e in esc_infos if e is not None]:
                    if USE_CACHE_S2:
                        emask = np.ones(V, dtype=bool)
                        emask[etopk] = False
                        erest = all_ids[emask]
                        cpost = cache_posterior_dense(
                            V, eprev, eprev2, bi_counts, bi_totals,
                            tri_counts, tri_totals)
                        co = (uniform_cum_32(len(erest)) if cpost is None
                              else cache_rest_cum_32(cpost, erest, S2_FLOOR))
                    else:
                        # uniform over rest: shared precomputed cum, no
                        # mask/erest build (~1ms/escape saved, same bits).
                        co = (uni_rest_cum if len(etopk) == TOP_K
                              else uniform_cum_32(V - len(np.unique(etopk))))
                    n2_total += int(nb_encode_count_32(
                        np.ascontiguousarray(co).reshape(1, -1),
                        np.array([rank], dtype=np.int64)))
                total_bits += n2_total
                ph_esc += _cpu() - _ta
                _ta = _cpu()

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
                ph_verify += _cpu() - _ta
                seq = ([prev_tail] if prev_tail is not None else []) + new
                for a, b in zip(seq, seq[1:]):
                    d = bi_counts.get(a)
                    if d is None:
                        d = {}
                        bi_counts[a] = d
                    d[b] = d.get(b, 0) + 1
                    bi_totals[a] = bi_totals.get(a, 0) + 1
                    _DIRTY_BI.add(a)
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
                        _DIRTY_TRI.add(key)
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
                if not USE_CUDA_GRAPH:
                    torch.cuda.empty_cache()
            if frag_cache is not None:
                del frag_cache
                if not USE_CUDA_GRAPH:
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
          f"frz={ph_frz:.1f}s loop={ph_loop:.1f}s code={ph_code:.1f}s esc={ph_esc:.1f}s verify={ph_verify:.1f}s")
    print(f"  MIX: plain={_n_plain} blend={_n_blend} loopA_cpu={ph_loopA:.1f}s loopB_cpu={ph_loopB:.1f}s loop_wall={ph_loop:.1f}s")
    print(f"  CLOCKS: gpu_fwd={ph_fwd_gpu:.1f}s fwd_wall={ph_fwd:.1f}s "
          f"stolen_fwd~={ph_fwd - ph_fwd_gpu:.1f}s stolen_loop~={ph_loop - (ph_loopA / 4 + ph_loopB):.1f}s")
    _pool.shutdown()
    print(f"  PEAK_VRAM={torch.cuda.max_memory_allocated()/1024**3:.2f}GB "
          f"(bf={BATCH_FWD} ov={OVERLAP})")

    _kb = int(os.environ.get("ENWIK8_KB", "100"))
    tag = (f"off{_off_mb}_lam{BIGRAM_LAMBDA}_gputopk{USE_GPU_TOPK}_bf{BATCH_FWD}"
           f"_pf{PREFILTER}_tri{USE_TRIGRAM}_bc{BIGRAM_CONF}_tc{TRIGRAM_CONF}_k{TOP_K}_ov{OVERLAP}"
           f"_f{FLOOR_FRAC:g}_s2{USE_CACHE_S2}_nb{USE_NUMBA_LOOP}_kb{_kb}")
    results = {
        "model": "SmolLM2-135M-practical-ensemble-v11",
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
    with open(f"./data/smollm2_ensemble_v11_{tag}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to data/smollm2_ensemble_v11_{tag}.json")


if __name__ == "__main__":
    main()
