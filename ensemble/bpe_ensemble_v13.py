"""SmolLM2 overlap-KN ensemble (v13: KV-cache chaining) on enwik8.

v13 = v11 with full-window re-forward replaced by cache chaining: seg0
forwards its window; each later seg forwards ONLY its `new` tokens with
the carried (trailing-8192-trimmed) KV cache + absolute position_ids.
Forward tokens collapse ~65K -> ~33K per 100KB (ov4096). Fresh rows +
one carried row are assembled into full-block tv/ti/pv/pi/blk tables so
all downstream code (loop/coders) is untouched. Rotary now uses
absolute positions (recompute windows restarted at 0): bpb WILL move --
measure, don't assume. Gate: losslessness absolute.

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
import atexit
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

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
OVERLAP = int(os.environ.get("OVERLAP", "0"))
assert 0 <= OVERLAP < BLOCK_TOKENS  # (v11-sweep: lifted 4096 cap; stride=BT-OV, cost ~BT/stride x forward)
BLOCK_NEW = BLOCK_TOKENS - OVERLAP
FLOOR_FRAC = float(os.environ.get("FLOOR_FRAC", "1e-6"))
S2_FLOOR = float(os.environ.get("S2_FLOOR", "1.5e-5"))
USE_CACHE_S2 = int(os.environ.get("USE_CACHE_S2", "0"))
USE_NUMBA_LOOP = int(os.environ.get("USE_NUMBA_LOOP", "1"))
USE_FP16_XFER = int(os.environ.get("USE_FP16_XFER", "1"))
USE_CUDA_GRAPH = int(os.environ.get("USE_CUDA_GRAPH", "0"))
GRAPH_FULL = int(os.environ.get("GRAPH_FULL", "0"))  # v13-graph2: full-model static-shape replay (flash era retry; old zeros bug was math-era trunk-only). 0 = off.
PREFILTER = int(os.environ.get("PREFILTER", "2048"))
USE_GPU_TOPK = int(os.environ.get("USE_GPU_TOPK", "1"))
BATCH_FWD = int(os.environ.get("BATCH_FWD", "1"))
N_LOOP_WORKERS = int(os.environ.get("N_LOOP_WORKERS", "1"))
AFF_BASE = int(os.environ.get("AFF_BASE", "24"))  # first dedicated E-core
USE_PROC_LOOP = int(os.environ.get("USE_PROC_LOOP", "0"))  # v13-proc: Phase-A in worker procs (own GIL); default off
N_PROC = int(os.environ.get("N_PROC", "4"))
PIPELINE = int(os.environ.get("PIPELINE", "0"))  # v13-pipe: helper-thread finish overlaps next fwd; default off
EMPTY_EVERY = int(os.environ.get("EMPTY_EVERY", "0"))  # v13: empty_cache cadence (segs); measured null, default off (was every seg)
SKIP_EVERY = int(os.environ.get("SKIP_EVERY", "0"))  # v13-skip: skip LM forward every Nth seg (1st kept); 0 = off. Cache-only topk (lam=0), decoder mirrors via same seg counter; verify guards.
SKIP_MOD = int(os.environ.get("SKIP_MOD", "1"))  # which residue to skip: (si-1)%N==MOD. MOD=3 skips the 4th seg (warmest tables: best-case probe).
NUMBA_DRIVER = int(os.environ.get("NUMBA_DRIVER", "0"))  # v13-numdrv: Phase-A driver as njit in the worker (nogil compute, free main). Bit-identical mirror; verify + bpb-EXACT gate. Refused unless threads + gputopk + 1 worker + probe off.
_EXCLPROBE = int(os.environ.get("EXCLPROBE", "0"))  # v13-exclprobe: count exclusive-key target hits (pi-only gather safety)
_PIONLY = int(os.environ.get("PI_ONLY", "0"))  # v13-pionly: truncate brow/trow to pi-members (ratio-cost probe)
BLEND_BT_MIN = int(os.environ.get("BLEND_BT_MIN", "5"))  # v13 blend-gate: skip blend unless bigram row total >= this (5/2 proven EXACT: low-count blends are pure waste; 0 = classic existence gate)
BLEND_TT_MIN = int(os.environ.get("BLEND_TT_MIN", "2"))  # same for trigram row total
GATHER_PI = int(os.environ.get("GATHER_PI", "1"))  # v13-gather: kill full-V softmax+transfer (topk on logits + lse + gather pi-logits + CPU exp). Implies pi-only math; ratio gate vs PI_ONLY number.
_PIONLY_EFF = 1 if (_PIONLY or GATHER_PI) else 0  # run-constant: no cross-seg race
GC_OFF = int(os.environ.get("GC_OFF", "0"))  # v13-micro: gc.disable() during run (cyclic trash can't form here); default off
CUDNN_BM = int(os.environ.get("CUDNN_BM", "0"))  # v13-micro: cudnn.benchmark (no convs in Llama; expected null)
RESTART_EVERY = int(os.environ.get("RESTART_EVERY", "16384"))  # v13-restart: fresh full-window chain restart bound (0 = pure chain = KNOWN GARBAGE beyond 8K positions)
CHAIN = int(os.environ.get("CHAIN", "0"))  # v13: 1 = KV chaining (FALSIFIED: fp16-RoPE drift compounds to 2.64); 0 = full-window recompute (correct, v11 math)
MEMDIAG = int(os.environ.get("MEMDIAG", "0"))  # v13: tracemalloc census (costs ~0.3s/seg); default off
USE_ORT = int(os.environ.get("USE_ORT", "0"))  # v13-ort: ONNX Runtime CUDA backend (EXPERIMENTAL/UNRUN); default off
ORT_MODEL_DIR = os.environ.get("ORT_MODEL_DIR", "./ort-smollm2")
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
        # v13-unchunk: retry single-shot (flash-era workspace is smaller;
        # fewer launches = tighter burst = better boost residency).
        _UNCH = int(os.environ.get("UNCHUNKED_TOPK", "1"))
        _step = probs_b.shape[0] if _UNCH else 2048
        for c0 in range(0, probs_b.shape[0], _step):
            pc = probs_b[c0:c0 + _step]
            _pv, _pi = torch.topk(pc, PREFILTER, dim=1)
            # v13-1topk: ti is the exact global top-TOP_K (subset of pi,
            # PREFILTER > TOP_K) -- one full-V topk pass killed, zero
            # numerical change (same values, same set).
            _tv2, _tpos = torch.topk(_pv, TOP_K, dim=1)
            _ti = torch.gather(_pi, 1, _tpos)
            tvs.append(_tv2)
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


def _past_len(past):
    """Sequence length of a KV cache (new Cache API, legacy attrs, tuple)."""
    try:
        return int(past.get_seq_length())
    except Exception:
        pass
    try:
        return past.key_cache[0].shape[2]
    except Exception:
        return past[0][0].shape[2]


_ORT_SESS = None
_ORT_IO = None    # (in_names, out_names, past_in_names, present_out_names, logits_idx)
_ORT_PAST = {}    # past-input name -> np array (numpy feeds; IO-binding is a follow-up)


def _ort_init_session(n_kv, hdim):
    """Create the CUDA ORT session once (USE_ORT=1 only). Loud errors:
    run ort_export.py first, install optimum[onnxruntime] + onnxruntime-gpu."""
    global _ORT_SESS, _ORT_IO
    import onnxruntime as _ort
    _so = _ort.SessionOptions()
    _so.graph_optimization_level = _ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _so.log_severity_level = 3
    _ORT_SESS = _ort.InferenceSession(
        os.path.join(ORT_MODEL_DIR, "model.onnx"),
        sess_options=_so, providers=["CUDAExecutionProvider"])
    _ins = [i.name for i in _ORT_SESS.get_inputs()]
    _outs = [o.name for o in _ORT_SESS.get_outputs()]
    _past_ins = [n for n in _ins if "past" in n or "key_value" in n]
    _pres_outs = [n for n in _outs if "present" in n]
    _logit_idx = next(i for i, n in enumerate(_outs) if "logit" in n)
    _ORT_IO = (_ins, _outs, _past_ins, _pres_outs, _logit_idx, n_kv, hdim)
    print(f"  [ort] session ready ({len(_past_ins)} past inputs)")


def _ort_forward(xin, abs_pos, device):
    """One cached forward via ORT CUDA. Returns fp16-GPU [n, V] logits.

    Past/present inputs are zipped BY ORDER (optimum export keeps them
    aligned); a name mismatch shows up as bpb drift, caught by the gate.
    """
    import numpy as _np
    _ids = xin[0].detach().cpu().numpy().astype(_np.int64)
    _n = _ids.shape[0]
    _ins, _outs, _past_ins, _pres_outs, _logit_idx, _n_kv, _hdim = _ORT_IO
    _feeds = {}
    for _nm in _ins:
        if "input_ids" in _nm or _nm == _ins[0]:
            _feeds[_nm] = _ids.reshape(1, -1)
        elif "position_ids" in _nm:
            _feeds[_nm] = _np.arange(abs_pos, abs_pos + _n).reshape(1, -1)
        elif "attention_mask" in _nm:
            _feeds[_nm] = _np.ones((1, abs_pos + _n), dtype=_np.int64)
        elif _nm in _past_ins:
            _feeds[_nm] = _ORT_PAST.get(
                _nm, _np.zeros((1, _n_kv, 0, _hdim), dtype=_np.float32))
    _vals = _ORT_SESS.run(None, _feeds)
    _by_name = dict(zip(_outs, _vals))
    for _pn, _pr in zip(_past_ins, _pres_outs):
        _arr = _by_name[_pr]
        if _arr.shape[2] > BLOCK_TOKENS:
            _arr = _arr[:, :, -BLOCK_TOKENS:, :]
        _ORT_PAST[_pn] = _arr
    _lg = _by_name[_outs[_logit_idx]][0].astype(_np.float16)
    return torch.from_numpy(_lg).to(device)


def _trim_past(past, keep):
    """Keep trailing `keep` positions (absolute positions stay valid).

    New API: crop() drops oldest in place. Legacy: contiguous copies
    (views would pin the whole original allocation).
    """
    try:
        _n = int(past.get_seq_length())
        if _n > keep:
            past.crop(_n - keep)
        return past
    except Exception:
        pass
    try:
        kc, vc = past.key_cache, past.value_cache
        for i in range(len(kc)):
            kc[i] = kc[i][..., -keep:, :].contiguous()
            vc[i] = vc[i][..., -keep:, :].contiguous()
        return past
    except Exception:
        pass
    return tuple(
        (k[..., -keep:, :].contiguous(), v[..., -keep:, :].contiguous())
        for k, v in past)


_SHM_MAIN = {}
_PROC_CTX = {}

_E64 = np.empty(0, dtype=np.int64)
_E64f = np.empty(0, dtype=np.float64)


def _shm_alloc(name, shape, dtype):
    """Create (or recreate) a named shared buffer. Stale segments left
    by killed runs are unlinked first so names never collide."""
    from multiprocessing import shared_memory
    try:
        _stale = shared_memory.SharedMemory(name=name)
        _stale.close()
        _stale.unlink()
    except Exception:
        pass
    shm = shared_memory.SharedMemory(
        name=name, create=True,
        size=int(np.prod(shape)) * np.dtype(dtype).itemsize)
    return shm, np.ndarray(tuple(shape), dtype=dtype, buffer=shm.buf)


def _shm_cleanup():
    for _shm, _view in _SHM_MAIN.values():
        try:
            _shm.close()
            _shm.unlink()
        except Exception:
            pass
    _SHM_MAIN.clear()


def _range_core(block, T0, lo, hi, coded, prev_tail,
                bi_id_of, tri_id_of, bi_keys, bi_vals, bi_tot,
                tri_keys, tri_vals, tri_tot,
                blk, tv, ti, pi, obatch, orank, opath, otidx, S,
                nolm=False):
    """Single-source Phase-A compute for positions [lo, hi).

    Pure: reads block/tables/blk/tv/ti/pi, writes distinct rows of
    obatch/orank/opath/otidx. Thread pool calls it with in-process
    arrays; proc pool calls it with shared-memory views -- one math
    source, the parity gate guards both.
    nolm (v13-skip): cache-only mode, no LM forward ran. Forces the
    blend branch with lam=0 (p_row must be zeros); short rows are
    padded (escape absorbs) instead of raising. Deterministic, so the
    decoder mirrors exactly; verify guards.
    """
    _sp, _sr, _wk = S[0], S[1], S[2]
    _hs, _his, _oi, _op, _ro, _tg = S[3], S[4], S[5], S[6], S[7], S[8]
    for _t in range(lo, hi):
        _pos = _t - T0
        _prev = block[_t]
        _tgt = block[_t + 1]
        _prev2 = block[_t - 1] if _t > 0 else prev_tail
        _ib = bi_id_of.get(_prev, -1)
        if USE_TRIGRAM and _prev2 is not None:
            _it = tri_id_of.get((_prev2, _prev), -1)
        else:
            _it = -1
        _ps = ((OVERLAP > 0 and _t == T0)
               or ((coded + (_t - T0)) % 130 == 0))
        # v13 blend-gate: existence gate (classic) PLUS count gate: rows
        # below the minima fall back to plain (LM-only). Low-count blends
        # carry ~no cache weight, but cost a full 72us kernel call.
        # Deterministic on causal tables -> decoder mirrors; verify guards.
        _gate_bt = bi_tot[_ib] if _ib >= 0 else 0
        _gate_tt = tri_tot[_it] if _it >= 0 else 0
        if ((_ib < 0 and _it < 0)
                or ((BLEND_BT_MIN > 0 or BLEND_TT_MIN > 0)
                    and _gate_bt < BLEND_BT_MIN and _gate_tt < BLEND_TT_MIN)
                ) and not nolm:
            opath[_pos] = 0
            if USE_GPU_TOPK:
                _tidx = ti[_t]
                _tp = tv[_t].astype(np.float64)
            else:
                _pf = blk[_t]
                _tidx = np.argpartition(_pf, -TOP_K)[-TOP_K:]
                _tidx = _tidx[np.argsort(-_pf[_tidx])]
                _tp = _pf[_tidx]
            _hit = np.where(_tidx == _tgt)[0]
            _rk = int(_hit[0]) if len(_hit) else -1
            obatch[_pos] = _tp
            if _rk < 0 or _ps:
                otidx[_pos] = _tidx
        else:
            opath[_pos] = 1
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
            _pfull = blk[_t]
            if USE_GPU_TOPK:
                _pre = pi[_t]
            else:
                _pre = np.argpartition(_pfull, -PREFILTER)[-PREFILTER:]
            _wc = 1.0 - (1.0 - _wT) * (1.0 - _wB)
            _le = 1.0 - (1.0 - BIGRAM_LAMBDA) * _wc
            if nolm:
                _le = 0.0  # cache-only: LM term exactly zero
            # v13-gather: numba pi-truncation (V-stamp, exact: pi-subset
            # always fits PREFILTER temps). Replaces the np.isin probe
            # path (~3s/run). _sp doubles as stamp (fresh tag from the
            # shared counter; the kernel re-stamps with its own tag after).
            if _PIONLY_EFF:
                _ctag = _tg[0] + 1
                _tg[0] = _ctag
                _ok, _ov, _otk, _otv = S[9], S[10], S[11], S[12]
                _cnb, _cnt = _compact_pi(
                    _brk if _ib >= 0 else _E64, _brv if _ib >= 0 else _E64f,
                    _trk if _it >= 0 else _E64, _trv if _it >= 0 else _E64f,
                    _pre, _sp, _ctag, _ok, _ov, _otk, _otv)
                _brk, _brv = _ok[:_cnb], _ov[:_cnb]
                _trk, _trv = _otk[:_cnt], _otv[:_cnt]
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
                if nolm:
                    # short cache rows: zero-pad (escape absorbs the mass).
                    # Pad ids with 1<<62 (never < tgt, never a real id, so
                    # the escape rank cross-check stays exact on both sides).
                    _op[_n:TOP_K] = 0.0
                    _oi[_n:TOP_K] = (1 << 62)
                    if _n == 0:
                        _rk = -1  # no candidates: forced escape (orank
                        # scratch would otherwise leak the previous row)
                    _n = TOP_K
                else:
                    raise ArithmeticError(
                        f"candidate shortfall {_n} < {TOP_K}")
            else:
                _rk = int(_ro[0])
            obatch[_pos] = _op[:_n]
            if _rk < 0 or _ps:
                otidx[_pos] = _oi[:_n]
        orank[_pos] = _rk
        if _EXCLPROBE and opath[_pos] == 1:
            _hit_pre = bool(np.any(_pre == _tgt))
            with _EXCL_LOCK:
                _EXCL[1] += 1
                if _rk >= 0 and not _hit_pre:
                    _EXCL[0] += 1


def _proc_init(specs, _V, _K):
    """Proc-worker init: attach shared buffers (handles pinned in the
    worker global so views stay valid), build private scratch."""
    import numpy as np
    from multiprocessing import shared_memory
    for _name, _shape, _dtype in specs:
        _shm = shared_memory.SharedMemory(name=_name)
        _PROC_CTX[_name] = (
            _shm, np.ndarray(tuple(_shape), dtype=_dtype, buffer=_shm.buf))
    _PROC_CTX["scratch"] = [
        np.zeros(_V, dtype=np.int32), np.zeros(_V, dtype=np.int32),
        np.zeros(_V, dtype=np.float64), np.empty(_K, dtype=np.float64),
        np.empty(_K, dtype=np.int64), np.empty(_K, dtype=np.int64),
        np.empty(_K, dtype=np.float64), np.zeros(1, dtype=np.int64),
        [0],
        # (v13-gather pi-truncation temps; PREFILTER captured at spawn.
        # If the driver later changes PREFILTER per run, procs must
        # respawn -- gather is refused under PROC anyway.)
        np.empty(PREFILTER, dtype=np.int64), np.empty(PREFILTER, dtype=np.float64),
        np.empty(PREFILTER, dtype=np.int64), np.empty(PREFILTER, dtype=np.float64),
    ]


def _proc_range(block, T0, coded, prev_tail,
                bi_id_of, tri_id_of, bi_keys, bi_vals, bi_tot,
                tri_keys, tri_vals, tri_tot, lo, hi, parg, nolm=False):
    """Proc-side entry: resolve shm views (input slice pinned by parity),
    run the single-source core. Outputs land in shared memory; the
    future itself carries nothing."""
    _g = _PROC_CTX
    _range_core(block, T0, lo, hi, coded, prev_tail,
                bi_id_of, tri_id_of, bi_keys, bi_vals, bi_tot,
                tri_keys, tri_vals, tri_tot,
                _g["blk"][1][parg], _g["tv"][1][parg],
                _g["ti"][1][parg], _g["pi"][1][parg],
                _g["obt"][1], _g["ork"][1], _g["opa"][1], _g["oti"][1],
                _g["scratch"], nolm)
    return True


_DIRTY_BI = set()   # v13-incr: rows touched by the last tables-update
_DIRTY_TRI = set()

# v13-exclprobe: EXCLPROBE=1 counts blend targets found ONLY outside
# prefilter pi (via brow/trow-exclusive keys). Decides pi-only gather.
import threading as _th
_EXCL_LOCK = _th.Lock()
_EXCL = [0, 0]  # [exclusive_target_hits, blend_positions]

try:
    import numba as _nb

    @_nb.njit
    def _compact_pi(brk, brv, trk, trv, pi, stamp, tag, ok, ov, otk, otv):
        """Truncate brow/trow rows to pi-members (V-stamp, no hash).
        ok/ov/otk/otv sized >= PREFILTER (pi-subset always fits).
        Returns (nb, nt). Stamp tag must be fresh vs stale stamps."""
        for _j in range(pi.shape[0]):
            stamp[pi[_j]] = tag
        _nb2 = 0
        _cap = ok.shape[0]
        for _j in range(brk.shape[0]):
            if stamp[brk[_j]] == tag:
                ok[_nb2] = brk[_j]
                ov[_nb2] = brv[_j]
                _nb2 += 1
                if _nb2 >= _cap:
                    break
        _nt2 = 0
        for _j in range(trk.shape[0]):
            if stamp[trk[_j]] == tag:
                otk[_nt2] = trk[_j]
                otv[_nt2] = trv[_j]
                _nt2 += 1
                if _nt2 >= _cap:
                    break
        return _nb2, _nt2
except Exception:
    _compact_pi = None  # numba missing: fall back to np.isin path


try:
    import numba as _nb2

    @_nb2.njit(cache=True)
    def _range_numba(block, T0, lo, hi, coded, pt_i, has_pt,
                     bi_idarr, bi_rk, bi_rv, bi_tots,
                     tri_sk, tri_si, tri_rk, tri_rv, tri_tots,
                     blk, tv, ti, pi, obatch, orank, opath, otidx,
                     sp, sr, wk, hs, his, oi, op, ro, tagbox,
                     ok, ov, otk, otv, e64, e64f, errbox,
                     f_tri, f_ov0, f_cB, f_cT, f_lam, f_K, f_P,
                     f_eff, f_nolm):
        """njit mirror of _range_core (v13-numdrv).

        Same reads, same branches, same kernel calls in the same order,
        so results are bit-identical (verify + bpb-EXACT gate). Runs
        nogil in a pool worker: main launches freely, worker grinds.
        Differences vs the Python twin, all unreachable-or-identical:
        shortfall pad path initializes _rk=-1 (Python would reuse a
        stale one -- reachable only when pre<TOP_K, impossible since
        prefilter>=K by the 1topk exactness requirement); EXCLPROBE and
        CPU-argpartition paths are refused at the wrapper (fall back).
        tri lookup via sorted composite keys + binary search
        (key=(p2<<16)|p1; ids < 2**16 guaranteed by wrapper assert).
        """
        for _t in range(lo, hi):
            _pos = _t - T0
            _prev = block[_t]
            _tgt = block[_t + 1]
            _p2ok = (_t > 0) or has_pt
            _prev2 = block[_t - 1] if _t > 0 else pt_i
            if 0 <= _prev < bi_idarr.shape[0]:
                _ib = bi_idarr[_prev]
            else:
                _ib = -1
            _it = -1
            if f_tri and _p2ok:
                _key = (_prev2 << 16) | _prev
                _a2 = 0
                _b2 = tri_sk.shape[0]
                while _a2 < _b2:
                    _m2 = (_a2 + _b2) // 2
                    if tri_sk[_m2] < _key:
                        _a2 = _m2 + 1
                    else:
                        _b2 = _m2
                if _a2 < tri_sk.shape[0] and tri_sk[_a2] == _key:
                    _it = tri_si[_a2]
            _rk = -1
            _ps = (f_ov0 and _t == T0) or ((coded + (_t - T0)) % 130 == 0)
            if ((_ib < 0 and _it < 0) and not f_nolm):
                opath[_pos] = 0
                _hit0 = -1
                for _j in range(f_K):
                    if ti[_t, _j] == _tgt:
                        _hit0 = _j
                        break
                _rk = _hit0
                for _j in range(f_K):
                    hs[_j] = np.float64(tv[_t, _j])
                    otidx[_pos, _j] = ti[_t, _j] if (_rk < 0 or _ps) else otidx[_pos, _j]
                for _j in range(f_K):
                    obatch[_pos, _j] = hs[_j]
            else:
                opath[_pos] = 1
                if _ib >= 0:
                    _brk = bi_rk[_ib]
                    _brv = bi_rv[_ib]
                    _bt = bi_tots[_ib]
                    _wB = _bt / (_bt + f_cB)
                else:
                    _brk = e64
                    _brv = e64f
                    _bt = 1.0
                    _wB = 0.0
                if _it >= 0:
                    _trk = tri_rk[_it]
                    _trv = tri_rv[_it]
                    _tt = tri_tots[_it]
                    _wT = _tt / (_tt + f_cT)
                else:
                    _trk = e64
                    _trv = e64f
                    _tt = 1.0
                    _wT = 0.0
                _pfull = blk[_t]
                _pre = pi[_t]
                _wc = 1.0 - (1.0 - _wT) * (1.0 - _wB)
                _le = 1.0 - (1.0 - f_lam) * _wc
                if f_nolm:
                    _le = 0.0
                if f_eff:
                    _ctag = tagbox[0] + 1
                    tagbox[0] = _ctag
                    _cnb, _cnt = _compact_pi(
                        _brk, _brv, _trk, _trv,
                        _pre, sp, _ctag, ok, ov, otk, otv)
                    _brk = ok[:_cnb]
                    _brv = ov[:_cnb]
                    _trk = otk[:_cnt]
                    _trv = otv[:_cnt]
                if _ib >= 0:
                    _sb = ((1.0 - _wT) * _wB / _bt)
                else:
                    _sb = 0.0
                if _it >= 0:
                    _st = (_wT / _tt)
                else:
                    _st = 0.0
                tagbox[0] += 1
                _n = nb_blend_row(
                    _pfull, _pre, _brk, _brv, _trk, _trv,
                    _sb, _st, _le, f_K,
                    sp, sr, wk, tagbox[0],
                    hs, his, oi, op,
                    _tgt, ro)
                if _n > f_K:
                    errbox[0] = 2  # heap contract broken (Python would
                    return  # broadcast-crash here too); abort loudly
                if _n < f_K:
                    if f_nolm:
                        for _j in range(_n, f_K):
                            op[_j] = 0.0
                            oi[_j] = (1 << 62)
                        if _n == 0:
                            _rk = -1
                        _n = f_K
                    else:
                        errbox[0] = 1
                        return
                else:
                    _rk = int(ro[0])
                for _j in range(_n):
                    obatch[_pos, _j] = op[_j]
                if _rk < 0 or _ps:
                    for _j in range(_n):
                        otidx[_pos, _j] = oi[_j]
            orank[_pos] = _rk
except Exception:
    _range_numba = None  # numba missing/uncompilable: driver unavailable


def _range_numba_wrap(VV, block_list, T0, lo, hi, coded, prev_tail,
                      bi_id_of, tri_id_of, bi_keys, bi_vals, bi_tot,
                      tri_keys, tri_vals, tri_tot,
                      blk, tv, ti, pi, obatch, orank, opath, otidx, S,
                      f_tri, f_ov0, f_cB, f_cT, f_lam, f_K, f_P,
                      f_eff, f_nolm):
    """Worker-side entry for the njit driver: flatten tables to numba
    shapes (id array + typed row Lists + sorted tri keys), then run.
    Raises like the Python twin on shortfall (via errbox)."""
    from numba import types
    from numba.typed import List
    def _tl(_items, _dt, _as):
        _tl2 = List.empty_list(_dt)
        for _a in _items:
            _tl2.append(np.ascontiguousarray(_a, dtype=_as))
        return _tl2
    block = np.asarray(block_list, dtype=np.int64)
    if prev_tail is None:
        pt_i = np.int64(-1)
        has_pt = False
    else:
        pt_i = np.int64(prev_tail)
        has_pt = True
    bi_idarr = np.full(VV, -1, dtype=np.int64)
    for _tok, _idx in bi_id_of.items():
        if 0 <= _tok < VV:
            bi_idarr[_tok] = int(_idx)
    bi_rk = _tl(bi_keys, types.int64[:], np.int64)
    bi_rv = _tl(bi_vals, types.float64[:], np.float64)
    bi_tots = np.ascontiguousarray(bi_tot, dtype=np.float64)
    _titems = sorted(tri_id_of.items())
    tri_sk = np.empty(len(_titems), dtype=np.int64)
    tri_si = np.empty(len(_titems), dtype=np.int64)
    for _i, ((_a, _b), _idx) in enumerate(_titems):
        if _a >= 65536 or _b >= 65536:
            raise ValueError("tri id overflow for 16-bit packing")
        tri_sk[_i] = (_a << 16) | _b
        tri_si[_i] = _idx
    tri_rk = _tl(tri_keys, types.int64[:], np.int64)
    tri_rv = _tl(tri_vals, types.float64[:], np.float64)
    tri_tots = np.ascontiguousarray(tri_tot, dtype=np.float64)
    sp, sr, wk, hs, his, oi, op, ro = S[0], S[1], S[2], S[3], S[4], S[5], S[6], S[7]
    tagbox = S[8]
    ok, ov, otk, otv = S[9], S[10], S[11], S[12]
    e64 = np.empty(0, dtype=np.int64)
    e64f = np.empty(0, dtype=np.float64)
    errbox = np.zeros(1, dtype=np.int64)
    _range_numba(block, T0, lo, hi, coded, pt_i, has_pt,
                 bi_idarr, bi_rk, bi_rv, bi_tots,
                 tri_sk, tri_si, tri_rk, tri_rv, tri_tots,
                 blk, tv, ti, pi, obatch, orank, opath, otidx,
                 sp, sr, wk, hs, his, oi, op, ro, tagbox,
                 ok, ov, otk, otv, e64, e64f, errbox,
                 f_tri, f_ov0, f_cB, f_cT, f_lam, f_K, f_P,
                 f_eff, f_nolm)
    if errbox[0]:
        raise ArithmeticError("candidate shortfall (numba path)")
    return True


def _freeze_update(counts, totals, id_of, keys, vals, tots, dirty):
    """Incremental freeze: only new + dirty rows are rebuilt, the rest
    are reused verbatim (same arrays, same values as a full rebuild:
    untouched rows can't gain keys or counts). Makes freeze O(dirty)
    instead of O(table) -- mandatory for full-file scaling, where the
    table grows to ~1M rows but dirty stays ~3K/seg."""
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
    print("SMOLLM2 PRACTICAL ENSEMBLE V13 ON ENWIK8")
    print("=" * 70)
    device = torch.device("cuda")
    t0 = time.time()
    if GC_OFF:
        import gc as _gc
        _gc.disable()
    if int(os.environ.get("TORCH1T", "0")):
        # v13-micro: single CPU thread for torch (our work is GPU-bound;
        # stops OpenMP/MKL pool spin stealing launch latency). Default off.
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        print("  [threads] torch single-threaded")
    if CUDNN_BM:
        torch.backends.cudnn.benchmark = True
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
        if int(os.environ.get("MAIN_HIPRI", "0")):
            # v13-micro: main (launch-feeding) thread to HIGHEST priority
            # (no affinity change: keep cache locality, scheduler decides).
            _ct2.windll.kernel32.SetThreadPriority(
                _ct2.windll.kernel32.GetCurrentThread(), 2)  # THREAD_PRIORITY_HIGHEST
            print("  [prio] main thread HIGHEST")
    except Exception as _e:
        print(f"  [prio] skip ({_e})")

    print("\n[1/4] Loading model + tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, torch_dtype=torch.float16
    ).to(device).eval()
    V = model.config.vocab_size
    print(f"  Vocab: {V}")
    # v13-sdpa: auto SDPA picks the MATH fallback on this model (2.3s/fwd);
    # forcing flash/mem-efficient gives 0.20s/fwd (11x) on identical input.
    # SDPA_BACKEND=flash|mem disables math globally; =auto keeps stock.
    _sdpa_be = os.environ.get("SDPA_BACKEND", "flash")
    if _sdpa_be in ("flash", "mem"):
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        print(f"  SDPA backend forced: {_sdpa_be} (math disabled)")
    # v13-rope: rotary cos/sin are identical every same-length forward
    # (9.9s/73s in profile, pure recompute). Cache them: bitwise-identical
    # (same tensors reused), tripwire falls back on non-arange positions
    # (which also keeps CHAIN=1 poisoning impossible).
    _rope_cache = {}
    _rope_mod = model.model.rotary_emb
    _rope_orig_fwd = _rope_mod.forward

    def _rope_cached(hidden_states, position_ids=None):
        try:
            _n = int(position_ids.shape[-1])
            _p0 = int(position_ids[0, 0])
            _p1 = int(position_ids[0, -1])
            if _p0 != 0 or _p1 != _n - 1:
                return _rope_orig_fwd(hidden_states, position_ids)
            _key = (_n, str(position_ids.device), str(hidden_states.dtype))
            _hit = _rope_cache.get(_key)
            if _hit is None:
                _hit = _rope_orig_fwd(hidden_states, position_ids)
                _rope_cache[_key] = _hit
            return _hit
        except Exception:
            return _rope_orig_fwd(hidden_states, position_ids)

    _rope_mod.forward = _rope_cached
    if USE_ORT:
        _ort_init_session(model.config.num_key_value_heads,
                          model.config.hidden_size // model.config.num_attention_heads)  # raises loud if export/packages missing
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
    # v13-proc: shared buffers. INPUTS are double-buffered (parity dim):
    # deferred finish(prev) reads inputs while fwd(cur) writes them, so
    # consecutive segs must land on alternating slices. OUTPUTS stay
    # single-buffered (workers(i) and finish(i) never overlap fwd's
    # writes: fwd never touches outputs). ~3.9GB pagefile-backed.
    _SHM_MAIN.clear()
    _shm_specs = [
        ("blk", (2, BLOCK_TOKENS, V), "float32"),
        ("tv", (2, BLOCK_TOKENS, TOP_K), "float32"),
        ("ti", (2, BLOCK_TOKENS, TOP_K), "int64"),
        ("pi", (2, BLOCK_TOKENS, PREFILTER), "int64"),
        ("obt", (BLOCK_TOKENS, TOP_K), "float64"),
        ("ork", (BLOCK_TOKENS,), "int64"),
        ("opa", (BLOCK_TOKENS,), "int8"),
        ("oti", (BLOCK_TOKENS, TOP_K), "int64"),
    ]
    # v13-proc: shared buffers, allocated LAZILY (proc path only).
    # Default (threads) uses plain numpy: same footprint as v11.
    if USE_PROC_LOOP:
        for _nm, _sh, _dt in _shm_specs:
            _SHM_MAIN[_nm] = _shm_alloc(_nm, _sh, _dt)
        atexit.register(_shm_cleanup)
        _shm_mb = sum(int(np.prod(_sh)) * np.dtype(_dt).itemsize
                      for _, _sh, _dt in _shm_specs) / 1024**2
        print(f"  [shm] {len(_shm_specs)} segments, {_shm_mb:.0f} MB reserved (pagefile-backed, freed at exit)")
        _shm_blk = _SHM_MAIN["blk"][1]
        _shm_tv = _SHM_MAIN["tv"][1]
        _shm_ti = _SHM_MAIN["ti"][1]
        _shm_pi = _SHM_MAIN["pi"][1]
        _shm_obt = _SHM_MAIN["obt"][1]
        _shm_ork = _SHM_MAIN["ork"][1]
        _shm_opa = _SHM_MAIN["opa"][1]
        _shm_oti = _SHM_MAIN["oti"][1]
    else:
        _shm_blk = _shm_tv = _shm_ti = _shm_pi = None
        _shm_obt = _shm_ork = _shm_opa = _shm_oti = None
    bi_counts = {}
    bi_totals = {}
    tri_counts = {}  # (a,b) -> {c: n}
    tri_totals = {}  # (a,b) -> n
    _fz_bi = [{}, [], [], []]  # v13-incr: persistent frozen bigram state
    _fz_tri = [{}, [], [], []]  # v13-incr: persistent frozen trigram state
    prev_tail = None
    prev2_tail = None
    stored = []  # per coded position: (topk_idx, topk_p) for verify reuse
    # v8-memdiag: RSS census per segment (gated: tracing slows allocs)
    import tracemalloc as _tm
    if MEMDIAG:
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
                1 << (AFF_BASE + (_i % max(1, N_LOOP_WORKERS))))
        except Exception:
            pass

    _thr_scratch = []
    # (inline mode N_LOOP_WORKERS<=0 still needs one scratch set)
    for _w in range(max(1, N_LOOP_WORKERS)):
        _thr_scratch.append([
            np.zeros(V, dtype=np.int32), np.zeros(V, dtype=np.int32),
            np.zeros(V, dtype=np.float64), np.empty(TOP_K, dtype=np.float64),
            np.empty(TOP_K, dtype=np.int64), np.empty(TOP_K, dtype=np.int64),
            np.empty(TOP_K, dtype=np.float64), np.zeros(1, dtype=np.int64),
            # v13-numdrv: tag box is numpy (was [0] list) so the njit
            # driver can bump it; same Python syntax everywhere.
            np.zeros(1, dtype=np.int64),
            # v13-gather: pi-truncation temps (>= PREFILTER: pi-subset
            # always fits, so compaction is exact, never lossy)
            np.empty(PREFILTER, dtype=np.int64), np.empty(PREFILTER, dtype=np.float64),
            np.empty(PREFILTER, dtype=np.int64), np.empty(PREFILTER, dtype=np.float64),
        ])
    _pool = ThreadPoolExecutor(max_workers=max(1, N_LOOP_WORKERS),
                               initializer=_pin_worker)
    _rest_pool = ThreadPoolExecutor(max_workers=1)  # v13-pipe: finish helper (never deadlocks: its waits target _pool/_proc_pool)
    _proc_pool = None  # v13-proc: lazy (spawn cost only when enabled)
    _graphs = {}  # CUDA-graph cache: shape -> (graph, static_in, static_out)
    _graph_dead = [False]  # GRAPH_FULL falsified flag (falls back to eager)
    # v13-pinned: PIN_XFER=1 preallocates page-locked host buffers once.
    # D2H copies run async on the compute stream (no drain: the queue
    # stays full, boost residency holds); the CPU waits only at first
    # read via a reused event. Zero math change -> parity must be EXACT.
    _PIN_ON = int(os.environ.get("PIN_XFER", "1"))
    if _PIN_ON:
        _pin_g = torch.empty((BLOCK_TOKENS, PREFILTER), dtype=torch.float32, pin_memory=True)
        _pin_lse = torch.empty((BLOCK_TOKENS,), dtype=torch.float32, pin_memory=True)
        _pin_pi = torch.empty((BLOCK_TOKENS, PREFILTER), dtype=torch.int64, pin_memory=True)
        _pin_ti = torch.empty((BLOCK_TOKENS, TOP_K), dtype=torch.int64, pin_memory=True)
        _pin_tpos = torch.empty((BLOCK_TOKENS, TOP_K), dtype=torch.int64, pin_memory=True)
        _pin_ev = torch.cuda.Event()
        print("  [xfer] pinned async D2H armed")

    def _full_forward_graphed(xin):
        """Full-model single-shape replay. Returns logits [1, L, V] clone.
        Capture: warmup + static buffers + replay. Any failure -> mark
        dead (eager forever) and run eager. Static [1,8192] only; the
        driver guarantees length before calling.
        """
        key = tuple(xin.shape)
        g = _graphs.get(key, "missing")
        if g == "missing" and not _graph_dead[0]:
            try:
                # capture forbids CPU syncs: park the rope-cache patch
                # (its int() syncs) for warmup+capture; replay replays
                # the recorded rope ops, values identical either way.
                try:
                    _rope_mod.forward = _rope_orig_fwd
                except Exception:
                    pass
                st_in = torch.empty_like(xin)
                st_in.copy_(xin)
                for _ in range(3):
                    _ = model(st_in, use_cache=False).logits
                torch.cuda.synchronize()
                st_out = None
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    st_out = model(st_in, use_cache=False).logits
                _graphs[key] = (gr, st_in, st_out)
                g = _graphs[key]
            except Exception as _e:
                print(f"  [graph2] capture failed {key}: {str(_e)[:160]}; eager forever", flush=True)
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                _graph_dead[0] = True
                _graphs[key] = None
                g = None
        if g is None or _graph_dead[0]:
            return model(xin, use_cache=False).logits
        gr, st_in, st_out = g
        st_in.copy_(xin)
        gr.replay()
        return st_out.clone()

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
    _n_plain = _n_blend = 0
    ph_loopA = ph_loopB = 0.0

    torch.cuda.synchronize()
    t = time.time()
    ph_fwd = ph_st = ph_xfer = ph_loop = ph_code = ph_esc = ph_verify = ph_frz = 0.0
    ph_fwd_gpu = 0.0  # CUDA-event pure-GPU forward (immune to CPU contention)
    _ev_pairs = []  # (ev0, ev1) per forward; elapsed summed lazily at end
    # v13-subclocks: split the contaminated fwd window (attn vs topk-launch
    # vs D2H-sync vs shm-writes). Wall clocks; topk is launch-only (async).
    t_attn = t_topk = t_d2h = t_shmw = t_attn_gpu = 0.0
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
    _past = None    # v13: carried KV cache (trimmed to trailing window)
    _abs_pos = 0    # v13: absolute file position of next fresh token
    _carry = None   # v13: last topk rows of previous forward (T0 pair)

    def _seg_finish(block, T0, NC, n_new, tv, ti, pi, blk_probs, lg_full,
                    obatch, esc_vec, sym_arr, orank, opath, otidx,
                    esc_infos, sbase, futs, taA, coded_in, ptail_in,
                    p2tail_in, vok_in, vfail_in, uni_rest, all_ids, V,
                    floor_int, ord=0):
        """Wait + Phase-B + stage-1/2 coding + verify + table update.

        Pure motion of the former inline tail (only renames): waits on
        Phase-A futures, then runs serially. Closes over the run tables
        (mutated in place, never rebound) and _cpu. Returns everything
        the driver accumulates (scalars can't be mutated in place).
        """
        for _f in futs:
            _f.result()
        dA = _cpu() - taA
        _DIRTY_BI.clear()
        _DIRTY_TRI.clear()
        # v11 Phase B: serial bookkeeping in position order, so
        # esc_infos/stored/n_escapes are identical to serial.
        _taB = _cpu()
        # v13-fast: whole-array escape mass (identical values to the
        # per-row loop; numpy reduces each row independently).
        esc_vec[:] = np.maximum(1e-12, 1.0 - obatch.sum(axis=1))
        esc_new = 0
        for _pos in range(NC):
            _t = T0 + _pos
            _rk = int(orank[_pos])
            _tp = obatch[_pos]
            _sample = ((OVERLAP > 0 and _t == T0)
                       or ((coded_in + _pos) % 130 == 0)
                       ) and len(stored) < 300
            if _rk >= 0:
                sym_arr[_pos] = _rk
                esc_infos.append(None)
                if _sample:
                    stored.append((_t, otidx[_pos].copy(), _tp))
            else:
                esc_new += 1
                sym_arr[_pos] = TOP_K
                _tidx = otidx[_pos]
                # rank of target in V-minus-topk, sorted: exactly
                # `target` ints in [0,target), minus those in topk.
                # (Same integer the mask+where path computes; the
                # verify block still uses mask+where as a cross-check.)
                _tgt = block[_t + 1]
                rank_rest = int(_tgt) - len(np.unique(_tidx[_tidx < _tgt]))
                _pv = block[_t]
                _pv2 = block[_t - 1] if _t > 0 else ptail_in
                esc_infos.append((rank_rest, _pv, _pv2, _tidx))
        _n_plain = int((opath == 0).sum())
        _n_blend = int((opath == 1).sum())
        dB = _cpu() - _taB

        _ta = _cpu()
        # K topk + 1 escape = K+1 symbols -> K+2 cum cols
        C = np.empty((NC, TOP_K + 2), dtype=np.int64)
        nb_batch_cum_32(obatch, esc_vec, floor_int, C)
        n1 = int(nb_encode_count_32(C, sym_arr))
        if n1 < 0:
            raise ArithmeticError(f"stage-1 encoder error {n1}")
        bits = n1
        dC = _cpu() - _ta
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
                co = (uni_rest if len(etopk) == TOP_K
                      else uniform_cum_32(V - len(np.unique(etopk))))
            n2_total += int(nb_encode_count_32(
                np.ascontiguousarray(co).reshape(1, -1),
                np.array([rank], dtype=np.int64)))
        bits += n2_total
        dE = _cpu() - _ta
        _ta = _cpu()

        # Verify sampled positions (v8-lean: stored[sbase:] holds
        # ONLY verify candidates as (t_idx, topk, probs) tuples).
        vok, vfail = vok_in, vfail_in
        for (v_tidx, v_topk, v_p) in stored[sbase:]:
            if vok + vfail >= 220:
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
                assert dec.decode_symbol(v_cum) == TOP_K
                mask = np.ones(V, dtype=bool)
                mask[v_topk] = False
                rest = all_ids[mask]
                rank = int(np.where(rest == target)[0][0])
                vprev = block[v_tidx]
                vprev2 = block[v_tidx - 1] if v_tidx > 0 else ptail_in
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
                vok += 1
            else:
                vfail += 1
                print(f"  FAIL at coded-pos {coded_in + (v_tidx - T0)}, pos {v_tidx}")

        new = block[len(block) - n_new:]
        dV = _cpu() - _ta
        seq = ([ptail_in] if ptail_in is not None else []) + new
        for a, b in zip(seq, seq[1:]):
            d = bi_counts.get(a)
            if d is None:
                d = {}
                bi_counts[a] = d
            d[b] = d.get(b, 0) + 1
            bi_totals[a] = bi_totals.get(a, 0) + 1
            _DIRTY_BI.add(a)
        new_p2t = p2tail_in
        if USE_TRIGRAM:
            tseq = ([p2tail_in] if p2tail_in is not None else []) + seq
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
                new_p2t = seq[-2]
        new_pt = new[-1]
        return (bits, esc_new, new_pt, new_p2t, vok, vfail,
                dA, dB, dC, dE, dV, _n_plain, _n_blend)

    with torch.no_grad():
        si = 0
        Pend = None  # v13-pipe: prev seg's submit bundle (finish deferred one step)
        _stash = None  # v13-batch: prefetched (seg meta + logits) from a paired forward
        BATCH_SEGS = int(os.environ.get("BATCH_SEGS", "1"))
        _fpre = None  # v13-prefetch: staged NEXT-seg logits (forward-only;
        # post stays in its own iteration). Lets N+1's GPU work run during
        # N's worker Phase-A + finish instead of idling.
        PREFETCH = int(os.environ.get("PREFETCH", "0"))
        while si < len(segs) or _stash is not None or _fpre is not None:
            # v13: strictly one segment per step -- the KV cache chains
            # seg -> seg, so batching is impossible by design. (BATCH_FWD
            # is accepted but ignored.)
            _have_lg = None
            _skip = False
            if _fpre is not None:
                # consume staged prefetched forward (slot eaten at stage
                # time; tables/carry stayed sequential, so this is exact).
                (_fseg, _flg) = _fpre
                _fpre = None
                (block, T0, NC, n_new) = _fseg
                _have_lg = _flg
                chunk = [(block, T0, NC, n_new)]  # (submit-loop rebind guard)
            elif _stash is not None:
                # consume pair-mate prefetched by the previous step
                # (logits already on GPU; tables/carry stayed sequential).
                # NOTE: no chunk consumed here -- the mate's slot was
                # already eaten when the pair formed (skipping this eats
                # a segment: falsified once at 1.8240, never again).
                (block, T0, NC, n_new, _have_lg) = _stash
                _stash = None
                chunk = [(block, T0, NC, n_new)]  # the submit loop below
                # rebinds block/T0/NC from chunk -- stale chunk would
                # poison the mate with the previous seg's ids (1.3576)
                if int(os.environ.get("SEGBITS", "0")):
                    print(f"  [consume] blk0={block[0]} lgrows={_have_lg.shape[0]}", flush=True)
                if int(os.environ.get("DISCARD_STASH", "0")):
                    _have_lg = None  # autopsy probe: re-forward singly
            else:
                chunk = [segs[si]]
                si += 1
                (block, T0, NC, n_new) = chunk[0]
                # v13-skip gate: periodic, seg0 always kept (cold tables).
                # Causal (seg counter only) so the decoder mirrors exactly;
                # refused unless threads + recompute + ov0 + no-ORT + no-batch.
                _skip = (SKIP_EVERY > 0 and BATCH_SEGS <= 1 and not USE_PROC_LOOP
                         and OVERLAP == 0 and not CHAIN and not USE_ORT
                         and ((si - 1) % SKIP_EVERY == SKIP_MOD % SKIP_EVERY))
            _fr = None
            if PIPELINE and Pend is not None:
                # prev seg's finish runs on the helper while main forwards
                _fr = _rest_pool.submit(_seg_finish, **Pend)
            # v13: KV-cache chaining. Forward ONLY fresh tokens (`new`
            # part, or the whole window for seg0); history rides in the
            # carried cache (trimmed to trailing BLOCK_TOKENS) with
            # absolute position_ids. Fresh rows + one carried row are
            # assembled into full-block tables for the untouched loop.
            _B = len(block)
            _F0 = OVERLAP if (OVERLAP > 0 and _carry is not None) else 0
            # v13-restart: bound rope positions (pure chains diverge past
            # 8K, falsified). A restart re-forwards the full window with
            # positions from 0 -- costs one window per 16K, keeps 1.45x.
            if RESTART_EVERY > 0 and _past is not None and (_abs_pos + NC > RESTART_EVERY):
                _past = None
                _carry = None
                _abs_pos = 0
                _F0 = 0
            _fresh = block if _past is None else block[_F0:_F0 + NC]
            if not CHAIN:
                # recompute mode: every seg is a fresh full window (v11
                # math exactly); chaining machinery idles.
                _past = None
                _carry = None
                _F0 = 0
                _fresh = block
            _ta = time.time()
            _prev_stream = torch.cuda.current_stream()
            if _hi_stream is not None:
                torch.cuda.set_stream(_hi_stream)
            _ev0 = torch.cuda.Event(enable_timing=True)
            _ev1 = torch.cuda.Event(enable_timing=True)
            _ev0.record()
            _xin = None if _have_lg is not None else torch.tensor(_fresh, device=device).unsqueeze(0)
            _nfr = len(_fresh)
            if int(os.environ.get("SHAPE_DEBUG", "0")) and si == 1:
                print(f"  [shdbg] fresh={_nfr} B={_B} NC={NC} F0={_F0} xin={tuple(_xin.shape)}", flush=True)
            _t_attn0 = time.time()
            if _skip:
                # v13-skip: no forward at all; Phase-A runs cache-only
                # (nolm). Tables/carry/finish stay sequential.
                _lg_b = None
            elif _have_lg is not None:
                _lg_b = _have_lg
                _nfr = _lg_b.shape[0]
            elif USE_ORT:
                # EXPERIMENTAL/UNRUN path (see _ort_forward).
                _lg_b = _ort_forward(_xin, _abs_pos, device)
                _abs_pos += _nfr
                del _xin
            elif _past is None:
                # v13-batch: recompute segs are independent (OVERLAP=0),
                # so pair two equal-length blocks into ONE model call.
                # Tables/carry/finish stay strictly sequential, so math is
                # identical up to batched-GEMM last-bit noise (parity gate).
                _bmate = None
                if (BATCH_SEGS > 1 and not CHAIN and OVERLAP == 0
                        and not USE_ORT and si < len(segs)):
                    (_bb, _bT0, _bNC, _bn) = segs[si]
                    if len(_bb) == _nfr:
                        _bmate = (_bb, _bT0, _bNC, _bn)
                if _bmate is not None:
                    if int(os.environ.get("SEGBITS", "0")):
                        print(f"  [pair] si={si} Ablk0={_fresh[0]} Bblk0={_bmate[0][0]} Alen={len(_fresh)} Blen={len(_bmate[0])}", flush=True)
                    _xin2 = torch.tensor(_bmate[0], device=device).unsqueeze(0)
                    _out = model(torch.cat([_xin, _xin2], dim=0),
                                 use_cache=False)
                    _abs_pos += _nfr
                    _lg_b = _out.logits[0]
                    _stash = (_bmate[0], _bmate[1], _bmate[2], _bmate[3],
                              _out.logits[1])
                    si += 1
                    del _out, _xin, _xin2
                else:
                    if GRAPH_FULL and not CHAIN and _nfr == BLOCK_TOKENS:
                        _lg_all = _full_forward_graphed(_xin)
                        _abs_pos += _nfr
                        _lg_b = _lg_all[0]  # [nfr, V] fp16 GPU
                        del _lg_all, _xin
                    else:
                        _out = model(_xin, use_cache=bool(CHAIN))
                        _past = _out.past_key_values if CHAIN else None
                        _abs_pos += _nfr
                        _lg_b = _out.logits[0]  # [nfr, V] fp16 GPU
                        del _out, _xin
            else:
                _cpos = torch.arange(_abs_pos, _abs_pos + _nfr,
                                       device=device)
                _pids = _cpos.unsqueeze(0)
                _out = model(_xin, past_key_values=_past,
                             position_ids=_pids, cache_position=_cpos,
                             use_cache=True)
                del _pids, _cpos
                _past = _out.past_key_values
                _abs_pos += _nfr
                # v13-trim-OFF (falsified 2026-09-13): transformers 5.17
                # re-derives key positions from slot order, so crop()
                # silently renumbers kept keys and absolute queries
                # misalign (2.5229). No trim: 100KB grows cache to ~30K
                # entries (~0.7GB, fits). Full-file needs chain-restarts.
                _lg_b = _out.logits[0]  # [nfr, V] fp16 GPU
                del _out, _xin
            if int(os.environ.get("SHAPE_DEBUG", "0")) and si == 1:
                print(f"  [shdbg] lg={tuple(_lg_b.shape)}", flush=True)
            t_attn += time.time() - _t_attn0
            # v13-drain: SYNC_ATTN=1 inserts a GPU drain right after the
            # model call, so t_attn_gpu isolates attention-GPU-exec from
            # the softmax/topk pile-up that lands in d2h's .cpu() sync.
            # Diagnostic only (a drain the pipeline would pay anyway).
            if int(os.environ.get("SYNC_ATTN", "0")):
                _t_dr0 = time.time()
                torch.cuda.synchronize()
                t_attn_gpu += time.time() - _t_dr0
            if GATHER_PI and USE_GPU_TOPK and not _skip and not CHAIN and not USE_ORT and _have_lg is None:
                # v13-gather: no full-V softmax, no full-V transfer.
                # topk on logits (rank-identical to probs-topk), lse
                # reduction, gather pi-logits, CPU exp. Values match the
                # PI_ONLY number up to fp noise (ratio gate); verify guards.
                _t_topk0 = time.time()
                with torch.no_grad():
                    _lse = torch.logsumexp(_lg_b.float(), dim=-1)  # [nfr] fp32
                    _pv_l, _pi_l = torch.topk(_lg_b.float(), PREFILTER, dim=1)
                    _tv_l, _tpos = torch.topk(_pv_l, TOP_K, dim=1)
                    _ti_l = torch.gather(_pi_l, 1, _tpos)
                    _g = torch.gather(_lg_b.float(), 1, _pi_l)  # [nfr, P]
                t_topk += time.time() - _t_topk0
                _t_d2h0 = time.time()
                if _PIN_ON:
                    _ng = _pin_g[:_nfr]
                    _ng.copy_(_g, non_blocking=True)
                    _nl = _pin_lse[:_nfr]
                    _nl.copy_(_lse, non_blocking=True)
                    _npi = _pin_pi[:_nfr]
                    _npi.copy_(_pi_l, non_blocking=True)
                    _nti = _pin_ti[:_nfr]
                    _nti.copy_(_ti_l, non_blocking=True)
                    _ntp = _pin_tpos[:_nfr]
                    _ntp.copy_(_tpos, non_blocking=True)
                    _pin_ev.record()
                    del _lse, _g, _pi_l, _ti_l, _pv_l, _tv_l, _tpos
                    t_d2h += time.time() - _t_d2h0
                    _t_shmw0 = time.time()
                    # views alias the pinned buffers (no astype copies:
                    # topk indices are int64 already). Alive until the
                    # next seg's copies (this seg's reads finish first).
                    _pin_ev.synchronize()
                    _lse_n = _nl.numpy()
                    _g_n = _ng.numpy()
                    pi_f = _npi.numpy()
                    ti_f = _nti.numpy()
                    _tpos_n = _ntp.numpy()
                else:
                    _lse_n = _lse.cpu().numpy()
                    _g_n = _g.cpu().numpy()
                    pi_f = _pi_l.cpu().numpy().astype(np.int64)
                    ti_f = _ti_l.cpu().numpy().astype(np.int64)
                    _tpos_n = _tpos.cpu().numpy()
                    del _lse, _g, _pi_l, _ti_l, _pv_l, _tv_l, _tpos
                t_d2h += time.time() - _t_d2h0
                _t_shmw0 = time.time()
                # v13-exp16: EXP_FP16=1 does exp in fp16 (2x less traffic;
                # l-lse <= 0 always so no overflow; deep underflow -> 0).
                # Ratio gate +/-5e-4; default off.
                if int(os.environ.get("EXP_FP16", "0")):
                    _pv_f = np.exp(
                        (_g_n - _lse_n[:, None]).astype(np.float16)
                    ).astype(np.float32)
                else:
                    _pv_f = np.exp(_g_n - _lse_n[:, None]).astype(np.float32)
                del _g_n, _lse_n
                _rr0 = np.arange(_nfr)
                tv_f = _pv_f[_rr0[:, None], _tpos_n]  # [nfr, K] probs at ti
                del _tpos_n
                _par = si % 2
                tv = _shm_tv[_par, :_B] if USE_PROC_LOOP else np.zeros((_B, TOP_K), dtype=np.float32)
                ti = _shm_ti[_par, :_B] if USE_PROC_LOOP else np.zeros((_B, TOP_K), dtype=np.int64)
                pi = _shm_pi[_par, :_B] if USE_PROC_LOOP else np.zeros((_B, PREFILTER), dtype=np.int64)
                blk_probs = _shm_blk[_par, :_B] if USE_PROC_LOOP else np.zeros((_B, V), dtype=np.float32)
                lg_full = None
                tv[_F0:_F0 + _nfr] = tv_f
                ti[_F0:_F0 + _nfr] = ti_f
                pi[_F0:_F0 + _nfr] = pi_f
                # sparse blk: values only at pi keys (truncated rows never
                # query outside pi after _PIONLY_EFF truncation).
                _srows = _rr0 + _F0
                blk_probs[_srows[:, None], pi_f] = _pv_f
                if _F0 > 0:
                    tv[T0] = _carry[0]
                    ti[T0] = _carry[1]
                    pi[T0] = _carry[2]
                    blk_probs[T0] = _carry[3]
                _cb = np.zeros(V, dtype=np.float32)
                _cb[pi_f[-1]] = _pv_f[-1]
                _carry = (tv_f[-1].copy(), ti_f[-1].copy(),
                          pi_f[-1].copy(), _cb)
                del _pv_f
                del tv_f, ti_f, pi_f, _cb, _rr0
                t_shmw += time.time() - _t_shmw0
            elif USE_GPU_TOPK and not _skip:
                _t_topk0 = time.time()
                # v13-fp16sm: fp16 softmax halves GPU traffic (WDDM-proven:
                # pure PCIe is 0.12s; the 6.1s d2h is GPU fp32-softmax exec).
                # Values shift in last bits -> ratio gate +/-3e-4, verify must pass.
                _FP16SM = int(os.environ.get("USE_FP16_SOFTMAX", "1"))
                with torch.no_grad():
                    probs_b = torch.softmax(
                        _lg_b if _FP16SM else _lg_b.float(), dim=-1)
                (tv_f, ti_f), (pv_f, pi_f) = block_topk_gpu(probs_b, 0)
                if int(os.environ.get("SHAPE_DEBUG", "0")) and si == 1:
                    print(f"  [shdbg] probs={tuple(probs_b.shape)} tv_f={tuple(np.shape(tv_f))} ti_f={tuple(np.shape(ti_f))}", flush=True)
                t_topk += time.time() - _t_topk0
                _t_d2h0 = time.time()
                # (no explicit sync: stream order + the .cpu() below already syncs)
                if USE_FP16_XFER:
                    blk_f = probs_b.half().cpu().numpy().astype(np.float32)
                else:
                    blk_f = probs_b.float().cpu().numpy()
                probs_b = None
                t_d2h += time.time() - _t_d2h0
                _t_shmw0 = time.time()
                _par = si % 2  # ping-pong: prev seg's finish may still read the other slice
                tv = _shm_tv[_par, :_B] if USE_PROC_LOOP else np.zeros((_B, TOP_K), dtype=np.float32)
                ti = _shm_ti[_par, :_B] if USE_PROC_LOOP else np.zeros((_B, TOP_K), dtype=np.int64)
                pi = _shm_pi[_par, :_B] if USE_PROC_LOOP else np.zeros((_B, PREFILTER), dtype=np.int64)
                blk_probs = _shm_blk[_par, :_B] if USE_PROC_LOOP else np.zeros((_B, V), dtype=np.float32)
                lg_full = None
                tv[_F0:_F0 + _nfr] = tv_f
                ti[_F0:_F0 + _nfr] = ti_f
                pi[_F0:_F0 + _nfr] = pi_f
                blk_probs[_F0:_F0 + _nfr] = blk_f
                if _F0 > 0:
                    tv[T0] = _carry[0]
                    ti[T0] = _carry[1]
                    pi[T0] = _carry[2]
                    blk_probs[T0] = _carry[3]
                _carry = (tv_f[-1].copy(), ti_f[-1].copy(),
                          pi_f[-1].copy(), blk_f[-1].copy())
                del tv_f, ti_f, pv_f, pi_f, blk_f
                t_shmw += time.time() - _t_shmw0
            elif not _skip:
                _par = si % 2  # same ping-pong (submit always references it)
                lg_f = _lg_b.float().cpu().numpy()
                lg_full = np.zeros((_B, V), dtype=np.float64)
                lg_full[_F0:_F0 + _nfr] = lg_f
                tv = ti = pv = pi = blk_probs = None
                if _F0 > 0:
                    lg_full[T0] = _carry[0]
                _carry = (lg_f[-1].copy(),)
                lg = lg_full - lg_full.max(axis=1, keepdims=True)
                e = np.exp(lg, dtype=np.float64)
                probs_cpu = e / e.sum(axis=1, keepdims=True)
                blk_probs = probs_cpu
                probs_b = None
                del lg, e, lg_f
            else:
                # v13-skip inits: zeros (p_row for lam=0 reads 0.0);
                # pi overwritten by the cache-union fill before submit.
                # tv/ti unread (nolm forces blend); _carry cleared
                # (only read when _F0 > 0, impossible at ov0).
                _par = 0  # skip: shm ping-pong unused (threads zeros path)
                tv = np.zeros((_B, TOP_K), dtype=np.float32)
                ti = np.zeros((_B, TOP_K), dtype=np.int64)
                pi = np.zeros((_B, PREFILTER), dtype=np.int64)
                blk_probs = np.zeros((_B, V), dtype=np.float32)
                lg_full = None
                _carry = None
            if _lg_b is not None:
                del _lg_b
            if EMPTY_EVERY > 0:
                torch.cuda.empty_cache()
            _ev1.record()
            # (no explicit sync: ordering via stream + .cpu() syncs; GPU
            # time is summed lazily at end from recorded events, so this
            # segment's queue-wait is not paid here)
            if _hi_stream is not None:
                torch.cuda.set_stream(_prev_stream)
            ph_fwd += time.time() - _ta
            _ev_pairs.append((_ev0, _ev1))
            # v13-prefetch: launch NEXT seg's forward NOW (tokens known
            # upfront) so its GPU work runs during this seg's worker
            # Phase-A + finish instead of idling. Forward-only: post stays
            # in its own iteration (needs its syncs). Same stream as main
            # forwards (ordered, no cross-stream hazard). Tables/finish/
            # carry stay strictly sequential; values bit-identical
            # (same call, earlier launch); verify guards the mirror.
            # Strict recompute-single path only, else silent fallback.
            if (PREFETCH and not CHAIN and OVERLAP == 0 and not USE_ORT
                    and BATCH_SEGS <= 1 and SKIP_EVERY == 0 and not GRAPH_FULL
                    and _stash is None and _fpre is None and si < len(segs)):
                (_pb2, _pT0, _pNC, _pn2) = segs[si]
                _px = torch.tensor(_pb2, device=device).unsqueeze(0)
                if _hi_stream is not None:
                    torch.cuda.set_stream(_hi_stream)
                _fev0 = torch.cuda.Event(enable_timing=True)
                _fev1 = torch.cuda.Event(enable_timing=True)
                _fev0.record()
                _pout = model(_px, use_cache=False)
                _fev1.record()
                _ev_pairs.append((_fev0, _fev1))
                if _hi_stream is not None:
                    torch.cuda.set_stream(_prev_stream)
                _abs_pos += len(_pb2)
                _fpre = ((_pb2, _pT0, _pNC, _pn2), _pout.logits[0])
                si += 1
                del _pout, _px
            _t0w = time.time()
            if Pend is not None:
                if _fr is not None:
                    _res = _fr.result()
                else:
                    _res = _seg_finish(**Pend)
                (_b_new, _e_new, _pt_new, _p2t_new, _vo, _vf,
                 _dA, _dB, _dC, _dE, _dV, _np, _nb) = _res
                total_bits += _b_new
                n_escapes += _e_new
                prev_tail, prev2_tail = _pt_new, _p2t_new
                n_verify_ok, n_verify_fail = _vo, _vf
                ph_loopA += _dA
                ph_loopB += _dB
                ph_code += _dC
                ph_esc += _dE
                ph_verify += _dV
                _n_plain += _np
                _n_blend += _nb
                coded_count += Pend["NC"]
                _done = Pend["ord"]
                if int(os.environ.get("SEGBITS", "0")):
                    print(f"  [segbits] ord={_done} NC={Pend['NC']} bits={_b_new} blk0={Pend['block'][0]} T0={Pend['T0']}", flush=True)
                Pend = None
                if EMPTY_EVERY > 0 and si % EMPTY_EVERY == 0:
                    torch.cuda.empty_cache()
                if MEMDIAG:
                    _snap = _tm.take_snapshot()
                    _top = _snap.statistics("filename")[:3]
                    _top_s = "; ".join(f"{s.traceback[0].filename.split(chr(92))[-1]}:{s.traceback[0].lineno} {s.size/1e6:.0f}MB" for s in _top)
                    _meminfo = f" | {_top_s}"
                else:
                    _meminfo = ""
                print(f"  ... seg {_done}/{len(segs)}, escapes: {n_escapes}, "
                      f"bigram keys: {len(bi_counts)}, coded: {coded_count}, "
                      f"stored={len(stored)} trikeys={len(tri_counts)}{_meminfo}", flush=True)
            ph_loop += time.time() - _t0w
            for k, (block, T0, NC, n_new) in enumerate(chunk):
                if NC < 1 or len(block) < 2:
                    continue

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
                # v13-unionprobe: DUMP_TABLES=N pickles frozen tables for
                # seg ord N (offline union-size analysis; zero hot-path
                # cost otherwise).
                if int(os.environ.get("DUMP_TABLES", "-1")) == si - 1:
                    import pickle as _pk
                    with open(f"logs/tables_seg{si - 1}.pkl", "wb") as _fh:
                        _pk.dump({"block": block, "T0": T0, "NC": NC,
                                  "prev_tail": prev_tail,
                                  "bi_id_of": bi_id_of, "bi_keys": bi_keys,
                                  "tri_id_of": tri_id_of, "tri_keys": tri_keys},
                                 _fh, protocol=4)
                    print(f"  [unionprobe] dumped seg {si - 1}", flush=True)
                # v13-skip: cache-union prefilter fill (no LM topk ran).
                # pre = dedup(bigram row + trigram row) capped at
                # PREFILTER; the kernel's brow/trow-exclusive loops cover
                # anything truncated, and zero-tail is mirror-consistent
                # (decoder runs this same fill). Timed into freeze.
                _nolm = bool(_skip)
                if _nolm:
                    _taU = _cpu()
                    for _upos in range(NC):
                        _t = T0 + _upos
                        _pv = block[_t]
                        _pv2 = block[_t - 1] if _t > 0 else prev_tail
                        _uk = []
                        _useen = set()
                        _ib = bi_id_of.get(_pv, -1)
                        if _ib >= 0:
                            for _kk in bi_keys[_ib]:
                                _kk = int(_kk)
                                if _kk not in _useen:
                                    _useen.add(_kk)
                                    _uk.append(_kk)
                                    if len(_uk) >= PREFILTER:
                                        break
                        if USE_TRIGRAM and _pv2 is not None:
                            _it = tri_id_of.get((_pv2, _pv), -1)
                            if _it >= 0:
                                for _kk in tri_keys[_it]:
                                    _kk = int(_kk)
                                    if _kk not in _useen:
                                        _useen.add(_kk)
                                        _uk.append(_kk)
                                        if len(_uk) >= PREFILTER:
                                            break
                        if _uk:
                            pi[_t, :len(_uk)] = _uk
                    ph_frz += _cpu() - _taU  # union fill timed as freeze
                # ---- v9 phase-1: per-position blend+rank, batched cum ----
                # v13-proc: outputs live in shared memory (workers write,
                # main reads); esc_vec/sym_arr stay local (serial Phase-B).
                topk_batch = _shm_obt[:NC] if USE_PROC_LOOP else np.empty((NC, TOP_K), dtype=np.float64)
                esc_vec = np.empty(NC, dtype=np.float64)
                sym_arr = np.empty(NC, dtype=np.int64)
                _ta = time.time()
                rank_arr = _shm_ork[:NC] if USE_PROC_LOOP else np.empty(NC, dtype=np.int64)
                path_arr = _shm_opa[:NC] if USE_PROC_LOOP else np.empty(NC, dtype=np.int8)
                tidx_batch = _shm_oti[:NC] if USE_PROC_LOOP else np.empty((NC, TOP_K), dtype=np.int64)
                # v11 Phase A: pure per-position work, 4 threads. Reads:
                # block, frozen tables, blk_probs, ti/tv/pi. Writes:
                # distinct rows of topk_batch/rank_arr/path_arr/tidx_batch.
                _T0 = T0
                _coded = coded_count
                _taA = _cpu()

                # v13-proc: single-source core, two executors. Threads share
                # arrays in-process; procs resolve the same math through
                # shared memory. Default: threads (validated path).
                # N_LOOP_WORKERS<=0: inline in main (zero threads, zero GIL
                # fight during launches; same math, verify guards).
                # v13-numdrv: njit driver in the worker (same math, nogil
                # compute + free main for launches); refused unless
                # threads + gputopk + 1 worker + probe off.
                if N_LOOP_WORKERS <= 0 and not USE_PROC_LOOP:
                    _range_core(block, T0, T0, T0 + NC, _coded, prev_tail,
                                bi_id_of, tri_id_of, bi_keys, bi_vals, bi_tot,
                                tri_keys, tri_vals, tri_tot,
                                blk_probs, tv, ti, pi,
                                topk_batch, rank_arr, path_arr, tidx_batch,
                                _thr_scratch[0], _nolm)
                    _futs = []
                elif (NUMBA_DRIVER and _range_numba is not None
                        and not USE_PROC_LOOP and USE_GPU_TOPK
                        and not _EXCLPROBE and N_LOOP_WORKERS == 1):
                    _futs = [_pool.submit(
                        _range_numba_wrap, V, block, T0, T0, T0 + NC,
                        _coded, prev_tail,
                        bi_id_of, tri_id_of, bi_keys, bi_vals, bi_tot,
                        tri_keys, tri_vals, tri_tot,
                        blk_probs, tv, ti, pi,
                        topk_batch, rank_arr, path_arr, tidx_batch,
                        _thr_scratch[0],
                        USE_TRIGRAM, OVERLAP > 0, BIGRAM_CONF, TRIGRAM_CONF,
                        BIGRAM_LAMBDA, TOP_K, PREFILTER,
                        bool(_PIONLY_EFF), bool(_nolm))]
                elif USE_PROC_LOOP:
                    if _proc_pool is None:
                        _proc_pool = ProcessPoolExecutor(
                            max_workers=N_PROC, initializer=_proc_init,
                            initargs=([[_nm, list(_sh), _dt]
                                       for _nm, _sh, _dt in _shm_specs],
                                      V, TOP_K))
                        print(f"  [proc] {N_PROC} workers on shared buffers")
                    _step = (NC + N_PROC - 1) // N_PROC
                    _futs = []
                    for _w in range(N_PROC):
                        _lo = T0 + min(_w * _step, NC)
                        _hi = T0 + min((_w + 1) * _step, NC)
                        if _hi > _lo:
                            _futs.append(_proc_pool.submit(
                                _proc_range, block, T0, _coded, prev_tail,
                                bi_id_of, tri_id_of, bi_keys, bi_vals, bi_tot,
                                tri_keys, tri_vals, tri_tot, _lo, _hi, _par))
                else:
                    _step = (NC + N_LOOP_WORKERS - 1) // N_LOOP_WORKERS
                    _futs = []
                    for _w in range(N_LOOP_WORKERS):
                        _lo = T0 + min(_w * _step, NC)
                        _hi = T0 + min((_w + 1) * _step, NC)
                        if _hi > _lo:
                            _futs.append(_pool.submit(
                                _range_core, block, T0, _lo, _hi, _coded, prev_tail,
                                bi_id_of, tri_id_of, bi_keys, bi_vals, bi_tot,
                                tri_keys, tri_vals, tri_tot,
                                blk_probs, tv, ti, pi,
                                topk_batch, rank_arr, path_arr, tidx_batch,
                                _thr_scratch[_w], _nolm))
                Pend = dict(block=block, T0=T0, NC=NC, n_new=n_new, tv=tv, ti=ti, pi=pi, blk_probs=blk_probs, lg_full=lg_full, obatch=topk_batch, esc_vec=esc_vec, sym_arr=sym_arr, orank=rank_arr, opath=path_arr, otidx=tidx_batch, esc_infos=esc_infos, sbase=sbase, futs=_futs, taA=_taA, coded_in=_coded, ptail_in=prev_tail, p2tail_in=prev2_tail, vok_in=n_verify_ok, vfail_in=n_verify_fail, uni_rest=uni_rest_cum, all_ids=all_ids, V=V, floor_int=_FLOOR_INT, ord=si)
                # (Phase-B/code/verify/tables moved into _seg_finish above;
                # this segment's finish runs deferred via Pend.)

                # (stage-1/2 coding moved into _seg_finish; runs deferred.)

                # (verify moved into _seg_finish; runs deferred.)

                # (tables moved into _seg_finish; runs deferred.)

                # (accumulate/release/progress moved to the Pend join above;
                # the bundle's rebinding frees last segment's views.)

        # v13-pipe drain: finish the last pending segment (synchronous;
        # nothing left to overlap it with).
        if Pend is not None:
            _t0w = time.time()
            _res = _seg_finish(**Pend)
            (_b_new, _e_new, _pt_new, _p2t_new, _vo, _vf,
             _dA, _dB, _dC, _dE, _dV, _np, _nb) = _res
            total_bits += _b_new
            n_escapes += _e_new
            prev_tail, prev2_tail = _pt_new, _p2t_new
            n_verify_ok, n_verify_fail = _vo, _vf
            ph_loopA += _dA
            ph_loopB += _dB
            ph_code += _dC
            ph_esc += _dE
            ph_verify += _dV
            _n_plain += _np
            _n_blend += _nb
            coded_count += Pend["NC"]
            Pend = None
            if EMPTY_EVERY > 0 and si % EMPTY_EVERY == 0:
                torch.cuda.empty_cache()
            print(f"  ... seg {len(segs)}/{len(segs)}, escapes: {n_escapes}, "
                  f"bigram keys: {len(bi_counts)}, coded: {coded_count}, "
                  f"stored={len(stored)} trikeys={len(tri_counts)}", flush=True)
            ph_loop += time.time() - _t0w

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
    if _EXCLPROBE:
        _e1, _e0 = _EXCL[0], max(1, _EXCL[1])
        print(f"  EXCL: exclusive-target hits={_e1}/{_e0} ({100.0 * _e1 / _e0:.2f}% of blend positions)")
    torch.cuda.synchronize()  # single drain: all recorded events complete here
    for _a, _b in _ev_pairs:
        ph_fwd_gpu += _a.elapsed_time(_b) / 1000.0
    print(f"  CLOCKS: gpu_fwd={ph_fwd_gpu:.1f}s fwd_wall={ph_fwd:.1f}s "
           f"stolen_fwd~={ph_fwd - ph_fwd_gpu:.1f}s stolen_loop~={ph_loop - (ph_loopA / 4 + ph_loopB):.1f}s")
    print(f"  SUBCLOCKS: attn={t_attn:.1f}s topk_launch={t_topk:.1f}s d2h={t_d2h:.1f}s shmw={t_shmw:.1f}s attn_gpu={t_attn_gpu:.1f}s")
    _pool.shutdown()
    _rest_pool.shutdown()
    if _proc_pool is not None:
        _proc_pool.shutdown()
    if GC_OFF:
        _gc.enable()
        _gc.collect()
    print(f"  PEAK_VRAM={torch.cuda.max_memory_allocated()/1024**3:.2f}GB "
          f"(bf={BATCH_FWD} ov={OVERLAP})")

    _kb = int(os.environ.get("ENWIK8_KB", "100"))
    tag = (f"off{_off_mb}_lam{BIGRAM_LAMBDA}_gputopk{USE_GPU_TOPK}_bf{BATCH_FWD}"
           f"_pf{PREFILTER}_tri{USE_TRIGRAM}_bc{BIGRAM_CONF}_tc{TRIGRAM_CONF}_k{TOP_K}_ov{OVERLAP}"
           f"_f{FLOOR_FRAC:g}_s2{USE_CACHE_S2}_nb{USE_NUMBA_LOOP}_kb{_kb}")
    results = {
        "model": "SmolLM2-135M-practical-ensemble-v13",
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
    with open(f"./data/smollm2_ensemble_v13_{tag}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to data/smollm2_ensemble_v13_{tag}.json")


if __name__ == "__main__":
    main()
