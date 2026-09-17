"""Locate the first enc/dec divergence on the real-distribution path.

This is the tool to reach for when `tools/test_seg_codec_mirror.py` reports a
mismatch on the F1/F2 path but not on the stand-in path: it runs the same
round-trip and compares the encoder's and decoder's per-position logs, so the
first divergent position is printed with the state around it (segment, prev /
prev2, the head of the alphabet, the CDF total, the escape mass).

Two traps it already caught, both in the HARNESS rather than in seg_codec:

  1. `Enc.finish()` returns `(bitstr, count)`; passing the tuple straight into
     `Dec(...)` makes the decoder zero-pad after 2 "bits" and diverge at the
     first symbol. Unpack it.
  2. The pure-Python port in tools/audit_ac32_packing.py does exact big-int
     arithmetic. Feed it a numpy int64 CDF (what the real builder produces) and
     its state turns into numpy scalars, after which `x * TOTAL` raises
     OverflowError: "Python int too large to convert to C long". The stand-in
     encoder/decoder in the mirror test now coerce with `.tolist()`.

Run:  python3 tools/debug_f1_mirror.py            (needs numpy + numba)
"""
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np                      # noqa: E402
import seg_codec as sc                  # noqa: E402
import test_seg_codec_mirror as m       # noqa: E402

V = m.V
CFG = dict(m.REAL_CFG)
ENC_LOG, DEC_LOG = [], []


def make_dist(log):
    """sc.build_distribution, but recording what each position produced."""
    def f(probs, tables, s, prev, prev2, cfg, uni_cache):
        p = np.asarray(probs, dtype=np.float64)
        d = sc.build_distribution(p, tables, s, prev, prev2, cfg, uni_cache)
        tk = np.asarray(d.topk)
        log.append(dict(
            n=len(log), s=s, prev=prev, prev2=prev2,
            tk=[int(x) for x in tk[:4]], nk=len(tk),
            cdf=int(np.asarray(d.cum_s1)[-1]),
            esc=round(float(1.0 - np.sum(p[tk])), 12),
        ))
        return d
    return f


def wrap_coder(obj, log, is_enc):
    """Record every symbol the coder handles, in order."""
    if is_enc:
        orig = obj.encode_symbol

        def enc_symbol(cum, sym):
            log.append(dict(n=len(log), sym=int(sym),
                            total=int(np.asarray(cum)[-1])))
            return orig(cum, sym)
        obj.encode_symbol = enc_symbol
    else:
        orig = obj.decode_symbol

        def dec_symbol(cum):
            sym = orig(cum)
            log.append(dict(n=len(log), sym=int(sym),
                            total=int(np.asarray(cum)[-1])))
            return sym
        obj.decode_symbol = dec_symbol
    return obj


def run(n_segments, n_tokens, seed=4242, model=None, dec_model=None):
    del ENC_LOG[:], DEC_LOG[:]
    model = model or m.fake_next_logits
    dec_model = dec_model or model
    rng = random.Random(seed)
    ids = [rng.randrange(V) for _ in range(n_tokens)]

    enc = wrap_coder(m.StandInEncoder(), ENC_LOG, True)
    ucache_e, ucache_d = {}, {}
    te, td = sc.SegTables(n_segments), sc.SegTables(n_segments)
    stats = dict(coded=0, escapes=0, uniform=0)
    sc.encode_stream(ids, n_segments, CFG, model, enc, te,
                     m.uniform_V_cum(ucache_e), make_dist(ENC_LOG), ucache_e,
                     stats)
    bitstr, count = enc.finish()          # trap 1: unpack, do not pass the tuple

    dec = wrap_coder(m.StandInDecoder(bitstr), DEC_LOG, False)
    stats_d = dict(coded=0, escapes=0, uniform=0)
    out = sc.decode_stream(n_tokens, n_segments, CFG, dec_model, dec, td,
                           m.uniform_V_cum(ucache_d), make_dist(DEC_LOG),
                           ucache_d, stats_d)
    ok = list(out) == list(ids)
    print(f"K={n_segments} n={n_tokens} roundtrip={ok} bits={count} "
          f"escapes={stats['escapes']}")
    n = min(len(ENC_LOG), len(DEC_LOG))
    first = next((i for i in range(n) if ENC_LOG[i] != DEC_LOG[i]), None)
    if first is None and len(ENC_LOG) != len(DEC_LOG):
        first = n
    print(f"  enc log={len(ENC_LOG)} dec log={len(DEC_LOG)} "
          f"first diff at {first}")
    if first is not None:
        for j in range(max(0, first - 2), min(n, first + 3)):
            print(f"   [{j}] enc={ENC_LOG[j]}")
            print(f"        dec={DEC_LOG[j]}")
    if not ok:
        bad = next((i for i, (a, b) in enumerate(zip(ids, out)) if a != b), None)
        print(f"  first token mismatch at position {bad}: "
              f"enc={ids[bad] if bad is not None else None} "
              f"dec={out[bad] if bad is not None else None}")
    return ok


if __name__ == "__main__":
    try:
        np.array([1.0]) @ np.array([1.0])
        sc._ac32()
    except Exception as exc:
        print(f"needs numpy + numba ({type(exc).__name__}: {exc})")
        sys.exit(2)
    good = run(1, 120)
    print()
    good &= run(4, 220, model=m.degenerate_next_logits)
    print()
    bad = run(1, 60, dec_model=m.fake_next_logits_sabotaged)
    print(f"\nself-check: clean run passes={good}, sabotaged run is caught={not bad}")
    sys.exit(0 if (good and not bad) else 1)
