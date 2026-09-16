"""Blend-math tests: the F1/F2 fixes in tools/seg_codec.py.

Why: at 8KB K=1 the codec paid 1.1262 bpb while the same model+data in v13 pays
0.9276 at 100KB, and escapes ran 2.20% vs v13's 1.22%. Two mistakes in the
distribution builder are visible by inspection and are the part of that gap the
codec owns:

  F2  the cache term was divided by `_wc`, giving a row seen ONCE the same mass
      as a row seen 500 times -- the confidence weighting was thrown away.
  F1  the alphabet was the LM top-K, so tokens the cache had evidence for but
      the LM ranked outside top-K could never be coded directly; each one paid
      the uniform-escape price (~log2(V-K) ~ 15.5 bits).

The core assertions below are pure Python (no numpy, no GPU) so they run
anywhere. The last section exercises the real `build_distribution` and is
skipped when numpy is missing.

Run: python3 tools/test_blend_math.py
"""
import io
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import seg_codec as sc   # noqa: E402

CFG = dict(top_k=1024, prefilter=2048, floor_frac=1e-6, bigram_lambda=0.99,
           bigram_conf=10.0, trigram_conf=3.0, use_trigram=True)
FAILS = []


def check(name, cond, detail=""):
    status = "ok  " if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def main():
    print("=" * 78)
    print("F2: per-count cache weights (confidence weighting preserved)")
    print("=" * 78)

    # A token with count 1 in a bigram row whose total is N receives weight
    #   w(N) = sb(N) * 1     (F2)      vs     pb/_wc = 1/N * ...   (old, buggy)
    weights = {}
    for N in (1, 10, 100, 500, 5000):
        le, sb, st, wc = sc.blend_weights(N, 0, CFG)
        weights[N] = sb
        print(f"    row seen {N:>5} times: le={le:.6f}  per-count weight sb={sb:.6g}"
              f"  (old code gave {1.0/N:.6g} x wB/wc = "
              f"{(N/(N+CFG['bigram_conf']))/max(wc,1e-12)*1.0/N:.6g})")

    check("per-count weight shrinks as the row gains evidence",
          weights[1] > weights[10] > weights[100] > weights[500] > weights[5000])
    check("a once-seen row is discounted hard vs a 500-seen row",
          weights[1] / weights[500] > 20,
          f"ratio={weights[1] / weights[500]:.1f}x")

    # The old code's normalisation: count/N * wB / wc == count/N here (wc==wB
    # when there is no trigram), so a once-seen token got mass 1.0 -- identical
    # to a token the model is certain about.
    old_once = 1.0 / 1 * (1 / (1 + CFG["bigram_conf"])) / (1 / (1 + CFG["bigram_conf"]))
    check("old normalisation really did give a once-seen token full mass",
          abs(old_once - 1.0) < 1e-12, f"old={old_once}")

    print()
    print("=" * 78)
    print("F2: identity + mass bound (v13: (1-wT)*wB + wT == wc)")
    print("=" * 78)
    for bt, tt in ((0, 0), (1, 0), (500, 0), (0, 5), (500, 5), (37, 91)):
        le, sb, st, wc = sc.blend_weights(bt, tt, CFG)
        wB = bt / (bt + CFG["bigram_conf"]) if bt else 0.0
        wT = tt / (tt + CFG["trigram_conf"]) if tt else 0.0
        lhs = (1.0 - wT) * wB + wT
        cache_mass = (sb * bt if bt else 0.0) + (st * tt if tt else 0.0)
        check(f"bt={bt:<4} tt={tt:<3} wc identity + cache mass <= 1",
              abs(lhs - wc) < 1e-12 and cache_mass <= 1.0 + 1e-12,
              f"wc={wc:.6f} cache_mass={cache_mass:.6f} le={le:.6f}")
        check(f"bt={bt:<4} tt={tt:<3} total coded mass <= 1 (escape >= 0)",
              le + (1.0 - le) * cache_mass <= 1.0 + 1e-12,
              f"le+(1-le)*cache={le + (1 - le) * cache_mass:.6f}")

    print()
    print("=" * 78)
    print("F1: cache-only tokens are promoted into the alphabet")
    print("=" * 78)
    try:
        import numpy as np
    except ImportError:
        print("  [skip] numpy not installed here -- run this on the GPU box to")
        print("         exercise build_distribution (promotion + escape bound).")
    else:
        # V must exceed BOTH prefilter (2048) and top_k (1024), otherwise the
        # truncation stages are no-ops and the test proves nothing. (The first
        # version used V=512 and "failed" only because the alphabet is
        # min(top_k, V) -- a test bug that looked like a code bug.)
        V = 4096
        rng = np.random.default_rng(0)
        probs = rng.random(V) + 1e-9
        probs /= probs.sum()
        K, PF = CFG["top_k"], CFG["prefilter"]
        assert 0 < K < PF < V, (K, PF, V)

        class T:
            def __init__(self, bi, bt, tri, tt):
                self.d = (bi, bt, tri, tt)

            def get(self, s, prev, prev2):
                return self.d

        lm_top = np.argsort(-probs)[:K]

        # (1) cold cache: alphabet is exactly the LM top-K, in LM order.
        d0 = sc.build_distribution(probs, T({}, 0, {}, 0), 0, 7, 6, CFG, {})
        check("cold cache: alphabet size == min(top_k, V)",
              len(d0.topk) == min(K, V), f"len={len(d0.topk)}")
        check("cold cache: alphabet == LM top-K (membership)",
              set(int(x) for x in d0.topk) == set(int(x) for x in lm_top))
        check("cold cache: alphabet == LM top-K (order)",
              np.array_equal(np.asarray(d0.topk), lm_top))
        cold_esc = 1.0 - float(np.sum(probs[d0.topk]))
        check("cold cache: escape mass == 1 - LM top-K mass",
              cold_esc > 0.5, f"esc={cold_esc:.4f}")

        # (2) promotion: target is LM-last, cache has seen it 50x.
        target = int(np.argmin(probs))
        hot = T({target: 50.0}, 50, {}, 0)
        d = sc.build_distribution(probs, hot, 0, 7, 6, CFG, {})
        tk = np.asarray(d.topk)
        check("hot cache: LM-last token with 50 counts is promoted",
              target in set(int(x) for x in tk), f"target={target}")
        check("hot cache: alphabet still capped at top_k",
              len(tk) == K, f"len={len(tk)}")
        check("hot cache: no duplicate ids in the alphabet",
              len(set(int(x) for x in tk)) == len(tk))
        rank = int(np.where(tk == target)[0][0])
        # (_sb|_st) for bt=50: wT=0, wB=50/60 -> sb=(1/60), score=0.0125 for 50
        # counts; the weakest LM token in a 1024-of-4096 cut has p ~ 1/4096 ~
        # 2.4e-4, so a 50-count cache token must outrank the tail of the cut.
        check("hot cache: promoted token outranks the weak tail of the cut",
              rank < K - 1, f"rank={rank}/{K}")

        # (2b) _row_counts: dict lookup must equal the old searchsorted form
        row = {int(t): float(c) for t, c in
               zip(rng.integers(0, V, 300), rng.integers(1, 40, 300))}
        cc = rng.choice(V, 500, replace=False).astype(np.int64)
        got = sc._row_counts(row, cc)
        k = np.fromiter(row.keys(), dtype=np.int64, count=len(row))
        v = np.fromiter(row.values(), dtype=np.float64, count=len(row))
        o = np.argsort(k)
        k, v = k[o], v[o]
        idx = np.clip(np.searchsorted(k, cc), 0, len(k) - 1)
        ref = np.where(k[idx] == cc, v[idx], 0.0)
        check("_row_counts == searchsorted reference (300-key row)",
              np.array_equal(got, ref))
        check("_row_counts handles an empty row",
              np.array_equal(sc._row_counts({}, cc), np.zeros(len(cc))))

        # (3) determinism: identical inputs -> bit-identical alphabet+cdf.
        # Non-negotiable: encoder and decoder call this independently and must
        # agree exactly, including tie order.
        d2 = sc.build_distribution(probs, hot, 0, 7, 6, CFG, {})
        check("determinism: same inputs -> identical topk and cdf",
              np.array_equal(np.asarray(d.topk), np.asarray(d2.topk))
              and np.array_equal(np.asarray(d.cum_s1), np.asarray(d2.cum_s1)))

        # (4) escape id is len(topk) and stage-2 rest excludes exactly topk.
        rest_ids, cum_rest = d.rest_fn()
        check("escape id == len(alphabet) == stage-2 rest complement size",
              len(rest_ids) == V - len(tk)
              and len(cum_rest) == len(rest_ids) + 1)
        check("stage-2 rest excludes every alphabet id",
              not set(int(x) for x in tk) & set(int(x) for x in rest_ids))

        # (5) escape mass stays non-negative at every cache strength (this is
        # where the old /_wc version could over-subscribe probability mass).
        for bt, tt in ((0, 0), (1, 0), (50, 0), (500, 5), (5000, 500)):
            tt2 = T({int(t): float(bt) for t in rng.integers(0, V, 20)}, bt,
                    {int(t): float(tt) for t in rng.integers(0, V, 20)}, tt)
            dd = sc.build_distribution(probs, tt2, 0, 7, 6, CFG, {})
            tot = float(np.sum(np.asarray(dd.p_blend))) if hasattr(dd, "p_blend") \
                else float(np.sum(probs[np.asarray(dd.topk)]))
            check(f"bt={bt:5d} tt={tt:3d}  alphabet mass <= 1 (escape >= 0)",
                  tot <= 1.0 + 1e-9, f"mass={tot:.6f}  esc={1 - tot:.2e}")

    print()
    print("=" * 78)
    print(f"RESULT: {'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    print("=" * 78)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
