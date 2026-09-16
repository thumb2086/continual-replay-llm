"""CPU mirror test for tools/seg_codec.py -- runs on the stdlib alone.

WHAT THIS PROVES
----------------
The GPU driver (tools/seg_token_compressor.py) has two loops that MUST be exact
mirrors: `encode_stream` writes symbols from ground-truth tokens, `decode_stream`
reads them back from decoded tokens. If they disagree by one position, the
roundtrip silently fails -- and a GPU run takes minutes to hours, so the failure
is expensive to discover.

This test drives BOTH loops through the real schedule/tables/step code with:

  * a stand-in coder  -- the exact pure-Python port of ac32's 32-bit coder that
    tools/audit_ac32_packing.py already validates against nb_encode_count_32;
  * a stand-in model  -- a deterministic pseudo-distribution derived from the
    fed tokens (so the "decoder" can only reproduce it via decoded tokens, the
    same way a real decoder must);
  * a stand-in distribution builder -- plain lists instead of numpy; a second
    section then re-runs the round-trip through the REAL sc.build_distribution
    (numpy+numba) so the F1 promotion path is covered too.

So it verifies: segment plan, activation schedule, per-segment tables, escape
branch, empty-segment handling, and encode/decode symmetry. It does NOT verify
torch numerics or ac32's internal math (covered elsewhere / by GPU run).

Run: python3 tools/test_seg_codec_mirror.py
"""
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import audit_ac32_packing as ap          # noqa: E402  (stand-in coder + cum math)
import seg_codec as sc                   # noqa: E402

V = 64                                   # tiny vocab -> escapes happen often
CFG = dict(top_k=8, floor_frac=1e-6, bigram_lambda=0.99,
           bigram_conf=10.0, trigram_conf=3.0)


# ── stand-ins ────────────────────────────────────────────────────────
def _pyints(cum):
    """numpy int64 CDF -> list of Python ints.

    The pure-Python port does exact big-int arithmetic; feeding it numpy int64
    scalars instead poisons its state and makes `x * TOTAL` raise
    OverflowError ("Python int too large to convert to C long"). Values are
    identical, so this cannot mask a mismatch -- but without it, the real-
    distribution section below aborts for a reason that has nothing to do with
    seg_codec. (Production is unaffected: its coder is the numba one.)
    """
    tolist = getattr(cum, "tolist", None)
    return tolist() if tolist is not None else cum


class StandInEncoder:
    """Exposes ac32's method names over the validated pure-Python port."""

    def __init__(self):
        self._e = ap.Enc()

    def encode_symbol(self, cum, sym):
        self._e.encode(_pyints(cum), int(sym))

    def finish(self):
        return self._e.finish()


class StandInDecoder:
    def __init__(self, bitstr):
        self._d = ap.Dec(bitstr)

    def decode_symbol(self, cum):
        return self._d.decode(_pyints(cum))


def _fake_probs(inp_token, step, salt=0):
    """Deterministic pseudo-distribution. Depends only on what was fed -> the
    decoder can reproduce it only if its fed tokens match the encoder's."""
    rng = random.Random((inp_token * 1_000_003) ^ (step * 7919) ^ (salt * 104_729))
    raw = [rng.random() + 0.02 for _ in range(V)]
    s = sum(raw)
    return [r / s for r in raw]


def fake_next_logits(inp, step):
    return [_fake_probs(t, step) for t in inp]


HOT = list(range(8))          # the tokens this "model" believes in


def degenerate_next_logits(inp, step):
    """LM that is certain about ONE token and flat over the rest.

    This is the regime F1 exists for: the K-th ranked token has a probability of
    ~1.6e-5, while a single cache observation is worth ~1.5e-4, so a cache-only
    key beats the weakest in-alphabet token and gets promoted. (With the peaked
    model above, the K-th token still holds ~3.5e-3 and NOTHING gets promoted --
    that is the blend's real reach, not a bug. Worth knowing before blaming F1
    for a residual escape rate.)
    """
    p = [1.6e-5] * V
    p[0] = 1.0 - sum(p)
    return [list(p) for _ in inp]           # one row per segment


def _peaked_probs(inp_token, step):
    """A model that is confidently WRONG: 90% of its mass sits on 8 tokens.

    Stand-in for the regime where F1 matters -- the LM ranks most of the actual
    data outside top-K, but the cache has seen it. With this model the real-
    distribution section promotes cache-only keys at a high rate instead of the
    handful of times a near-uniform model produces.
    """
    rng = random.Random((inp_token * 2_654_435_761) ^ (step * 40_511) ^ 0x5EED)
    probs = [0.0] * V
    left = 0.9
    for i, t in enumerate(HOT):
        take = left * 0.5 if i < len(HOT) - 1 else left
        probs[t] = take
        left -= take
    spine = [rng.random() + 0.05 for _ in range(V)]
    tot = sum(spine)
    for t in range(V):
        probs[t] += 0.1 * spine[t] / tot
    return probs


def peaked_next_logits(inp, step):
    return [_peaked_probs(t, step) for t in inp]


def fake_next_logits_sabotaged(inp, step, bad_step=5):
    """Same model, but one step is perturbed -> the mirror MUST break."""
    if step == bad_step:
        return [_fake_probs(t, step, salt=99) for t in inp]
    return [_fake_probs(t, step) for t in inp]


def fake_make_dist(probs, tables, s, prev, prev2, cfg, uni_cache):
    K = min(cfg["top_k"], len(probs))
    order = sorted(range(len(probs)), key=lambda t: (-probs[t], t))[:K]
    bi_ctx, bi_tot, tri_ctx, tri_tot = tables.get(s, prev, prev2)
    wB = bi_tot / (bi_tot + cfg["bigram_conf"]) if bi_tot > 0 else 0.0
    wT = tri_tot / (tri_tot + cfg["trigram_conf"]) if tri_tot > 0 else 0.0
    wc = 1.0 - (1.0 - wT) * (1.0 - wB)
    le = 1.0 - (1.0 - cfg["bigram_lambda"]) * wc

    blended = []
    for t in order:
        pb = bi_ctx.get(t, 0) / bi_tot if bi_tot > 0 else 0.0
        pt = tri_ctx.get(t, 0) / tri_tot if tri_tot > 0 else 0.0
        pc = (wB * pb + wT * pt) / wc if wc > 1e-12 else 0.0
        blended.append(le * probs[t] + (1.0 - le) * pc)

    esc = max(1e-12, 1.0 - sum(blended))
    cum_s1 = ap.cum_from_probs(blended + [esc], cfg["floor_frac"])

    def rest_fn():
        chosen = set(order)
        rest = [t for t in range(len(probs)) if t not in chosen]
        key = ("rest", len(rest))
        cum = uni_cache.get(key)
        if cum is None:
            cum = uni_cache[key] = ap.cum_from_probs([1.0 / len(rest)] * len(rest),
                                                     1e-6)
        return rest, cum

    return sc.Dist(order, cum_s1, rest_fn)


def uniform_V_cum(uni_cache):
    key = ("V", V)
    cum = uni_cache.get(key)
    if cum is None:
        cum = uni_cache[key] = ap.cum_from_probs([1.0 / V] * V, 1e-6)
    return cum


def roundtrip(ids, n_segments, next_logits=None, dec_next_logits=None,
              dec_segments=None, corrupt_bit=None, cfg=None, make_dist=None):
    """Encode with `next_logits`; decode with `dec_next_logits` (defaults to the
    same model) and `dec_segments` (defaults to the same plan). The knobs exist
    so the negative controls below can break exactly ONE side."""
    next_logits = next_logits or fake_next_logits
    dec_next_logits = dec_next_logits or next_logits
    dec_segments = dec_segments or n_segments
    cfg = CFG if cfg is None else cfg
    make_dist = fake_make_dist if make_dist is None else make_dist
    n = len(ids)
    enc = StandInEncoder()
    uni_cache = {}
    ucum = uniform_V_cum(uni_cache)
    t_enc = sc.SegTables(n_segments)
    stats_enc = dict(coded=0, escapes=0, uniform=0)
    enc_ret = sc.encode_stream(ids, n_segments, cfg, next_logits, enc, t_enc,
                               ucum, make_dist, uni_cache, stats_enc)
    # seg_codec's encode_stream returns (plan, timings) since the phase timers
    # landed; keep accepting a bare plan so this test cannot silently rot again.
    plan, timings = enc_ret if isinstance(enc_ret, tuple) else (enc_ret, {})
    bitstr, count = enc.finish()
    if corrupt_bit is not None:
        i = corrupt_bit % count
        bitstr = bitstr[:i] + ("1" if bitstr[i] == "0" else "0") + bitstr[i + 1:]

    dec = StandInDecoder(bitstr)
    uni_cache_dec = {}
    t_dec = sc.SegTables(dec_segments)
    stats_dec = dict(coded=0, escapes=0, uniform=0)
    err = None
    try:
        out = sc.decode_stream(n, dec_segments, cfg, dec_next_logits, dec, t_dec,
                               uniform_V_cum(uni_cache_dec), make_dist,
                               uni_cache_dec, stats_dec)
    except Exception as e:                       # a wrong plan may raise instead
        out, err = None, f"{type(e).__name__}: {e}"
    return dict(plan=plan, out=out, bits=count, ok=(out == list(ids)), err=err,
                timings=timings, stats_enc=stats_enc, stats_dec=stats_dec,
                stats=stats_enc)


REAL_CFG = dict(top_k=8, prefilter=16, floor_frac=1e-6, bigram_lambda=0.99,
                bigram_conf=10.0, trigram_conf=7.0)


def real_dist_section():
    """Drive the real `sc.build_distribution` through encode+decode.

    The stand-in distribution above never promotes a cache-only key and never
    touches the numpy/numba CDF path, so a lossless-ness bug in F1 (candidate
    set) or F2 (weights) could only surface in a GPU run. This section closes
    that gap: real builder, real tables, stand-in coder+model, and it counts
    promotions so it cannot silently degrade into "cold cache everywhere".
    """
    global V
    try:
        import numpy as np
        sc._ac32()          # raises when numba/ac32 is unavailable
    except Exception as exc:
        print(f"  [skip] needs numpy+numba ({type(exc).__name__}: {exc})")
        print("         run this on the GPU box for F1/F2 round-trip coverage.")
        return True

    tally = dict(calls=0, promo_events=0, promo_ids=0)

    def real_make_dist(probs, tables, s, prev, prev2, cfg, uni_cache):
        p = np.asarray(probs, dtype=np.float64)
        d = sc.build_distribution(p, tables, s, prev, prev2, cfg, uni_cache)
        tk = np.asarray(d.topk)
        tally["calls"] += 1
        k = min(int(cfg["top_k"]), V)
        lm_top = np.argsort(-p)[:k]
        extra = np.setdiff1d(tk, lm_top)
        if len(extra):
            tally["promo_events"] += 1
            tally["promo_ids"] += int(len(extra))
        return d

    rng = random.Random(4242)
    ok = True
    ids = [rng.randrange(V) for _ in range(120)]
    r = roundtrip(ids, 1, cfg=REAL_CFG, make_dist=real_make_dist)
    esc_a = r["stats"]["escapes"]
    ok &= r["ok"] and esc_a > 0
    print(f"  K=1 n=120 real dist: roundtrip={r['ok']} bits={r['bits']} "
          f"escapes={esc_a}/{r['stats']['coded']}")
    ok &= (r["stats_enc"] == r["stats_dec"])
    print(f"  enc/dec counters agree: {r['stats_enc'] == r['stats_dec']}")

    ids2 = [rng.randrange(V) for _ in range(220)]
    r2 = roundtrip(ids2, 4, cfg=REAL_CFG, make_dist=real_make_dist)
    ok &= r2["ok"] and r2["stats"]["escapes"] > 0
    print(f"  K=4 n=220 real dist: roundtrip={r2['ok']} bits={r2['bits']} "
          f"escapes={r2['stats']['escapes']}/{r2['stats']['coded']}")

    # F1 must actually fire, or this whole section is decorative. The peaked
    # model below makes promotion the common case, not a curiosity.
    ids3 = [rng.choices(range(V), weights=[1.0 / (i + 1) ** 1.15
                                          for i in range(V)])[0]
            for _ in range(240)]
    base_events = tally["promo_events"]
    r4 = roundtrip(ids3, 3, next_logits=peaked_next_logits, cfg=REAL_CFG,
                   make_dist=real_make_dist)
    ok &= r4["ok"]
    print(f"  K=3 n=240 peaked model: roundtrip={r4['ok']} bits={r4['bits']} "
          f"escapes={r4['stats']['escapes']}/{r4['stats']['coded']} "
          f"promos={tally['promo_events'] - base_events}")
    print("       (0 promos here is CORRECT: the K-th token still holds "
          "~3.5e-3, so no cache count can outrank it)")

    # degenerate LM + cold data: promotion is the common case
    ids4 = [(1 + i % 6) for i in range(240)]
    base4 = tally["promo_events"]
    r5 = roundtrip(ids4, 3, next_logits=degenerate_next_logits, cfg=REAL_CFG,
                   make_dist=real_make_dist)
    ok &= r5["ok"]
    prom5 = tally["promo_events"] - base4
    print(f"  K=3 n=240 degenerate LM: roundtrip={r5['ok']} bits={r5['bits']} "
          f"escapes={r5['stats']['escapes']}/{r5['stats']['coded']} "
          f"promos={prom5}")
    ok &= prom5 > 0
    print(f"       promotion branch exercised: {prom5} positions")

    ok &= tally["promo_events"] > 0 and tally["promo_ids"] > 0
    print(f"  F1 promotions fired: {tally['promo_events']} positions "
          f"({tally['promo_ids']} promoted ids over {tally['calls']} calls; "
          f"{tally['promo_events'] - base4} of them under the degenerate LM)")

    # non-vacuity: sabotage ONE side on the real path -> must be caught
    r3 = roundtrip(ids2, 4, cfg=REAL_CFG, make_dist=real_make_dist,
                   dec_next_logits=fake_next_logits_sabotaged)
    ok &= not r3["ok"]
    print(f"  negative control on the real path: caught={not r3['ok']}")
    return ok


def main():
    print("=" * 78)
    print("seg_codec mirror test (stdlib only; stand-in coder/model/distribution)")
    print("=" * 78)
    rng = random.Random(20260916)

    cases = [(1, 40), (1, 1), (2, 41), (3, 40), (5, 37), (6, 4), (8, 8), (4, 200)]
    all_ok = True
    for n_segments, n_tokens in cases:
        ids = [rng.randrange(V) for _ in range(n_tokens)]
        r = roundtrip(ids, n_segments)
        lens = [L for _, L in r["plan"]]
        all_ok &= r["ok"]
        print(f"  K={n_segments:<2d} n={n_tokens:<4d} seg_lens={str(lens):<28s} "
              f"bits={r['bits']:<6d} escapes={r['stats']['escapes']:<4d} "
              f"coded={r['stats']['coded']:<4d} {'OK' if r['ok'] else 'MISMATCH'}")

    # uneven index space: n_tokens not divisible by K, and K > n_tokens (empty segs)
    print()
    print("  -- symmetry of the two stats counters --")
    ids = [rng.randrange(V) for _ in range(300)]
    r = roundtrip(ids, 7)
    same = (r["stats_enc"] == r["stats_dec"])
    all_ok &= r["ok"] and same
    print(f"  K=7 n=300 coded enc={r['stats_enc']['coded']} dec={r['stats_dec']['coded']}"
          f"  escapes enc={r['stats_enc']['escapes']} dec={r['stats_dec']['escapes']}"
          f"  equal={same}  roundtrip={r['ok']}")

    print()
    print("  -- negative controls: each must break the mirror --")
    ids = [rng.randrange(V) for _ in range(200)]

    # (a) decoder model disagrees at one step (stale/incorrect history)
    r_bad = roundtrip(ids, 3, dec_next_logits=fake_next_logits_sabotaged)
    bad_pos = next((i for i, (a, b) in enumerate(zip(ids, r_bad["out"])) if a != b),
                   None)
    all_ok &= (not r_bad["ok"])
    print(f"  (a) decoder model perturbed at step 5 -> caught={not r_bad['ok']}, "
          f"first divergence at position {bad_pos}")

    # (b) one corrupted bit in the stream
    r_bit = roundtrip(ids, 3, corrupt_bit=40)
    all_ok &= (not r_bit["ok"])
    print(f"  (b) single flipped bit             -> caught={not r_bit['ok']}")

    # (c) decoder uses the wrong segment count (plan must be load-bearing)
    r_plan = roundtrip(ids, 3, dec_segments=4)
    all_ok &= (not r_plan["ok"])
    how = "mismatch" if r_plan["err"] is None else f"raised {r_plan['err']}"
    print(f"  (c) decoder K=4 vs encoder K=3     -> caught={not r_plan['ok']} ({how})")

    # and the clean run of the very same configuration must pass
    r_ok = roundtrip(ids, 3)
    all_ok &= r_ok["ok"]
    print(f"  (d) same config, no tampering      -> OK={r_ok['ok']}")

    print()
    print("  -- real distribution builder (F1/F2 path) round-trip --")
    all_ok &= real_dist_section()

    print()
    print("  -- API guards (catch silent drift between seg_codec and this test) --")
    import inspect
    sig_enc = inspect.signature(sc.encode_stream)
    sig_dec = inspect.signature(sc.decode_stream)
    checks = [
        ("encode_stream(ids, n_segments, cfg, next_logits, enc, tables, "
         "uni_V_cum, make_dist, uni_cache, stats)",
         list(sig_enc.parameters) == ["ids", "n_segments", "cfg", "next_logits",
                                      "enc", "tables", "uni_V_cum", "make_dist",
                                      "uni_cache", "stats"]),
        ("decode_stream(n_tokens, n_segments, cfg, next_logits, dec, tables, "
         "uni_V_cum, make_dist, uni_cache, stats)",
         list(sig_dec.parameters) == ["n_tokens", "n_segments", "cfg",
                                      "next_logits", "dec", "tables",
                                      "uni_V_cum", "make_dist", "uni_cache",
                                      "stats"]),
        ("build_distribution(probs, tables, s, prev, prev2, cfg, uni_cache)",
         list(inspect.signature(sc.build_distribution).parameters)
         == ["probs", "tables", "s", "prev", "prev2", "cfg", "uni_cache"]),
        ("seg_plan returns [(start, L), ...]",
         all(len(t) == 2 for t in sc.seg_plan(10, 3))),
    ]
    for label, ok in checks:
        all_ok &= ok
        print(f"  [{'ok  ' if ok else 'FAIL'}] {label}")

    print()
    print("=" * 78)
    print(f"RESULT: {'ALL PASS' if all_ok else 'FAILURES PRESENT'}")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
