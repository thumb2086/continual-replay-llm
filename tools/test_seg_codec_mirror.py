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
  * a stand-in distribution builder -- plain lists instead of numpy.

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
class StandInEncoder:
    """Exposes ac32's method names over the validated pure-Python port."""

    def __init__(self):
        self._e = ap.Enc()

    def encode_symbol(self, cum, sym):
        self._e.encode(cum, int(sym))

    def finish(self):
        return self._e.finish()


class StandInDecoder:
    def __init__(self, bitstr):
        self._d = ap.Dec(bitstr)

    def decode_symbol(self, cum):
        return self._d.decode(cum)


def _fake_probs(inp_token, step, salt=0):
    """Deterministic pseudo-distribution. Depends only on what was fed -> the
    decoder can reproduce it only if its fed tokens match the encoder's."""
    rng = random.Random((inp_token * 1_000_003) ^ (step * 7919) ^ (salt * 104_729))
    raw = [rng.random() + 0.02 for _ in range(V)]
    s = sum(raw)
    return [r / s for r in raw]


def fake_next_logits(inp, step):
    return [_fake_probs(t, step) for t in inp]


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
              dec_segments=None, corrupt_bit=None):
    """Encode with `next_logits`; decode with `dec_next_logits` (defaults to the
    same model) and `dec_segments` (defaults to the same plan). The knobs exist
    so the negative controls below can break exactly ONE side."""
    next_logits = next_logits or fake_next_logits
    dec_next_logits = dec_next_logits or next_logits
    dec_segments = dec_segments or n_segments
    n = len(ids)
    enc = StandInEncoder()
    uni_cache = {}
    ucum = uniform_V_cum(uni_cache)
    t_enc = sc.SegTables(n_segments)
    stats_enc = dict(coded=0, escapes=0, uniform=0)
    plan = sc.encode_stream(ids, n_segments, CFG, next_logits, enc, t_enc,
                            ucum, fake_make_dist, uni_cache, stats_enc)
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
        out = sc.decode_stream(n, dec_segments, CFG, dec_next_logits, dec, t_dec,
                               uniform_V_cum(uni_cache_dec), fake_make_dist,
                               uni_cache_dec, stats_dec)
    except Exception as e:                       # a wrong plan may raise instead
        out, err = None, f"{type(e).__name__}: {e}"
    return dict(plan=plan, out=out, bits=count, ok=(out == list(ids)), err=err,
                stats_enc=stats_enc, stats_dec=stats_dec, stats=stats_enc)


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
    print("=" * 78)
    print(f"RESULT: {'ALL PASS' if all_ok else 'FAILURES PRESENT'}")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
