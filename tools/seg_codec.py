"""Segment-parallel per-token codec core: schedule + tables + mirror steps.

WHY THIS EXISTS
---------------
`tools/real_compressor.py` is not decodable from the file alone. Its decoder
calls the model on the ground-truth token chunk:

    ctx_chunk = ids[ci:ci + chunk_len + 1]        # source tokens, not decoded

so the LM half of the coding distribution comes from the answer sheet; only the
n-gram cache half is rebuilt from decoded tokens.

A causal LM emits one next-token distribution per input position, so a block can
only have *block-internal* context if the block's own tokens are fed back into
the model -- i.e. one forward per token. That is the only self-contained route.
This module makes it affordable: split the token stream into K segments, run
them in lockstep as a batch of K, and keep every piece of state per segment
(its own KV cache, its own n-gram tables). The decoder then reproduces
everything from the bitstream alone.

WHAT'S HERE
-----------
    seg_plan(n_tokens, K)     deterministic boundaries (both sides compute it)
    SegTables                 per-segment bigram/trigram counts
    Dist                      what encode/decode exchange (topk, cum, escape)
    build_distribution(...)   numpy: top-K + cache blend + stage-1 CDF + escape
    encode_stream(...)        drive the lockstep loop, write symbols
    decode_stream(...)        mirror of encode_stream, read symbols

`encode_stream`/`decode_stream` take BOTH the model (`next_logits`) and the
distribution builder (`make_dist`) as callables and are numpy-free, so
`tools/test_seg_codec_mirror.py` can verify the mirror with stand-ins and no
torch/numpy installed. numpy and ac32 are therefore imported lazily.

TOKEN ACCOUNTING
----------------
Every segment codes its own token 0 with uniform(V) (~16 bits each, K of them),
then codes tokens 1..L-1 in the lockstep loop. `n_tokens` counts all tokens in
the file, so the archive is self-describing given the model: there is no seed
token in the header.
"""
DUMMY_TOKEN = 0          # fed to segments that already finished (output ignored)
TOTAL_SYM_MARGIN = 2     # stage-1 alphabet is len(topk)+1 (topk + escape)


def seg_plan(n_tokens, n_segments):
    """Deterministic boundaries: equal split, remainder to the earliest segments.

    Both encoder and decoder derive this from (n_tokens, n_segments) alone, so
    the plan never needs to be stored. Segments may be empty (L == 0) when
    n_segments > n_tokens.
    """
    assert n_segments >= 1
    base, rem = divmod(int(n_tokens), int(n_segments))
    plan, pos = [], 0
    for s in range(n_segments):
        L = base + (1 if s < rem else 0)
        plan.append((pos, L))
        pos += L
    assert pos == n_tokens, (pos, n_tokens)
    return plan


class SegTables:
    """Per-segment bigram/trigram counts.

    Segments are coded independently, so each keeps its own tables; the decoder
    rebuilds segment s's tables from segment s's own tokens. (Sharing one table
    across segments would make a segment depend on other segments' tokens, which
    a lockstep decoder cannot reproduce.)
    """

    def __init__(self, n_segments, use_trigram=True):
        self.use_trigram = use_trigram
        self.bi = [dict() for _ in range(n_segments)]
        self.bt = [dict() for _ in range(n_segments)]
        self.tri = [dict() for _ in range(n_segments)]
        self.tt = [dict() for _ in range(n_segments)]

    def get(self, s, prev, prev2):
        bi_ctx = self.bi[s].get(prev, {})
        bi_tot = self.bt[s].get(prev, 0)
        if self.use_trigram and prev2 is not None:
            key = (prev2, prev)
            tri_ctx = self.tri[s].get(key, {})
            tri_tot = self.tt[s].get(key, 0)
        else:
            tri_ctx, tri_tot = {}, 0
        return bi_ctx, bi_tot, tri_ctx, tri_tot

    def update(self, s, prev, prev2, token):
        d = self.bi[s].get(prev)
        if d is None:
            d = self.bi[s][prev] = {}
        d[token] = d.get(token, 0) + 1
        self.bt[s][prev] = self.bt[s].get(prev, 0) + 1
        if self.use_trigram and prev2 is not None:
            key = (prev2, prev)
            d = self.tri[s].get(key)
            if d is None:
                d = self.tri[s][key] = {}
            d[token] = d.get(token, 0) + 1
            self.tt[s][key] = self.tt[s].get(key, 0) + 1


class Dist:
    """One coded position's distributions.

    topk      : token ids in descending probability (sequence, len <= TOP_K)
    cum_s1    : cumulative freqs over [topk..., escape]; escape index == len(topk)
    rest_fn   : () -> (rest_ids, cum_rest) for the escape branch, or None
    """

    __slots__ = ("topk", "cum_s1", "rest_fn")

    def __init__(self, topk, cum_s1, rest_fn=None):
        self.topk = topk
        self.cum_s1 = cum_s1
        self.rest_fn = rest_fn


def _rank_of(seq, value):
    """index of `value` in seq, or -1. Works for lists, tuples and numpy arrays."""
    tolist = getattr(seq, "tolist", None)
    if tolist is not None:
        seq = tolist()
    try:
        return seq.index(value)
    except ValueError:
        return -1


def _np():
    import numpy as np
    return np


def _ac32():
    from ac32 import stage1_cum_32, uniform_cum_32
    return stage1_cum_32, uniform_cum_32


def uniform_cum_cached(n, cache):
    """uniform_cum_32(n) depends only on n -> safe to memoise on both sides."""
    c = cache.get(n)
    if c is None:
        _, uniform_cum_32 = _ac32()
        c = cache[n] = uniform_cum_32(int(n))
    return c


def build_distribution(probs, tables, s, prev, prev2, cfg, uni_cache):
    """top-K + bigram/trigram blend + 32-bit stage-1 CDF. Mirrors
    tools/real_compressor.py's math so the numbers stay comparable."""
    np = _np()
    stage1_cum_32, _ = _ac32()
    V = len(probs)
    top_k = min(int(cfg["top_k"]), V)
    tk = np.argpartition(probs, -top_k)[-top_k:]
    tk = tk[np.argsort(-probs[tk])]

    bi_ctx, bi_tot, tri_ctx, tri_tot = tables.get(s, prev, prev2)
    wB = bi_tot / (bi_tot + cfg["bigram_conf"]) if bi_tot > 0 else 0.0
    wT = tri_tot / (tri_tot + cfg["trigram_conf"]) if tri_tot > 0 else 0.0
    wc = 1.0 - (1.0 - wT) * (1.0 - wB)
    le = 1.0 - (1.0 - cfg["bigram_lambda"]) * wc

    p_blend = np.zeros(top_k, dtype=np.float64)
    for i in range(top_k):
        t = int(tk[i])
        pb = bi_ctx.get(t, 0) / bi_tot if bi_tot > 0 else 0.0
        pt = tri_ctx.get(t, 0) / tri_tot if tri_tot > 0 else 0.0
        pc = (wB * pb + wT * pt) / wc if wc > 1e-12 else 0.0
        p_blend[i] = le * float(probs[t]) + (1.0 - le) * pc

    esc = max(1e-12, 1.0 - float(p_blend.sum()))
    cum_s1 = stage1_cum_32(p_blend, esc, cfg["floor_frac"])

    def rest_fn():
        mask = np.ones(V, dtype=bool)
        mask[tk] = False
        rest_ids = np.arange(V, dtype=np.int64)[mask]
        return rest_ids, uniform_cum_cached(len(rest_ids), uni_cache)

    return Dist(tk, cum_s1, rest_fn)


def encode_stream(ids, n_segments, cfg, next_logits, enc, tables,
                  uni_V_cum, make_dist, uni_cache, stats=None):
    """Code every token of `ids` into `enc`. Returns the segment plan.

    next_logits(inp_tokens, step) -> list of K probability vectors, index-aligned
    with the plan. `inp_tokens[s]` is the last token of segment s (DUMMY_TOKEN
    once that segment is exhausted).
    """
    n = len(ids)
    plan = seg_plan(n, n_segments)

    # 1) seed: token 0 of each non-empty segment, uniform(V), in segment order.
    for (start, L) in plan:
        if L == 0:
            continue
        enc.encode_symbol(uni_V_cum, int(ids[start]))
        if stats is not None:
            stats["coded"] += 1
            stats["uniform"] += 1

    # 2) lockstep: step i feeds token i, yields the distribution for token i+1.
    maxL = max((L for _, L in plan), default=0)
    for i in range(0, maxL - 1):
        inp = [int(ids[start + i]) if i < L else DUMMY_TOKEN for (start, L) in plan]
        probs = next_logits(inp, i)
        for s, (start, L) in enumerate(plan):
            if i + 1 >= L:
                continue
            prev = int(ids[start + i])
            prev2 = int(ids[start + i - 1]) if i >= 1 else None
            target = int(ids[start + i + 1])
            dist = make_dist(probs[s], tables, s, prev, prev2, cfg, uni_cache)
            r = _rank_of(dist.topk, target)
            if r >= 0:
                enc.encode_symbol(dist.cum_s1, r)
            else:
                enc.encode_symbol(dist.cum_s1, len(dist.topk))
                rest_ids, cum_rest = dist.rest_fn()
                enc.encode_symbol(cum_rest, _rank_of(rest_ids, target))
                if stats is not None:
                    stats["escapes"] += 1
            tables.update(s, prev, prev2, target)
            if stats is not None:
                stats["coded"] += 1
    return plan


def decode_stream(n_tokens, n_segments, cfg, next_logits, dec, tables,
                  uni_V_cum, make_dist, uni_cache, stats=None):
    """Mirror of encode_stream: rebuild all `n_tokens` tokens from `dec`."""
    plan = seg_plan(n_tokens, n_segments)
    buf = [[] for _ in plan]

    for s, (start, L) in enumerate(plan):
        if L == 0:
            continue
        buf[s].append(int(dec.decode_symbol(uni_V_cum)))
        if stats is not None:
            stats["coded"] += 1
            stats["uniform"] += 1

    maxL = max((L for _, L in plan), default=0)
    for i in range(0, maxL - 1):
        inp = [buf[s][i] if i < L else DUMMY_TOKEN
               for s, (_, L) in enumerate(plan)]
        probs = next_logits(inp, i)
        for s, (start, L) in enumerate(plan):
            if i + 1 >= L:
                continue
            prev = buf[s][i]
            prev2 = buf[s][i - 1] if i >= 1 else None
            dist = make_dist(probs[s], tables, s, prev, prev2, cfg, uni_cache)
            sym = dec.decode_symbol(dist.cum_s1)
            if sym < len(dist.topk):
                tok = int(dist.topk[sym])
            else:
                rest_ids, cum_rest = dist.rest_fn()
                tok = int(rest_ids[dec.decode_symbol(cum_rest)])
                if stats is not None:
                    stats["escapes"] += 1
            buf[s].append(tok)
            tables.update(s, prev, prev2, tok)
            if stats is not None:
                stats["coded"] += 1

    out = []
    for s, (_, L) in enumerate(plan):
        assert len(buf[s]) == L, (s, len(buf[s]), L)
        out.extend(buf[s])
    return out
