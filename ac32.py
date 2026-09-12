"""32-bit arithmetic coder (companion to real_compression.py's 16-bit core).

Why: the 16-bit core quantizes every distribution to TOTAL=2**14=16384.
With a 2049-symbol stage-1 alphabet each symbol eats >=1 count, so tail
probs < 6e-5 are rounded UP (mass stolen from the head). Worse, a
full-vocab (49K) stage-2 alphabet is IMPOSSIBLE at 14 bits (49K > 16K).

This module mirrors the 16-bit math with 32-bit code values and
TOTAL32 = 2**20 = 1048576:
  - min post-renorm range scales with code bits (2**30ish for 32-bit),
    so TOTAL32 is safe with a ~1000x margin.
  - rng * sym <= 2**32 * 2**20 = 2**52 < 2**64: uint64 kernel is exact.
  - 49K-symbol cums need >=1 count each: 49K < 1M. Fits.

Exports:
  stage1_cum_32, uniform_cum_32, cache_rest_cum_32,
  nb_encode_count_32, ArithmeticEncoder32, ArithmeticDecoder32.
"""
import numpy as np
import numba

CODE_BITS32 = 32
TOP32 = (1 << CODE_BITS32) - 1
HALF32 = 1 << (CODE_BITS32 - 1)
QUARTER32 = 1 << (CODE_BITS32 - 2)
THREE_QUARTER32 = HALF32 + QUARTER32

TOTAL32 = 1 << 20

MAX_RENORM_ITERS = 10000


def probs_to_freqs_32(probs, floor_frac=6.1035e-5):
    """Quantize to int64 freqs summing to TOTAL32.

    floor_frac: smoothing mass per symbol as a fraction of TOTAL32.
      floor_frac = 1/16384 reproduces the 16-bit core's +1-count floor
      (parity mode). SMALLER floors (1e-5..3e-5) sharpen the head --
      this, not raw precision, is what TOTAL32 buys: at 14 bits the
      floor cannot go below 1 count = 6.1e-5.
    """
    probs = np.asarray(probs, dtype=np.float64)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / s
    V = len(probs)
    F = max(1, int(round(floor_frac * TOTAL32)))
    if V * F >= TOTAL32:
        raise ValueError(f"alphabet {V} x floor {F} exceeds TOTAL32")
    freqs = (probs * (TOTAL32 - V * F)).astype(np.int64) + F
    diff = int(TOTAL32 - freqs.sum())
    if diff != 0:
        idx = int(np.argmax(probs))
        freqs[idx] += diff
        if freqs[idx] < F:
            freqs[:] = TOTAL32 // V
            freqs[: TOTAL32 % V] += 1
    assert freqs.sum() == TOTAL32 and (freqs >= 1).all()
    cum = np.zeros(len(freqs) + 1, dtype=np.int64)
    np.cumsum(freqs, out=cum[1:])
    return freqs, cum


def stage1_cum_32(topk_probs, escape_mass, floor_frac=6.1035e-5):
    p = np.concatenate([np.asarray(topk_probs, dtype=np.float64),
                        [float(escape_mass)]])
    return probs_to_freqs_32(p, floor_frac)[1]


def uniform_cum_32(n, floor_frac=2.0e-5):
    # Big-n uniform needs a small floor (n x 64 counts > TOTAL32).
    p = np.ones(n, dtype=np.float64) / n
    return probs_to_freqs_32(p, floor_frac)[1]


def cache_rest_cum_32(cache_full, rest_ids, floor_frac=2.2e-5):
    """Stage-2 distribution over rest[] informed by shared cache state.

    q propto cache mass + uniform floor. floor_frac ~ 1/47K reproduces
    uniform-like cost where the cache is flat, and concentrates where
    the cache has evidence. Both sides hold identical tables, so this
    is exactly reproducible by the decoder.
    """
    c = np.asarray(cache_full, dtype=np.float64)[np.asarray(rest_ids)]
    s = c.sum()
    if not np.isfinite(s) or s <= 0:
        return uniform_cum_32(len(rest_ids))
    V = len(rest_ids)
    # Auto-clamp: V x F must fit TOTAL32 (e.g. 48K rest x F=23 overflows).
    # Deterministic both sides (same inputs -> same F), so sync is safe.
    F = max(1, int(round(floor_frac * TOTAL32)))
    if V * F >= TOTAL32:
        F = max(1, TOTAL32 // V // 2)
    return probs_to_freqs_32(c / s, F / TOTAL32)[1]


@numba.njit(cache=True)
def nb_blend_row(p_row, pre_idx, brow_keys, brow_vals, trow_keys, trow_vals,
                 sb, st, lam, TOP_K, seen_pre, seen_row, work, tag,
                 heap_s, heap_i, out_idx, out_p, target, out_rank):
    """One position of LLM/cache blend + top-K, fully in numba.

    score(t) = lam * p_row[t] + (1-lam) * pb(t), pb from scattered rows.
    seen_pre/seen_row: int32 stamp arrays (V,), work: float64 (V,).
    heap_s/heap_i/out_idx/out_p: scratch of length TOP_K (reused).
    target/out_rank: rank scan fused in (saves a second kernel dispatch
    per position -- dispatch, not heap work, dominates the loop).
    Returns n (== min(TOP_K, n_cand)). out_*[:n] hold ids/probs DESC.
    Tie order vs numpy argsort may differ (measure-zero) -- validated as
    ~1e-4 bpb noise, not bit-exactness (lossless-ness is unaffected:
    encoder/decoder use the same path).
    """
    one = 1.0 - lam
    # 1. scatter cache rows into work[]
    nb = brow_keys.shape[0]
    for j in range(nb):
        key = brow_keys[j]
        work[key] = sb * brow_vals[j]
        seen_row[key] = tag
    nt = trow_keys.shape[0]
    for j in range(nt):
        key = trow_keys[j]
        if seen_row[key] == tag:
            work[key] += st * trow_vals[j]
        else:
            work[key] = st * trow_vals[j]
            seen_row[key] = tag
    # 2. mark prefilter
    np_ = pre_idx.shape[0]
    for j in range(np_):
        seen_pre[pre_idx[j]] = tag
    # 3. min-heap over candidates (inline sift)
    n = 0
    K = TOP_K
    for j in range(np_):
        idx = pre_idx[j]
        pb = work[idx] if seen_row[idx] == tag else 0.0
        s = lam * float(p_row[idx]) + one * pb
        if n < K:
            heap_s[n] = s
            heap_i[n] = idx
            n += 1
            c = n - 1
            while c > 0:
                par = (c - 1) // 2
                if heap_s[c] < heap_s[par]:
                    ts = heap_s[c]
                    ti = heap_i[c]
                    heap_s[c] = heap_s[par]
                    heap_i[c] = heap_i[par]
                    heap_s[par] = ts
                    heap_i[par] = ti
                    c = par
                else:
                    break
        elif s > heap_s[0]:
            heap_s[0] = s
            heap_i[0] = idx
            i = 0
            while True:
                l = 2 * i + 1
                r = l + 1
                m = i
                if l < n and heap_s[l] < heap_s[m]:
                    m = l
                if r < n and heap_s[r] < heap_s[m]:
                    m = r
                if m == i:
                    break
                ts = heap_s[i]
                ti = heap_i[i]
                heap_s[i] = heap_s[m]
                heap_i[i] = heap_i[m]
                heap_s[m] = ts
                heap_i[m] = ti
                i = m
    # 4. row-exclusive keys (not in prefilter)
    for j in range(nb):
        key = brow_keys[j]
        if seen_pre[key] != tag:
            seen_pre[key] = tag
            s = lam * float(p_row[key]) + one * work[key]
            if n < K:
                heap_s[n] = s
                heap_i[n] = key
                n += 1
                c = n - 1
                while c > 0:
                    par = (c - 1) // 2
                    if heap_s[c] < heap_s[par]:
                        ts = heap_s[c]
                        ti = heap_i[c]
                        heap_s[c] = heap_s[par]
                        heap_i[c] = heap_i[par]
                        heap_s[par] = ts
                        heap_i[par] = ti
                        c = par
                    else:
                        break
            elif s > heap_s[0]:
                heap_s[0] = s
                heap_i[0] = key
                i = 0
                while True:
                    l = 2 * i + 1
                    r = l + 1
                    m = i
                    if l < n and heap_s[l] < heap_s[m]:
                        m = l
                    if r < n and heap_s[r] < heap_s[m]:
                        m = r
                    if m == i:
                        break
                    ts = heap_s[i]
                    ti = heap_i[i]
                    heap_s[i] = heap_s[m]
                    heap_i[i] = heap_i[m]
                    heap_s[m] = ts
                    heap_i[m] = ti
                    i = m
    for j in range(nt):
        key = trow_keys[j]
        if seen_pre[key] != tag:
            seen_pre[key] = tag
            s = lam * float(p_row[key]) + one * work[key]
            if n < K:
                heap_s[n] = s
                heap_i[n] = key
                n += 1
                c = n - 1
                while c > 0:
                    par = (c - 1) // 2
                    if heap_s[c] < heap_s[par]:
                        ts = heap_s[c]
                        ti = heap_i[c]
                        heap_s[c] = heap_s[par]
                        heap_i[c] = heap_i[par]
                        heap_s[par] = ts
                        heap_i[par] = ti
                        c = par
                    else:
                        break
            elif s > heap_s[0]:
                heap_s[0] = s
                heap_i[0] = key
                i = 0
                while True:
                    l = 2 * i + 1
                    r = l + 1
                    m = i
                    if l < n and heap_s[l] < heap_s[m]:
                        m = l
                    if r < n and heap_s[r] < heap_s[m]:
                        m = r
                    if m == i:
                        break
                    ts = heap_s[i]
                    ti = heap_i[i]
                    heap_s[i] = heap_s[m]
                    heap_i[i] = heap_i[m]
                    heap_s[m] = ts
                    heap_i[m] = ti
                    i = m
    # 5. heapsort ascending, then emit descending
    end = n - 1
    while end > 0:
        ts = heap_s[0]
        ti = heap_i[0]
        heap_s[0] = heap_s[end]
        heap_i[0] = heap_i[end]
        heap_s[end] = ts
        heap_i[end] = ti
        i = 0
        while True:
            l = 2 * i + 1
            r = l + 1
            m = i
            if l < end and heap_s[l] < heap_s[m]:
                m = l
            if r < end and heap_s[r] < heap_s[m]:
                m = r
            if m == i:
                break
            ts = heap_s[i]
            ti = heap_i[i]
            heap_s[i] = heap_s[m]
            heap_i[i] = heap_i[m]
            heap_s[m] = ts
            heap_i[m] = ti
            i = m
        end -= 1
    for i in range(n):
        out_p[i] = heap_s[n - 1 - i]
        out_idx[i] = heap_i[n - 1 - i]
    rk = -1
    for i in range(n):
        if out_idx[i] == target:
            rk = i
            break
    out_rank[0] = rk
    return n


@numba.njit(cache=True)
def nb_find_rank(sorted_idx, n, target):
    """Linear rank scan over desc top-K ids. Returns -1 if absent."""
    for j in range(n):
        if sorted_idx[j] == target:
            return j
    return -1


@numba.njit(cache=True)
def nb_batch_cum_32(topk_mat, esc_vec, F, out):
    """Vectorized stage-1 quantization over N rows (GIL-free).

    Replicates probs_to_freqs_32 per row EXACTLY (normalize, scale by
    TOTAL32-(K+1)*F, +F floor, diff-to-argmax fix, <F fallback), with the
    escape mass as the last symbol. topk_mat: [N,K] float64,
    esc_vec: [N] float64 (already floored by caller), F: floor counts,
    out: [N,K+1] int64 cums (row-sum TOTAL32).
    """
    N = topk_mat.shape[0]
    K = topk_mat.shape[1]
    K1 = K + 1
    TT = np.int64(TOTAL32)
    base = TT - np.int64(K1) * np.int64(F)
    for r in range(N):
        s = esc_vec[r]
        for j in range(K):
            s += topk_mat[r, j]
        inv = 1.0 / s
        tot = np.int64(0)
        amax = 0
        amaxq = -1.0
        for j in range(K):
            q = topk_mat[r, j] * inv
            if q > amaxq:
                amaxq = q
                amax = j
            fq = np.int64(q * float(base)) + np.int64(F)
            out[r, j + 1] = fq
            tot += fq
        # escape symbol lives at index K -> cum col K+1 (NOT K: that is the
        # last topk freq -- overwriting it caused zero-freq encoder errors)
        qe = esc_vec[r] * inv
        if qe > amaxq:
            amax = K
        fqe = np.int64(qe * float(base)) + np.int64(F)
        out[r, K + 1] = fqe
        tot += fqe
        diff = TT - tot
        if diff != np.int64(0):
            out[r, amax + 1] += diff
            # Mirror probs_to_freqs_32 exactly (fallback below F, not
            # below 1). Dead on 100KB runs (diff >= 0 always there), but
            # exact-mirror kills a whole class of doubt on full files.
            if out[r, amax + 1] < np.int64(1):
                per = TT // np.int64(K1)
                rem = TT % np.int64(K1)
                for j in range(K1):
                    out[r, j + 1] = per + (np.int64(1) if j < rem else np.int64(0))
        out[r, 0] = np.int64(0)
        for j in range(K1):
            out[r, j + 1] += out[r, j]


@numba.njit(cache=True)
def nb_encode_count_32(cums, symbols):
    """Count bits with 32-bit code values. cums: (N, A+1) int64 with
    row-sum TOTAL32. Returns total incl. finish bits, negative on error."""
    low = np.uint64(0)
    high = np.uint64(np.uint64(TOP32))
    pending = np.uint64(0)
    count = np.uint64(0)
    n = symbols.shape[0]
    H = np.uint64(HALF32)
    Q = np.uint64(QUARTER32)
    TQ = np.uint64(THREE_QUARTER32)
    TT = np.uint64(TOTAL32)
    for t in range(n):
        s = np.int64(symbols[t])
        sym_low = np.uint64(np.int64(cums[t, s]))
        sym_high = np.uint64(np.int64(cums[t, s + 1]))
        if sym_high <= sym_low:
            return np.int64(-1)
        if not (np.uint64(0) <= low and low <= high):
            return np.int64(-2)
        rng = high - low + np.uint64(1)
        high = low + (rng * sym_high) // TT - np.uint64(1)
        low = low + (rng * sym_low) // TT
        it = np.int64(0)
        while True:
            it += 1
            if it > 10000:
                return np.int64(-3)
            if high < H:
                count += np.uint64(1) + pending
                pending = np.uint64(0)
                low <<= np.uint64(1)
                high = (high << np.uint64(1)) + np.uint64(1)
            elif low >= H:
                count += np.uint64(1) + pending
                pending = np.uint64(0)
                low = (low - H) << np.uint64(1)
                high = ((high - H) << np.uint64(1)) + np.uint64(1)
            elif low >= Q and high < TQ:
                pending += np.uint64(1)
                if pending > np.uint64(10000):
                    return np.int64(-4)
                low = (low - Q) << np.uint64(1)
                high = ((high - Q) << np.uint64(1)) + np.uint64(1)
            else:
                break
    count += np.uint64(2) + pending
    return np.int64(count)


class ArithmeticEncoder32:
    def __init__(self, store=False):
        self.low = 0
        self.high = TOP32
        self.pending = 0
        self.count = 0
        self.store = store
        self._buf = bytearray()
        self._cur = 0
        self._ncur = 0

    def _emit(self, bit):
        self.count += 1 + self.pending
        if self.store:
            bits = [bit] + [1 - bit] * self.pending
            for b in bits:
                self._cur = (self._cur << 1) | b
                self._ncur += 1
                if self._ncur == 8:
                    self._buf.append(self._cur)
                    self._cur = 0
                    self._ncur = 0
        self.pending = 0

    def encode_symbol(self, cum, symbol):
        freq = int(cum[symbol + 1]) - int(cum[symbol])
        if freq <= 0:
            raise ArithmeticError(f"zero-frequency symbol {symbol}")
        if not (0 <= self.low <= self.high):
            raise ArithmeticError("invalid range at encode_entry")
        sym_low = int(cum[symbol])
        sym_high = int(cum[symbol + 1])
        rng = self.high - self.low + 1
        self.high = self.low + (rng * sym_high) // TOTAL32 - 1
        self.low = self.low + (rng * sym_low) // TOTAL32
        iters = 0
        while True:
            iters += 1
            if iters > MAX_RENORM_ITERS:
                raise ArithmeticError("renormalization did not converge")
            if self.high < HALF32:
                self._emit(0)
                self.low <<= 1
                self.high = (self.high << 1) + 1
            elif self.low >= HALF32:
                self._emit(1)
                self.low = (self.low - HALF32) << 1
                self.high = ((self.high - HALF32) << 1) + 1
            elif self.low >= QUARTER32 and self.high < THREE_QUARTER32:
                self.pending += 1
                if self.pending > MAX_RENORM_ITERS:
                    raise ArithmeticError("pending underflow overflow")
                self.low = (self.low - QUARTER32) << 1
                self.high = ((self.high - QUARTER32) << 1) + 1
            else:
                break

    def finish(self):
        self.pending += 1
        if self.low < QUARTER32:
            self._emit(0)
        else:
            self._emit(1)
        bitstr = None
        if self.store:
            if self._ncur:
                self._buf.append(self._cur << (8 - self._ncur))
            bitstr = "".join(f"{byte:08b}" for byte in self._buf)[: self.count]
        return bitstr, self.count


class ArithmeticDecoder32:
    def __init__(self, bitstr):
        self.bits = bitstr
        self.pos = 0
        self.low = 0
        self.high = TOP32
        self.code = 0
        for _ in range(CODE_BITS32):
            self.code = (self.code << 1) | self._read_bit()

    def _read_bit(self):
        if self.pos < len(self.bits):
            b = int(self.bits[self.pos])
            self.pos += 1
            return b
        return 0

    def decode_symbol(self, cum):
        if not (0 <= self.low <= self.high):
            raise ArithmeticError("invalid decoder range")
        rng = self.high - self.low + 1
        value = ((self.code - self.low + 1) * TOTAL32 - 1) // rng
        lo, hi = 0, len(cum) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if cum[mid] <= value:
                lo = mid
            else:
                hi = mid
        symbol = lo
        if int(cum[symbol + 1]) - int(cum[symbol]) <= 0:
            raise ArithmeticError(f"zero-frequency symbol {symbol} in decoder")
        sym_low = int(cum[symbol])
        sym_high = int(cum[symbol + 1])
        self.high = self.low + (rng * sym_high) // TOTAL32 - 1
        self.low = self.low + (rng * sym_low) // TOTAL32
        iters = 0
        while True:
            iters += 1
            if iters > MAX_RENORM_ITERS:
                raise ArithmeticError("decoder renormalization did not converge")
            if self.high < HALF32:
                pass
            elif self.low >= HALF32:
                self.low -= HALF32
                self.high -= HALF32
                self.code -= HALF32
            elif self.low >= QUARTER32 and self.high < THREE_QUARTER32:
                self.low -= QUARTER32
                self.high -= QUARTER32
                self.code -= QUARTER32
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) + 1
            self.code = (self.code << 1) | self._read_bit()
        return symbol


if __name__ == "__main__":
    import time
    rng = np.random.default_rng(0)
    # 1. roundtrip random alphabets incl. 49K (impossible at 14 bits).
    # Big alphabet needs a smaller floor (47K x 64 counts > TOTAL32).
    for A, ff in ((3, 6.1035e-5), (2049, 6.1035e-5), (47104, 2.0e-5)):
        p = rng.random(A) + 0.01
        p /= p.sum()
        _, cum = probs_to_freqs_32(p, ff)
        syms = rng.integers(0, A, size=200).astype(np.int64)
        C = np.ascontiguousarray(np.tile(cum, (200, 1)))
        n = int(nb_encode_count_32(C, syms))
        assert n > 0, (A, n)
        # store roundtrip on first 5
        enc = ArithmeticEncoder32(store=True)
        for s in syms[:5]:
            enc.encode_symbol(cum, int(s))
        bs, _ = enc.finish()
        dec = ArithmeticDecoder32(bs)
        got = [dec.decode_symbol(cum) for _ in syms[:5]]
        assert got == [int(s) for s in syms[:5]], (A, got)
        print(f"alphabet {A}: {n} bits/200 syms, roundtrip OK")
    # 2. floor_frac=1/16384 must reproduce 16-bit counts (parity);
    # smaller floors must sharpen skewed heads.
    import sys as _s
    _s.path.insert(0, ".")
    from real_compression import probs_to_freqs, nb_encode_count
    p = np.array([0.9] + [0.1 / 2048] * 2048)
    syms = np.zeros(500, dtype=np.int64)
    syms[::167] = np.arange(3) % 2049 + 1  # 0.6% tail rate = real escapes
    _, cum16 = probs_to_freqs(p)
    C16 = np.ascontiguousarray(np.tile(cum16, (500, 1)))
    n16 = int(nb_encode_count(C16, syms))
    for ff in (6.1035e-5, 3e-5, 1e-5):
        _, cum32 = probs_to_freqs_32(p, ff)
        C32 = np.ascontiguousarray(np.tile(cum32, (500, 1)))
        n32 = int(nb_encode_count_32(C32, syms))
        print(f"skewed floor_frac={ff:g}: 32-bit={n32} vs 16-bit={n16}")
    _, cum_par = probs_to_freqs_32(p, 6.1035e-5)
    n_par = int(nb_encode_count_32(
        np.ascontiguousarray(np.tile(cum_par, (500, 1))), syms))
    print(f"parity check (info only): {n_par} vs 16-bit {n16} "
          f"(diff = rounding residuals at different TOTALs)")
    print("ac32 self-test PASSED")
