"""Regression test: AC32 coder termination + the .zllm bit-packing.

History: commit b0d6579 reported "1 mismatch at the tail = arithmetic coding
known limitation". It was not. The coder's finish() is the canonical
Witten-Neal-Cleary termination (pending += 1, emit one bit, decoder reads zeros
past EOF), and the repo's own 16-bit coder -- same scheme -- gets 798/798 exact
chunks in ensemble/verify_enwik8_full.py. The corruption was in the file path:

    bytes(int(bitstr[i:i+8], 2) for i in range(0, len(bitstr), 8))   # buggy

A final chunk of k < 8 bits lands in the LOW k bits of the last byte, but reading
back (f"{b:08b}" per byte, then truncate to n_bits) expects them high-first, so
the tail returns shifted: "1010101" -> 0b01010101 -> "0101010". Invisible when
n_bits % 8 == 0; 95757 % 8 == 7, which is why exactly one token at the tail was
wrong. Fix (in tools/real_compressor.py and tools/seg_token_compressor.py):
right-pad the final chunk before packing.

This file keeps both packings so the bug cannot come back unnoticed. It needs
only the stdlib -- no torch, no GPU -- and is imported by
tools/test_seg_codec_mirror.py for its coder port.

Run: python3 tools/audit_ac32_packing.py
"""
import random

CODE_BITS = 32
TOP = (1 << CODE_BITS) - 1
HALF = 1 << (CODE_BITS - 1)
QUARTER = 1 << (CODE_BITS - 2)
THREE_QUARTER = HALF + QUARTER
TOTAL = 1 << 20
MAX_RENORM_ITERS = 10000


class Enc:
    def __init__(self):
        self.low = self.high = 0
        self.high = TOP
        self.pending = self.count = 0
        self._buf = bytearray()
        self._cur = self._ncur = 0

    def _emit(self, bit):
        self.count += 1 + self.pending
        for b in [bit] + [1 - bit] * self.pending:
            self._cur = (self._cur << 1) | b
            self._ncur += 1
            if self._ncur == 8:
                self._buf.append(self._cur)
                self._cur = self._ncur = 0
        self.pending = 0

    def encode(self, cum, symbol):
        assert cum[symbol + 1] - cum[symbol] > 0, "zero-frequency symbol"
        sym_low, sym_high = cum[symbol], cum[symbol + 1]
        rng = self.high - self.low + 1
        self.high = self.low + (rng * sym_high) // TOTAL - 1
        self.low = self.low + (rng * sym_low) // TOTAL
        while True:
            if self.high < HALF:
                self._emit(0)
                self.low <<= 1
                self.high = (self.high << 1) + 1
            elif self.low >= HALF:
                self._emit(1)
                self.low = (self.low - HALF) << 1
                self.high = ((self.high - HALF) << 1) + 1
            elif self.low >= QUARTER and self.high < THREE_QUARTER:
                self.pending += 1
                self.low = (self.low - QUARTER) << 1
                self.high = ((self.high - QUARTER) << 1) + 1
            else:
                break

    def finish(self):
        self.pending += 1
        self._emit(0 if self.low < QUARTER else 1)
        if self._ncur:
            self._buf.append(self._cur << (8 - self._ncur))
        return "".join(f"{b:08b}" for b in self._buf)[: self.count], self.count


class Dec:
    def __init__(self, bitstr):
        self.bits, self.pos = bitstr, 0
        self.low, self.high, self.code = 0, TOP, 0
        for _ in range(CODE_BITS):
            self.code = (self.code << 1) | self._read()

    def _read(self):
        if self.pos < len(self.bits):
            b = int(self.bits[self.pos])
            self.pos += 1
            return b
        return 0                                     # zero-pad past EOF

    def decode(self, cum):
        rng = self.high - self.low + 1
        value = ((self.code - self.low + 1) * TOTAL - 1) // rng
        lo, hi = 0, len(cum) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if cum[mid] <= value:
                lo = mid
            else:
                hi = mid
        sym = lo
        assert cum[sym + 1] - cum[sym] > 0
        sym_low, sym_high = cum[sym], cum[sym + 1]
        self.high = self.low + (rng * sym_high) // TOTAL - 1
        self.low = self.low + (rng * sym_low) // TOTAL
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.code -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTER:
                self.low -= QUARTER
                self.high -= QUARTER
                self.code -= QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) + 1
            self.code = (self.code << 1) | self._read()
        return sym


def pack_buggy(bitstr, count):
    """The pre-fix packing (kept as the negative case)."""
    bit_bytes = bytes(int(bitstr[i:i + 8], 2) for i in range(0, len(bitstr), 8))
    return "".join(f"{b:08b}" for b in bit_bytes)[:count]


def pack_fixed(bitstr, count):
    """Current packing (tools/seg_token_compressor.py: pack_bits)."""
    chunks = [bitstr[i:i + 8] for i in range(0, len(bitstr), 8)]
    bit_bytes = bytes(int(c.ljust(8, "0"), 2) for c in chunks)
    return "".join(f"{b:08b}" for b in bit_bytes)[:count]


def cum_from_probs(probs, floor_frac=1e-6):
    """Mirror ac32.probs_to_freqs_32."""
    s = sum(probs)
    probs = [p / s for p in probs]
    V = len(probs)
    F = max(1, int(round(floor_frac * TOTAL)))
    freqs = [int(p * (TOTAL - V * F)) + F for p in probs]
    d = TOTAL - sum(freqs)
    if d:
        i = max(range(V), key=lambda k: probs[k])
        freqs[i] += d
    cum, t = [0], 0
    for f in freqs:
        t += f
        cum.append(t)
    assert t == TOTAL
    return cum


def stage1_like(K, escape):
    return cum_from_probs([1.0 / (i + 1) ** 1.2 for i in range(K)] + [escape])


def run(cums, syms, pack):
    enc = Enc()
    for c, s in zip(cums, syms):
        enc.encode(c, s)
    bitstr, count = enc.finish()
    assert len(bitstr) == count, "finish() must return exactly `count` bits"
    seen = pack(bitstr, count)
    d = Dec(seen)
    out = [d.decode(c) for c in cums]
    return [i for i, (a, b) in enumerate(zip(out, syms)) if a != b], count, bitstr == seen


def main():
    print("=" * 78)
    print("PART 1 — the packing, in isolation (k = bits in the final chunk)")
    print("=" * 78)
    for k in range(1, 9):
        bits = "1" * (k - 1) + "1" if k > 1 else "1"
        count = len(bits)
        ok_b = pack_buggy(bits, count) == bits
        ok_f = pack_fixed(bits, count) == bits
        print(f"  tail={bits!r:12s} (k={k})  buggy replication exact? {str(ok_b):5s}"
              f"   fixed exact? {ok_f}")

    print()
    print("=" * 78)
    print("PART 2 — realistic 100KB-shaped stream (30791 symbols, K=1024,")
    print("         ~0.6% escapes over a 48128-symbol uniform stage-2)")
    print("=" * 78)
    rng = random.Random(7)
    K, V, N = 1024, 49152, 30791
    n_esc = int(N * 0.006)
    order = [0] * (N - n_esc) + [1] * n_esc
    rng.shuffle(order)
    uni_n = V - K
    q, r = divmod(TOTAL, uni_n)
    uni = cum_from_probs([q + (1 if i < r else 0) for i in range(uni_n)])
    pool = [stage1_like(K, e) for e in (1e-4, 1e-3, 1e-2)]
    cums, syms = [], []
    for is_esc in order:
        if is_esc:
            cums.append(uni)
            syms.append(rng.randrange(uni_n))
        else:
            cums.append(pool[rng.randrange(3)])
            syms.append(rng.randrange(K))

    for name, pack in (("buggy (real_compressor.py)", pack_buggy),
                       ("fixed (ljust)", pack_fixed)):
        bad, count, stream_ok = run(cums, syms, pack)
        last = "  <-- ALL IN THE LAST 1 SYMBOL" if bad == [N - 1] else ""
        print(f"  {name:28s} bits={count}  stream-exact={str(stream_ok):5s}  "
              f"mismatches={len(bad)} at {bad[:4]}{last}")

    print()
    print("=" * 78)
    print("PART 3 — 20 independent long streams, fixed packing only")
    print("=" * 78)
    worst = 0
    for trial in range(20):
        rng = random.Random(1000 + trial)
        N = rng.randint(5000, 30791)
        K = rng.choice([1024, 2048])
        cums, syms = [], []
        for _ in range(N):
            if rng.random() < 0.006:
                nn = 48128
                q, r = divmod(TOTAL, nn)
                cums.append(cum_from_probs([q + (1 if i < r else 0) for i in range(nn)]))
                syms.append(rng.randrange(nn))
            else:
                cums.append(stage1_like(K, rng.choice([1e-4, 1e-3, 1e-2])))
                syms.append(rng.randrange(K))
        bad, _, _ = run(cums, syms, pack_fixed)
        worst = max(worst, len(bad))
        if bad:
            print(f"  trial{trial}: {len(bad)} mismatches {bad[:5]}")
    print(f"  worst case over 20 streams: {worst} mismatches")

    print()
    print("=" * 78)
    print("PART 4 — 2000 adversarial short streams (extreme distributions)")
    print("=" * 78)
    fails = 0
    for trial in range(2000):
        rng = random.Random(trial)
        p = rng.choice([1e-7, 1e-6, 1e-5, 0.5, 0.999999, 1 - 1e-6])
        cum = cum_from_probs([p, 1 - p])
        n = rng.randint(1, 3000)
        cums, syms = [cum] * n, [rng.randrange(2) for _ in range(n)]
        bad, _, _ = run(cums, syms, pack_fixed)
        if bad:
            fails += 1
            print(f"  trial{trial}: p={p} n={n} mismatches={len(bad)} {bad[:5]}")
            if fails > 3:
                break
    print(f"  streams with any mismatch (fixed packing): {fails}/2000")


if __name__ == "__main__":
    main()
