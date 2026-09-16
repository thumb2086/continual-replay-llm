"""Audit: does zllm/cli.py's encoder produce a decodable stream?

Standalone re-implementation (pure Python, no numpy/torch) of the 16-bit
coder math from real_compression.py, used exactly the way zllm/cli.py
cmd_encode uses it:

    for each symbol:
        enc = ArithmeticEncoder(store=True)   # FRESH encoder per symbol
        enc.encode_symbol(cum, sym)
        bitstr, count = enc.finish()
        all_bits.extend(int(b) for b in bitstr)   # 1 int per BIT
    ...
    total_bits = len(all_bits) * 8                 # <- line 216-ish of cli.py
    bpb = total_bits / n_bytes

Run:  python3 tools/audit_zllm_stream.py
"""
import math

FREQ_BITS = 14
TOTAL = 1 << FREQ_BITS
CODE_BITS = 16
TOP = (1 << CODE_BITS) - 1
HALF = 1 << (CODE_BITS - 1)
QUARTER = 1 << (CODE_BITS - 2)
THREE_QUARTER = HALF + QUARTER


def probs_to_freqs(probs):
    """real_compression.probs_to_freqs (16-bit core), pure-python."""
    s = sum(probs)
    probs = [p / s for p in probs]
    V = len(probs)
    freqs = [int(p * (TOTAL - V)) + 1 for p in probs]
    diff = TOTAL - sum(freqs)
    am = max(range(V), key=lambda i: probs[i])
    freqs[am] += diff
    cum = [0] * (V + 1)
    for i, f in enumerate(freqs):
        cum[i + 1] = cum[i] + f
    return cum


class Enc:
    def __init__(self):
        self.low, self.high, self.pending, self.count = 0, TOP, 0, 0
        self.buf, self.cur, self.ncur = [], 0, 0

    def _emit(self, bit):
        self.count += 1 + self.pending
        bits = [bit] + [1 - bit] * self.pending
        for b in bits:
            self.cur = (self.cur << 1) | b
            self.ncur += 1
            if self.ncur == 8:
                self.buf.append(self.cur)
                self.cur, self.ncur = 0, 0
        self.pending = 0

    def encode(self, cum, sym):
        sym_low, sym_high = cum[sym], cum[sym + 1]
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
        """Return (bitstring, count) exactly like real_compression.ArithmeticEncoder."""
        self.pending += 1
        self._emit(0 if self.low < QUARTER else 1)
        if self.ncur:
            self.buf.append(self.cur << (8 - self.ncur))
        bitstr = "".join(f"{b:08b}" for b in self.buf)[: self.count]
        return bitstr, self.count


class Dec:
    def __init__(self, bitstr):
        self.bits, self.pos, self.low, self.high, self.code = bitstr, 0, 0, TOP, 0
        for _ in range(CODE_BITS):
            self.code = (self.code << 1) | self._read()

    def _read(self):
        if self.pos < len(self.bits):
            b = int(self.bits[self.pos])
            self.pos += 1
            return b
        return 0

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


def main():
    # A 4096-symbol alphabet whose probabilities look like a top-1024+escape tail
    V = 4096
    probs = [1.0 / (i + 1) ** 1.05 for i in range(V)]
    cum = probs_to_freqs(probs)
    syms = [3, 17, 250, 0, 999, 41]
    truth = -sum(math.log2(probs[s] / sum(probs)) for s in syms)
    print(f"alphabet V={V}, TOTAL={TOTAL}")
    print(f"ideal code length of {syms}: {truth:.2f} bits\n")

    # --- (1) reference: ONE continuous stream (the correct way) ---
    one = Enc()
    for s in syms:
        one.encode(cum, s)
    bstr, cnt = one.finish()
    print(f"[1] single continuous encoder : count={cnt} bits, len(bitstr)={len(bstr)}")
    d = Dec(bstr)
    back = [d.decode(cum) for _ in syms]
    print(f"    decode -> {back}   {'OK' if back == syms else 'MISMATCH'}")

    # --- (2) zllm/cli.py style: FRESH encoder per symbol, concatenate ---
    all_bits = []
    counts = []
    for s in syms:
        e = Enc()
        e.encode(cum, s)
        b, c = e.finish()
        assert len(b) == c, "finish() returns exactly `count` bit characters"
        counts.append(c)
        all_bits.extend(int(ch) for ch in b)
    concat = "".join(str(b) for b in all_bits)
    print(f"\n[2] zllm style (fresh encoder per symbol, concatenated)")
    print(f"    per-symbol counts = {counts}  (sum = {sum(counts)} bits)")
    print(f"    len(all_bits) = {len(all_bits)}  == sum(counts)? "
          f"{len(all_bits) == sum(counts)}")
    print(f"    cli.py reports total_bits = len(all_bits)*8 = {len(all_bits)*8}")
    print(f"    true bit total            = len(all_bits)   = {len(all_bits)}"
          f"   -> inflation factor {len(all_bits)*8/len(all_bits):.0f}x")
    d2 = Dec(concat)
    back2 = [d2.decode(cum) for _ in syms]
    print(f"    decode of concatenation (one decoder, sequential): {back2}")
    print(f"    matches original symbols? {back2 == syms}")
    print(f"    (correct prefix? {back2[:1] == syms[:1]}, "
          f"longest correct prefix = {_lcp(back2, syms)})")

    # --- (3) file size actually written by cli.py: 8 bits packed per byte ---
    nbytes = len(all_bits) // 8
    print(f"\n[3] bytes actually written to the .zllm file ~ {nbytes} B "
          f"({nbytes*8} bits), header claims total_bits={len(all_bits)*8} "
          f"-> bpb shown to the user is {len(all_bits)*8/(nbytes*8):.0f}x "
          f"the size of the file it wrote")


def _lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


if __name__ == "__main__":
    main()
