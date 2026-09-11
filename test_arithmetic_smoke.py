"""Smoke test for the arithmetic coder (no model, no data, seconds to run).

Tests:
1. Round-trip lossless on random distributions (including near-zero probs).
2. Zero-probability symbols (the exact case that caused the OOM).
3. Memory bound: bitstring length stays sane.
"""

import sys
import os
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from real_compression import (
    ArithmeticEncoder,
    ArithmeticDecoder,
    probs_to_freqs,
    TOTAL,
)


def roundtrip(probs_list, symbols):
    enc = ArithmeticEncoder(store=True)
    for probs, s in zip(probs_list, symbols):
        _, cum = probs_to_freqs(probs)
        enc.encode_symbol(cum, s)
    bitstr, nbits = enc.finish()
    dec = ArithmeticDecoder(bitstr)
    out = []
    for probs in probs_list:
        _, cum = probs_to_freqs(probs)
        out.append(dec.decode_symbol(cum))
    return out, nbits


def main():
    print("Smoke test: arithmetic coder")
    rng = np.random.default_rng(0)

    # 1. Random distributions, random symbols
    for trial in range(20):
        V = 50
        seq_len = 30
        probs_list = []
        symbols = []
        for _ in range(seq_len):
            p = rng.random(V)
            p = p / p.sum()
            probs_list.append(p)
            symbols.append(int(rng.integers(0, V)))
        out, nbits = roundtrip(probs_list, symbols)
        assert out == symbols, f"roundtrip fail trial {trial}"
        assert nbits < 64 * seq_len, f"bit blowup: {nbits}"
    print("  1. random distributions: 20/20 round-trip OK")

    # 2. Zero-probability symbols (the OOM case)
    for trial in range(20):
        V = 100
        p = rng.random(V)
        p[:90] = 0.0  # 90% of symbols have exactly zero probability
        p = p / p.sum()
        symbols = [int(rng.integers(90, 100)) for _ in range(30)]  # only nonzero ones
        out, nbits = roundtrip([p] * 30, symbols)
        assert out == symbols, f"zero-prob roundtrip fail trial {trial}"
    print("  2. zero-probability symbols: 20/20 round-trip OK (OOM case fixed)")

    # 3. Extremely skewed distribution
    p = np.zeros(3239)
    p[0] = 0.999999
    p[1:] = (1 - 0.999999) / 3238
    out, nbits = roundtrip([p] * 50, [0] * 50)
    assert out == [0] * 50
    print(f"  3. skewed distribution: OK ({nbits} bits for 50 symbols)")

    # 4. Uniform distribution over large vocab
    p = np.ones(3239) / 3239
    symbols = [int(x) for x in rng.integers(0, 3239, 50)]
    out, nbits = roundtrip([p] * 50, symbols)
    assert out == symbols
    print(f"  4. uniform 3239-class: OK ({nbits} bits, ~{nbits/50:.2f} bits/symbol)")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
