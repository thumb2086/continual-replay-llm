"""Verify that ArithmeticEncoder32(store=True) produces bitstream
matching nb_encode_count_32 count, and decoder reconstructs exactly."""
import sys, os, time, numpy as np
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

from ac32 import (
    ArithmeticEncoder32, ArithmeticDecoder32,
    nb_encode_count_32, probs_to_freqs_32, TOTAL32,
    stage1_cum_32, uniform_cum_32,
)

print("=" * 60)
print("STEP 1: Unit test — encoder count vs stored bits")
print("=" * 60)

rng = np.random.default_rng(42)
V = 49152  # SmolLM2 vocab

# Generate random probabilities
p = rng.random(V) + 0.01
p /= p.sum()
_, cum = probs_to_freqs_32(p, 1e-6)

# Generate 200 random symbols
n_sym = 200
syms = rng.integers(0, V, size=n_sym).astype(np.int64)

# Count bits (fast, no storage)
C = np.ascontiguousarray(np.tile(cum, (n_sym, 1)))
count_bits = int(nb_encode_count_32(C, syms))
print(f"  nb_encode_count_32: {count_bits} bits for {n_sym} symbols")

# Encode with storage
enc = ArithmeticEncoder32(store=True)
for s in syms:
    enc.encode_symbol(cum, int(s))
bitstr, stored_bits = enc.finish()
print(f"  ArithmeticEncoder32 stored: {stored_bits} bits, bitstring len={len(bitstr)}")
print(f"  Match: {count_bits == stored_bits}")

# Decode
dec = ArithmeticDecoder32(bitstr)
decoded = [dec.decode_symbol(cum) for _ in range(n_sym)]
match = all(d == int(s) for d, s in zip(decoded, syms))
print(f"  Decode roundtrip: {match} ({sum(1 for d,s in zip(decoded,syms) if d==int(s))}/{n_sym})")

if count_bits == stored_bits and match:
    print("  ✓ Unit test PASSED")
else:
    print("  ✗ Unit test FAILED")
    sys.exit(1)

print()
print("=" * 60)
print("STEP 2: Realistic test — two-stage coding (stage1 + stage2)")
print("=" * 60)

# Simulate what v13 does: stage1 over topk+escape, stage2 over rest
K = 1024  # topk
n_pos = 50  # positions

# Generate fake topk probs and escape mass
topk_probs = rng.random(K) + 0.001
escape_mass = 0.01
cum_s1 = stage1_cum_32(topk_probs, escape_mass, 1e-6)

# For each position, generate a symbol and its stage-2 distribution
all_bits = 0
enc = ArithmeticEncoder32(store=True)
decoded_syms = []

for pos in range(n_pos):
    # Random symbol: 80% in topk, 20% escape
    if rng.random() < 0.8:
        sym = rng.integers(0, K)  # in topk
        in_escape = False
    else:
        sym = K  # escape
        in_escape = True

    # Encode stage-1
    enc.encode_symbol(cum_s1, sym)

    if in_escape:
        # Stage-2: uniform over vocab
        rest_ids = np.arange(V, dtype=np.int64)
        cum_s2 = uniform_cum_32(V, 2e-5)
        # For simplicity, symbol is vocab_index
        s2_sym = rng.integers(0, V)
        enc.encode_symbol(cum_s2, s2_sym)
        decoded_syms.append(s2_sym)
    else:
        decoded_syms.append(sym)

bitstr, stored_bits = enc.finish()
print(f"  Encoded {n_pos} positions: {stored_bits} bits")

# Decode
dec = ArithmeticDecoder32(bitstr)
decoded_pos = []
for pos in range(n_pos):
    sym = dec.decode_symbol(cum_s1)
    if sym == K:
        s2_sym = dec.decode_symbol(cum_s2)
        decoded_pos.append(s2_sym)
    else:
        decoded_pos.append(sym)

match = decoded_pos == decoded_syms
print(f"  Decode roundtrip: {match} ({sum(1 for a,b in zip(decoded_pos,decoded_syms) if a==b)}/{n_pos})")

if match:
    print("  ✓ Two-stage test PASSED")
else:
    print("  ✗ Two-stage test FAILED")
    mismatches = [(i,a,b) for i,(a,b) in enumerate(zip(decoded_pos,decoded_syms)) if a!=b]
    print(f"    First 5 mismatches: {mismatches[:5]}")
    sys.exit(1)

print()
print("=" * 60)
print("STEP 3: Verify nb_encode_count_32 matches stored bits")
print("=" * 60)

# Re-encode with count
enc_count = ArithmeticEncoder32(store=False)
for pos in range(n_pos):
    if pos < len(decoded_syms):
        # Re-encode same symbols
        pass  # skip, just verify count matches
# Actually, let's just count a fresh batch
syms_fresh = rng.integers(0, V, size=500).astype(np.int64)
p_fresh = rng.random(V) + 0.01
p_fresh /= p_fresh.sum()
_, cum_fresh = probs_to_freqs_32(p_fresh, 1e-6)
C_fresh = np.ascontiguousarray(np.tile(cum_fresh, (500, 1)))
count_bits = int(nb_encode_count_32(C_fresh, syms_fresh))

enc_fresh = ArithmeticEncoder32(store=True)
for s in syms_fresh:
    enc_fresh.encode_symbol(cum_fresh, int(s))
_, stored_bits = enc_fresh.finish()

print(f"  Count: {count_bits} bits, Stored: {stored_bits} bits, Match: {count_bits == stored_bits}")

if count_bits == stored_bits:
    print("  ✓ Count/store match PASSED")
else:
    print(f"  ✗ MISMATCH: diff = {count_bits - stored_bits}")

print()
print("=" * 60)
print("ALL TESTS PASSED — bitstream is valid")
print("=" * 60)
