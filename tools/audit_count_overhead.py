"""Is v13's "counted" total comparable to a single-stream coder's output?

WHY THIS EXISTS
---------------
v13 reports `total_bits` for runs that write no file. That number comes from
`nb_encode_count_32`, which is the arithmetic coder's own exact bit accounting --
but it is called ONCE PER STAGE:

    n1 = nb_encode_count_32(C, sym_arr)              # all stage-1 symbols
    for ... in esc_infos:                            # then, per escape:
        n2_total += nb_encode_count_32(co_row, [rank])

and each call starts a coder from scratch and finishes it (`Returns total incl.
finish bits`). So every escape event costs its stage-2 symbol PLUS a fresh
finish, while a real single-stream coder would code the same symbol as a
continuation of the stream already in flight. The codec writes one stream, so
comparing `codec n_bits` against `v13 total_bits` compares two conventions.

This script measures the size of that convention difference using ac32's own
functions, and prints the inflation it implies for a given escape count. It
needs numpy + numba (it calls the real kernel); no torch, no GPU.

Run: python3 tools/audit_count_overhead.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np                                  # noqa: E402
import ac32                                         # noqa: E402

TOP_K = 1024
V = 49152
REST = V - TOP_K


def measure(esc_mass, ranks=(0, 1000, 24000, 48000)):
    """Bits for stage 2 coded alone (v13's counting) vs as a continuation."""
    stage1 = np.zeros(TOP_K + 1)
    body = (1.0 - esc_mass) / TOP_K
    stage1[:TOP_K] = body
    stage1[TOP_K] = esc_mass                       # the escape symbol itself
    c1 = ac32.stage1_cum_32(stage1[:TOP_K], esc_mass, 1e-6)
    c2 = ac32.uniform_cum_32(REST)

    alone_e = int(ac32.nb_encode_count_32(c1.reshape(1, -1),
                                          np.array([TOP_K], dtype=np.int64)))
    rows = []
    for r in ranks:
        alone_r = int(ac32.nb_encode_count_32(c2.reshape(1, -1),
                                              np.array([r], dtype=np.int64)))
        # One coder call, two symbols, two DIFFERENT alphabet sizes -> pad the
        # rows to a common width. The kernel only reads cums[t, s] and
        # cums[t, s+1] for each row's own symbol, so the padding beyond row 0's
        # cdf is never touched (kept monotone anyway).
        w = max(len(c1), len(c2))
        cu = np.empty((2, w), dtype=np.int64)
        cu[0, :len(c1)] = c1
        cu[0, len(c1):] = c1[-1]
        cu[1, :len(c2)] = c2
        both = int(ac32.nb_encode_count_32(
            cu, np.array([TOP_K, r], dtype=np.int64)))
        rows.append((r, alone_e, alone_r, both, alone_e + alone_r - both))
    return rows, REST


def main():
    print("=" * 78)
    print("v13 counted-total vs single-stream cost: the per-escape finish")
    print("=" * 78)
    print(f"  alphabet: TOP_K={TOP_K}, V={V}, rest={REST} "
          f"(log2(rest)={np.log2(REST):.2f} bits per escaped symbol)")

    for esc_mass in (0.001, 0.01, 0.05, 0.30):
        rows, rest = measure(esc_mass)
        ideal_e = -np.log2(esc_mass)
        ideal_r = np.log2(rest)
        print()
        print(f"  escape mass {esc_mass:g}: ideal stage-1 escape symbol = "
              f"{ideal_e:.2f} bits")
        print(f"    {'rank':>7} | {'alone:esc':>9} {'alone:rank':>10} "
              f"{'one stream':>10} | {'overhead':>8}")
        for r, ae, ar, both, ovh in rows:
            print(f"    {r:>7} | {ae:>9} {ar:>10} {both:>10} | {ovh:>+8}")

    print()
    print("  -- what that implies for the recorded runs --")
    obs = [("100KB, TOP_K=1024", 377, 517), ("100KB, TOP_K=8192", 27, 51)]
    rows, _ = measure(0.01)
    ovh = np.mean([r[4] for r in rows])
    print(f"    measured mean overhead per split call: {ovh:.2f} bits")
    for label, nesc, gap in obs:
        pred = nesc * ovh
        print(f"    {label:22s}: {nesc:4d} escapes -> predicted inflation "
              f"{pred:7.0f} bits vs observed codec-minus-v13 gap {gap:5d} bits")
    print()
    print("  Reading: if the observed gaps track escapes x overhead, the residual")
    print("  is the counting convention (one finish per escape call), not a")
    print("  difference in the blended distribution -- the codec's math is a port")
    print("  of nb_blend_row, so at matched context the two should agree.")


if __name__ == "__main__":
    main()
