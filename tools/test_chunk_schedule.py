"""Chunk-schedule tests for tools/seg_token_compressor.py (stdlib only).

WHY THIS EXISTS
---------------
The codec forwards one token at a time and the model derives each token's
position from the KV cache length, so the cache *is* the position budget:
SmolLM2-135M was trained to 8192 positions and nothing may exceed that.

Two ways to spend the budget, and the difference is worth 0.008 bpb at 100KB:

  reset only          a chunk is `window` new tokens; its first token sees
                      ~no context and the mean context is window/2
  reset + overlap O   a chunk is `O` re-fed history + `window - O` new tokens;
                      every token sees at least O context, mean (O + window)/2

`--overlap 4096` reproduces v13's block structure (one 8192-token block of
4096 re-fed + 4096 new per block, `BLOCK_NEW = BLOCK_TOKENS - OVERLAP`), which
is what the 0.9139 / 377-escape reference run used.

A SLIDING WINDOW IS NOT AN OPTION HERE, and the reason is not obvious: trimming
the cache to its last `window` entries keeps the length at 8192, but the model
assigns the next token position == cache length, so every token after the first
window gets position 8192 while the STORED entries keep whatever position they
were created with -- the recent tokens all collapse onto one position and the
relative distances attention depends on are gone. That is a silent quality bug,
not a crash, and at 8KB it never fires because the cache never fills. Resetting
is the honest version.

Run: python3 tools/test_chunk_schedule.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import seg_token_compressor as stc          # noqa: E402

FAILS = []


def check(label, ok, detail=""):
    print(f"  [{'ok  ' if ok else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


def context_of(step, window, overlap):
    """Tokens of context the token at `step` is forwarded with."""
    stride = window - overlap
    c0 = (step // stride) * stride
    return step + 1 if c0 == 0 else overlap + (step - c0) + 1


def main():
    print("=" * 78)
    print("chunk schedule: position budget, prefill ranges, context floors")
    print("=" * 78)

    W = 8192
    for ov in (0, 1024, 4096, 6144, 7680, 8191):
        stride = W - ov
        # 1. positions stay inside the trained range, for a long file
        worst = max(stc.position_of(t, W, ov) for t in range(0, 30792))
        check(f"overlap {ov:5d}: max position <= window-1 ({worst} <= {W - 1})",
              worst <= W - 1)
        # 2. prefill happens exactly at chunk boundaries, and never touches the
        #    current step or anything before the segment start
        bad = []
        for t in range(0, 20000):
            reset, lo, hi = stc.chunk_schedule(t, W, ov)
            at_boundary = (t % stride == 0)
            if reset != at_boundary:
                bad.append(("reset", t))
            if hi > t or lo < 0 or (hi - lo) > ov:
                bad.append(("range", t, lo, hi))
            if hi > lo and not (t > 0 and ov > 0 and at_boundary):
                bad.append(("unexpected prefill", t))
        check(f"overlap {ov:5d}: boundaries + prefill ranges exact over 20k steps",
              not bad, str(bad[:3]))
        # 3. the context floor that this whole change is about
        after_first_chunk = [context_of(t, W, ov) for t in range(stride, 30792)]
        check(f"overlap {ov:5d}: min context after chunk 0 == overlap + 1 "
              f"({min(after_first_chunk)})",
              min(after_first_chunk) == ov + 1)
        # Chunk 0 is special (no prefill, contexts 1..stride); every LATER chunk
        # has contexts overlap+1 .. window. Average those, not chunk 0, or the
        # expectation silently describes a different schedule.
        full = [context_of(t, W, ov) for t in range(stride, 4 * stride)]
        check(f"overlap {ov:5d}: max context (later chunks) == window "
              f"({max(full)})", max(full) == W)
        # Mean over whole later chunks is overlap + (stride+1)/2
        # = (overlap+window+1)/2. A truncated final chunk drags the file-wide
        # mean below this; the tail is genuinely shorter, so do not widen the
        # tolerance on a file-wide number to hide that.
        exact = sum(full) / len(full)
        check(f"overlap {ov:5d}: whole-chunk mean == (overlap+window+1)/2 "
              f"({exact:.1f})", abs(exact - (ov + W + 1) / 2) <= 1e-9)

    print()
    print("-- the two operating points that matter --")
    for ov, label in ((0, "current default (reset only)"),
                      (4096, "v13 reference regime (OVERLAP=4096)")):
        stride = W - ov
        mean = sum(context_of(t, W, ov) for t in range(30792)) / 30792
        n_prefill = sum(1 for t in range(30792)
                        if stc.chunk_schedule(t, W, ov)[2] >
                        stc.chunk_schedule(t, W, ov)[1])
        print(f"  overlap {ov:5d}: stride {stride:5d}  mean context {mean:6.0f}  "
              f"prefills {n_prefill:3d}  ({label})")

    print()
    print("-- prefill slice mapping onto the rolling history buffer --")
    # The buffer holds the last `window` tokens, so index 0 is not token 0.
    # Replay the real schedule and check the mapped rows are the right TOKENS.
    for ov in (0, 4096, 7680):
        stride = W - ov
        fed, bad = [], []
        for step in range(0, 4 * stride):
            fed.append(1000 + step)    # token id == absolute index
            buf = fed[-W:]             # what the deque(maxlen=window) holds
            reset, lo, hi = stc.chunk_schedule(step, W, ov)
            if hi > lo:
                rlo, rhi = stc.prefill_offsets(step, lo, hi, len(buf))
                got = buf[rlo:rhi]
                want = fed[max(0, step - ov):step]
                if got != want:
                    bad.append((step, got[:3], want[:3]))
        check(f"overlap {ov:5d}: prefill rows == the last `overlap` fed tokens",
              not bad, str(bad[:2]))

    print()
    print("-- guards --")
    try:
        stc.chunk_schedule(4, 8192, 8192)
        check("overlap >= window is rejected", False)
    except ValueError:
        check("overlap >= window is rejected", True)
    check("step 0 never prefills (nothing is known yet)",
          stc.chunk_schedule(0, W, 4096) == (True, 0, 0))
    check("default arguments reproduce the pre-overlap behaviour",
          stc.chunk_schedule(0, W, 0)[0] and stc.chunk_schedule(W, W, 0) ==
          (True, 0, 0), "overlap=0 -> reset only")

    print()
    print("=" * 78)
    print(f"RESULT: {'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    print("=" * 78)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
