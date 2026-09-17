"""Where does the per-step cost actually go? One run, decisive.

WHY THIS IS THE ONLY THING WORTH MEASURING RIGHT NOW
----------------------------------------------------
Fitting the two 100 KB points (K=1: 21.9 ms/step, K=4: 64.9 ms/step):

    per-step = 7.6 ms fixed + 14.3 ms PER ROW

That per-row cost caps the whole design. 100 MB is ~27 M positions, so:

    K=1     21.9 ms/step x 27.0 M steps = 164 h   (per pass)
    K=4     64.9 ms/step x  6.8 M steps = 122 h
    K=30   437.6 ms/step x  0.9 M steps = 109 h
    K=100 1440.9 ms/step x  0.3 M steps = 108 h

**More K does not help** -- it approaches 27 M x 14.3 ms = 108 h asymptotically --
and K cannot exceed ~38 anyway, because each 8192-context segment holds ~184 MB of
KV cache on an 8 GB card. So 100 MB is 108-164 hours per pass with the current
per-row cost, and StaticCache's 12-15% moves that to ~144 h. Before spending a
week of GPU time (or upgrading transformers for a 12% cache win), the question is
whether 14.3 ms per single-token row is real work or overhead.

It is overhead. A single-token forward must read the segment's KV cache (188 MB at
8192 context) at ~448 GB/s = **0.42 ms/row**, plus a trivial amount of arithmetic.
14.3 ms is **34x the memory-bound floor**.

WHAT THIS SCRIPT MEASURES
-------------------------
A. a kernel-level profile of the real loop shape, so the answer is a table of ops
   rather than a hypothesis (this is the decisive part);
B. the per-row slope over K = 1..16 (confirming/refuting the 14.3 ms/row fit);
C. cost vs cache length 0 -> 8192 (is the per-row cost O(seq)? then it is cache
   traffic or mask construction, not dispatch);
D. CPU-submit time vs wall time (is the loop CPU-bound or GPU-bound?);
E. eager vs CUDA-graph on a FIXED shape with no cache -- this quantifies what a
   graph buys WITHOUT needing the graph-safe mask that transformers 4.57.6 blocks,
   i.e. it bounds the prize before anyone patches mask machinery;
F. a 8192-token block forward, for reference: what the same model costs per token
   when it is not called one token at a time.

Run:  python tools/probe_forward_cost.py                 (about 3 minutes)
      python tools/probe_forward_cost.py --profile       (adds the kernel table)
      python tools/probe_forward_cost.py --k-max 8 --skip-l-sweep

Nothing here writes files or touches seg_codec; it is pure measurement.
"""
import argparse
import gc
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get(
        "MODEL_OVERRIDE", "./data/cloud/SmolLM2-135M"))
    ap.add_argument("--k-max", type=int, default=16)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--window", type=int,
                    default=int(os.environ.get("KV_WINDOW", "8192")))
    ap.add_argument("--profile", action="store_true",
                    help="add a torch.profiler kernel table at K=4, L=window")
    ap.add_argument("--skip-l-sweep", action="store_true")
    args = ap.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM
    except Exception as exc:
        print(f"needs torch + transformers ({type(exc).__name__}: {exc})")
        return 2
    if not torch.cuda.is_available():
        print("no CUDA device visible")
        return 2

    dev = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.float16
    ).to(dev).eval()
    cfg = model.config
    V = int(cfg.vocab_size)
    n_layers = getattr(cfg, "num_hidden_layers", "?")
    kv_heads = getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads", "?"))
    hd = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    import transformers
    print("=" * 78)
    print("per-step cost: where the 14.3 ms/row goes")
    print("=" * 78)
    print(f"  torch {torch.__version__}, transformers {transformers.__version__}")
    print(f"  attn_implementation = {getattr(cfg, '_attn_implementation', '?')}"
          f"   (sdpa/math choice changes the attention cost per seq length)")
    print(f"  layers={n_layers} kv_heads={kv_heads} head_dim={hd} V={V} "
          f"window={args.window}")
    kv_mb = args.window * 2 * kv_heads * hd * 2 * n_layers / 2 ** 20
    print(f"  KV per segment at full context: {kv_mb:.0f} MB "
          f"-> floor {kv_mb / 448000 * 1e3:.2f} ms/row at 448 GB/s")
    try:
        print(f"  flash_sdp={torch.backends.cuda.flash_sdp_enabled()} "
              f"mem_efficient_sdp={torch.backends.cuda.mem_efficient_sdp_enabled()} "
              f"math_sdp={torch.backends.cuda.math_sdp_enabled()}")
    except Exception:
        pass

    gen = torch.Generator(device="cpu").manual_seed(11)

    def fresh(L, K):
        """A [K, L] token block to warm a cache with (one forward, not L)."""
        return torch.randint(0, V, (K, L), generator=gen).to(dev)

    def warm_cache(K, L):
        """Cache of length L for K rows, built with ONE block forward."""
        if L == 0:
            return None
        with torch.inference_mode():
            out = model(input_ids=fresh(L, K), use_cache=True)
        cache = out.past_key_values
        del out
        return cache

    def build_step(K, cache):
        """One-token forward for K rows against `cache`; no input H2D in the
        timed region (the driver already measures that: prep = 0.12 ms)."""
        x = torch.zeros((K, 1), dtype=torch.long, device=dev)

        def step():
            with torch.inference_mode():
                out = model(input_ids=x, past_key_values=cache, use_cache=True)
                _ = out.logits[:, -1, :].float()
        return step

    def ms(fn, n, sync=True):
        for _ in range(3):
            fn()
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        if sync:
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1e3

    print()
    print("-- A. kernel profile at the real loop shape (K=4, L=window) --")
    if args.profile:
        try:
            K, L = 4, args.window
            cache = warm_cache(K, L)
            step = build_step(K, cache)
            for _ in range(3):
                step()
            torch.cuda.synchronize()
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         record_shapes=False) as prof:
                for _ in range(10):
                    step()
                torch.cuda.synchronize()
            for sort in ("cuda_time_total", "self_cpu_time_total"):
                try:
                    print()
                    print(f"  top ops by {sort}:")
                    print(prof.key_averages().table(sort_by=sort, row_limit=14))
                except Exception as exc:
                    print(f"  [{sort} unavailable: {exc}]")
            del cache, step
            gc.collect(); torch.cuda.empty_cache()
        except Exception as exc:
            print(f"  profiler failed: {type(exc).__name__}: {exc}")
    else:
        print("  (skipped; re-run with --profile)")

    print()
    print("-- B. per-row slope (cache at full context) --")
    print(f"  {'K':>4} {'ms/step':>9} {'ms/row':>8} {'submit':>8} "
          f"{'step GB moved':>13}")
    base = None
    for K in [1, 2, 4, 8, 16]:
        if K > args.k_max:
            break
        try:
            L = args.window
            cache = warm_cache(K, L)
            step = build_step(K, cache)
            wall = ms(step, args.steps, sync=True)
            submit = ms(step, args.steps, sync=False)
            if base is None:
                base = (K, wall)
            gb = K * kv_mb / 1024
            print(f"  {K:>4} {wall:>9.2f} {wall / K:>8.2f} {submit:>8.2f} "
                  f"{gb:>13.2f}")
            del cache, step
            gc.collect(); torch.cuda.empty_cache()
        except Exception as exc:
            name = type(exc).__name__
            hint = ("OOM: each row holds its own cache "
                    f"({kv_mb:.0f} MB/row)" if "OutOfMemory" in name else "")
            print(f"  {K:>4}  FAILED {name}: {exc}"[:110] + f"  {hint}")
            torch.cuda.empty_cache()
    if base:
        print(f"  -> slope from K={base[0]} to K={args.k_max}: "
              f"read the ms/row column; if it is ~flat the cost is per-step "
              f"(dispatch), if it grows the cost is per-row (cache traffic)")

    if not args.skip_l_sweep:
        print()
        print("-- C. cost vs cache length (K=4) --")
        print(f"  {'L':>6} {'ms/step':>9} {'ms/row':>8} {'KV MB/row':>10}")
        for L in (0, 1024, 4096, 8192):
            if L > args.window:
                continue
            try:
                cache = warm_cache(4, L)
                step = build_step(4, cache)
                wall = ms(step, args.steps, sync=True)
                kv = L * 2 * kv_heads * hd * 2 * n_layers / 2 ** 20
                print(f"  {L:>6} {wall:>9.2f} {wall / 4:>8.2f} {kv:>10.1f}")
                del cache, step
                gc.collect(); torch.cuda.empty_cache()
            except Exception as exc:
                print(f"  {L:>6}  FAILED {type(exc).__name__}: {exc}"[:100])
                torch.cuda.empty_cache()

    print()
    print("-- D/E. graph vs eager on a FIXED shape, no cache --")
    print("     (this is the prize a CUDA graph can win WITHOUT the graph-safe")
    print("      mask work: same shapes, nothing dynamic, no KV cache)")
    for B, L in ((4, 2048), (4, args.window)):
        try:
            x = fresh(L, B)
            with torch.inference_mode():
                for _ in range(3):
                    out = model(input_ids=x, use_cache=False)
                torch.cuda.synchronize()

                def eager():
                    with torch.inference_mode():
                        o = model(input_ids=x, use_cache=False)
                        _ = o.logits[:, -1, :].float()

                e_ms = ms(eager, 10)
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    with torch.inference_mode():
                        model(input_ids=x, use_cache=False)
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    o = model(input_ids=x, use_cache=False)
                    o_lg = o.logits[:, -1, :].float()
                g.replay(); torch.cuda.synchronize()
                g_ms = ms(lambda: g.replay(), 10)
                print(f"  [B={B}, L={L}] eager={e_ms:8.2f} ms  "
                      f"graph={g_ms:8.2f} ms  -> x{e_ms / g_ms:4.2f}  "
                      f"(per token: {e_ms / (B * L) * 1e3:.4f} -> "
                      f"{g_ms / (B * L) * 1e3:.4f} ms)")
                del x, o, o_lg
                gc.collect(); torch.cuda.empty_cache()
        except Exception as exc:
            print(f"  [B={B}, L={L}] FAILED {type(exc).__name__}: {exc}"[:130])

    print()
    print("-- F. reference: one 8192-token block forward --")
    try:
        B, L = 1, args.window
        x = fresh(L, B)

        def blk():
            with torch.inference_mode():
                o = model(input_ids=x, use_cache=False)
                _ = o.logits.float()
        b_ms = ms(blk, 5)
        us_per_tok = b_ms / L * 1e3            # us/token (NOT ms -- see below)
        loop_ms = 21.9                         # driver K=1, 100 KB
        print(f"  block forward [1, {L}]: {b_ms:.1f} ms for {L} tokens = "
              f"{us_per_tok:.2f} us/token (= {us_per_tok / 1e3:.4f} ms/token)")
        print(f"  the loop's single-token path costs {loop_ms:.2f} ms/token "
              f"(K=1, measured in the driver)")
        print(f"  -> the loop is {loop_ms * 1e3 / us_per_tok:.0f}x SLOWER per "
              f"token than the model in bulk.")
        print("     UNIT NOTE: the first version of this printout divided ms by")
        print("     tokens and multiplied by 1e3, which yields MICROseconds, yet")
        print("     labelled it ms/token -- a 1000x error that made the bulk")
        print("     forward look 2.4x SLOWER than the loop instead of 400x")
        print("     faster. The bulk path does strictly more per-token work")
        print("     (quadratic O(L^2) attention, no cache); the loop does O(L)")
        print("     attention plus 270 MB of weight reads, whose bandwidth floor")
        print("     is ~0.9 ms/token. Use the floor, not this row, to judge the")
        print("     loop.")
        print("     (The bulk path is not usable for coding: encoder and decoder")
        print("      must run the SAME arithmetic or the mirror breaks.)")
        del x
        gc.collect(); torch.cuda.empty_cache()
    except Exception as exc:
        print(f"  FAILED {type(exc).__name__}: {exc}"[:120])

    print()
    print("=" * 78)
    print("HOW TO READ IT")
    print("=" * 78)
    print("""  * A dominant `aten::cat` / `index_copy` / `copy_` in the profile, or
    ms/row growing with L in C  -> the per-step cache write is the cost. Fix:
    StaticCache (measured 12-15% here, so probably NOT the whole story).
  * A dominant `aten::masked_fill` / `_update_causal_mask` / `aten::full` ->
    mask construction per step is the cost, and it is also what blocks capture.
    Fix: a static mask buffer + a custom mask update, NOT a transformers upgrade.
  * Flat ms/row in B with the profile showing many small kernels and a large
    CPU-side gap -> launch/dispatch overhead. Fix: CUDA graph, and E says how
    much is on the table.
  * submit ~= wall in B -> CPU-bound; no GPU-side change will help.
  * If E shows only ~1.1x, the graph is not the answer on this box and the
    remaining 34x-over-floor must come from somewhere else entirely.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
