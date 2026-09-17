"""Feasibility probe: StaticCache + CUDA graph for the codec's decode loop.

WHAT THIS ANSWERS, IN ORDER
---------------------------
The codec's decode loop is one forward per step, batch K. Measured at K=1 (100 KB,
`--overlap 4096 --top-k 8192`): `fwd=674.5s` of 718s encode, i.e. 21.9 ms/step for
ONE row of a 135M model whose fp16 weights are 270 MB -- 36x the memory-bound
floor (~0.6 ms). Batching costs ~11 ms per extra row (K=4: 54.8 ms/step). So the
cost is per-call overhead, not arithmetic, and the fix is to take the Python /
dispatch / launch path out of the loop.

This script measures four configurations and checks that the fast ones are also
CORRECT -- a speedup that changes the logits changes the bitstream:

  1. eager, DynamicCache       -- what the codec does today
  2. eager, StaticCache        -- fixed-shape cache, still eager dispatch
  3. StaticCache + CUDA graph  -- one replay per step, incl. logits readback
  4. graph, no readback        -- lower bound; shows what the readback costs

CORRECTNESS IS CHECKED BY ALIGNING THE TWO CACHES, not by comparing numbers from
unrelated runs. A CUDA-graph capture performs real forwards and therefore writes
real tokens into the cache it was captured against, so the graph and the eager
reference are advanced with the SAME tokens in the SAME order (the capture warmup
replays steps 0..2, capture itself performs step 3), after which the two caches
hold identical contents and every later step can be compared pairwise.

WHY THE ANSWER IS NOT OBVIOUS
-----------------------------
* A CUDA graph cannot contain a `.cpu()` transfer, and the codec's loop does one
  every step (`p.cpu().numpy()`). A graph can therefore only cover the forward;
  [3] - [4] is what the readback costs, and it is the part a graph cannot remove.
* The codec RESETS its cache at chunk boundaries and re-warms it with a batched
  prefill. One static graph cannot express a variable-length prefill, so prefills
  stay eager and the graph covers steady-state steps only (prefills are <1% of
  wall clock at 100 KB, so this is not a real loss).
* StaticCache needs `cache_position` as a tensor to be replayable, and some
  transformers versions build the attention mask in Python from it, which breaks
  capture. Hence a probe, not a patch.

Run on the WSL box (GPU + model directory required):

    python tools/wsl_cuda_graph_probe.py                 # K=1 and K=4
    python tools/wsl_cuda_graph_probe.py --k 4 --steps 200

Writes nothing, imports nothing from seg_codec.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get(
        "MODEL_OVERRIDE", "./data/cloud/SmolLM2-135M"))
    ap.add_argument("--k", type=int, default=0, help="0 = run K=1 and K=4")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--window", type=int,
                    default=int(os.environ.get("KV_WINDOW", "8192")))
    ap.add_argument("--check-steps", type=int, default=10,
                    help="pairwise eager-vs-graph comparisons after alignment")
    args = ap.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        print(f"needs torch + transformers on a GPU box ({type(exc).__name__}: "
              f"{exc})")
        return 2
    if not torch.cuda.is_available():
        print("no CUDA device visible")
        return 2

    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.float16
    ).to(dev).eval()
    V = int(model.config.vocab_size)

    # Deterministic token stream; in-range values are all that matter for timing.
    gen = torch.Generator(device="cpu").manual_seed(7)
    stream = torch.randint(0, V, (args.window + args.steps + 64,),
                           generator=gen).tolist()

    def sync():
        torch.cuda.synchronize()

    print("=" * 78)
    print("StaticCache + CUDA graph probe for the codec decode loop")
    print("=" * 78)
    print(f"  model {args.model}, V={V}, window={args.window}, "
          f"steps={args.steps}")
    print(f"  torch {torch.__version__}, "
          f"transformers {getattr(__import__('transformers'), '__version__', '?')}")

    for K in ([args.k] if args.k else [1, 4]):
        print()
        print(f"--- K={K} ---")
        tokens = lambda i: torch.tensor(
            [stream[i + s] for s in range(K)],
            dtype=torch.long, device=dev).reshape(K, 1)
        rows = []

        def ms_per_step(fn, steps):
            for i in range(12):          # warmup, excluded from the timing
                fn(i)
            sync()
            t0 = time.perf_counter()
            for i in range(12, 12 + steps):
                fn(i)
            return (time.perf_counter() - t0) / steps * 1e3

        # ---- 1. eager + DynamicCache (the codec's current path) ----
        dyn = {"cache": None}

        def eager_dynamic(i, readback=True):
            with torch.inference_mode():
                kw = dict(input_ids=tokens(i), use_cache=True)
                if dyn["cache"] is not None:
                    kw["past_key_values"] = dyn["cache"]
                out = model(**kw)
                dyn["cache"] = out.past_key_values
                lg = out.logits[:, -1, :].float()
                lg = lg - lg.max(dim=-1, keepdim=True).values
                p = torch.exp(lg)
                p = p / p.sum(-1, keepdim=True)
                return p.cpu().numpy() if readback else p

        ms1 = ms_per_step(eager_dynamic, args.steps)
        rows.append(("1 eager + DynamicCache (today)", ms1, ""))
        print(f"  [1] eager + DynamicCache       {ms1:8.2f} ms/step")

        # ---- 2. eager + StaticCache ----
        ms2 = static_ok = None
        try:
            from transformers import StaticCache
        except Exception as exc:
            print(f"  [2] StaticCache unavailable    {type(exc).__name__}: {exc}")
            StaticCache = None

        def new_scache():
            return StaticCache(config=model.config, max_batch_size=K,
                               max_cache_len=args.window, device=dev,
                               dtype=torch.float16)

        def eager_static(i, cache, pos, readback=True):
            with torch.inference_mode():
                out = model(input_ids=tokens(i), past_key_values=cache,
                            cache_position=pos, use_cache=True)
                lg = out.logits[:, -1, :].float()
                lg = lg - lg.max(dim=-1, keepdim=True).values
                p = torch.exp(lg)
                p = p / p.sum(-1, keepdim=True)
                return p.cpu().numpy() if readback else p

        if StaticCache is not None:
            try:
                c2, pos2 = new_scache(), torch.zeros(1, dtype=torch.long, device=dev)

                def step2(i, readback=True):
                    pos2.fill_(i)
                    return eager_static(i, c2, pos2, readback)

                ms2 = ms_per_step(step2, args.steps)
                static_ok = True
                rows.append(("2 eager + StaticCache", ms2, ""))
                print(f"  [2] eager + StaticCache        {ms2:8.2f} ms/step")
            except Exception as exc:
                print(f"  [2] eager + StaticCache FAILED "
                      f"{type(exc).__name__}: {exc}"[:150])

        # ---- 3. StaticCache + CUDA graph, aligned to [2]'s cache state ----
        if static_ok:
            try:
                # `c3` is advanced by the SAME tokens as `c2` up to step 3, so
                # after the capture both caches hold identical contents and the
                # pairwise comparison below is meaningful.
                c3 = new_scache()
                pos3 = torch.zeros(1, dtype=torch.long, device=dev)
                static_x = torch.zeros((K, 1), dtype=torch.long, device=dev)
                for i in range(3):                      # capture warmup
                    static_x.copy_(tokens(i))
                    pos3.fill_(i)
                    with torch.inference_mode():
                        model(input_ids=static_x, past_key_values=c3,
                              cache_position=pos3, use_cache=True)
                static_x.copy_(tokens(3))
                pos3.fill_(3)
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    with torch.inference_mode():
                        model(input_ids=static_x, past_key_values=c3,
                              cache_position=pos3, use_cache=True)
                torch.cuda.current_stream().wait_stream(side)
                sync()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = model(input_ids=static_x, past_key_values=c3,
                                cache_position=pos3, use_cache=True)
                    lg = out.logits[:, -1, :].float()
                    lg = lg - lg.max(dim=-1, keepdim=True).values
                    g_p = torch.exp(lg)
                    g_p = g_p / g_p.sum(-1, keepdim=True)

                def snap_eager(i):
                    """Eager forward at step i -- position MUST be set first.

                    The first version of this probe reused `pos2` without
                    refilling it, so the eager side was compared while running at
                    a stale position and the check compared two different cache
                    states. Positions are per-step inputs; never assume one.
                    """
                    pos2.fill_(i)
                    return eager_static(i, c2, pos2, readback=False)

                def snap_graph(i):
                    static_x.copy_(tokens(i))
                    pos3.fill_(i)
                    graph.replay()
                    return g_p.clone()

                # capture itself performed step 3 -> both caches are at step 3
                checks = []
                for i in range(4, 4 + args.check_steps):
                    pe, pg = snap_eager(i), snap_graph(i)
                    checks.append((i,
                                   int((pe.argmax(-1) != pg.argmax(-1)).sum()),
                                   float((pe.max(-1).values
                                          - pg.max(-1).values).abs().max())))
                # Only the positions are rewound for the timing run; the cache
                # buffers keep whatever the checks wrote. Identical work, so the
                # ms/step below is unaffected -- the correctness verdict comes
                # from the aligned checks above, not from this run.
                static_x.copy_(tokens(3))
                pos3.fill_(3)

                def step3(i, readback=True):
                    pg = snap_graph(i)
                    return pg.cpu().numpy() if readback else None
                worst_i, worst_top1, worst_dp = max(checks, key=lambda r: r[2])
                verdict = ("SAME" if worst_top1 == 0 and worst_dp < 1e-3
                           else "DIFFERS")
                print(f"      graph vs eager, {len(checks)} aligned steps: "
                      f"top-1 diffs={sum(c[1] for c in checks)}, "
                      f"worst |dlogp|={worst_dp:.2e} at step {worst_i} -> "
                      f"{verdict}")
                ms3 = ms_per_step(step3, args.steps)
                rows.append(("3 StaticCache + CUDA graph", ms3,
                             f"aligned check: {verdict}"))
                print(f"  [3] graph (fwd+norm+readback) {ms3:8.2f} ms/step")
                ms4 = ms_per_step(lambda i: step3(i, readback=False), args.steps)
                rows.append(("4 graph, no readback", ms4, "lower bound"))
                print(f"  [4] graph (no readback)        {ms4:8.2f} ms/step")
                mb = K * V * 4 / 2 ** 20
                print(f"      readback cost {ms3 - ms4:6.2f} ms/step for "
                      f"{mb:.2f} MB/step (full [K,V] fp32 row)")
                print(f"      a GPU top-K of PREFILTER=8192 would send "
                      f"{K * 8192 * 4 / 2 ** 20:.2f} MB/step, i.e. "
                      f"{V / 8192:.0f}x less")
            except Exception as exc:
                rows.append(("3 StaticCache + CUDA graph", float("nan"),
                             f"{type(exc).__name__}: {exc}"[:90]))
                print(f"  [3] CUDA graph FAILED          "
                      f"{type(exc).__name__}: {exc}"[:150])
                print("      likely causes: attention mask built in Python from "
                      "cache_position; a capture-time allocation; an unsupported "
                      "op. Try GRAPH_FULL-style capture of the backbone only.")

        print(f"  -- K={K} summary --")
        for name, ms, note in rows:
            if ms == ms:
                print(f"    {name:32s} {ms:8.2f} ms/step  x{ms1 / ms:5.2f} vs "
                      f"today   {note}")
            else:
                print(f"    {name:32s}     FAILED               {note}")

    print()
    print("=" * 78)
    print("HOW TO READ THIS")
    print("=" * 78)
    print("""  * [2] much faster than [1]  -> the cost is DynamicCache's per-step
    bookkeeping (reallocation / concat); StaticCache alone buys it.
  * [3] ~= [2]                 -> the graph adds nothing beyond StaticCache on
    this box; the rest is the launch cost of a fixed kernel set (WSL2 adds its
    own per-call cost -- if the win looks too small, run the same probe on native
    Linux for comparison).
  * [3] - [4]                  -> what the full-row readback costs. If it is a
    large share, the fix is v13's: do the top-K on the GPU inside the graph and
    transfer only PREFILTER values. That shrinks the D2H payload 6x at our
    settings but changes build_distribution's input from a dense row to a sparse
    one, so it is a separate, larger change -- do it only if this number says so.
  * The reset + prefill stays eager regardless, so a win here applies to the
    steady-state steps. Prefills are <1% of wall clock at 100 KB.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
