"""Segment-parallel per-token compressor: a .zllm that decodes from the file alone.

WHAT MAKES THIS DIFFERENT FROM tools/real_compressor.py
------------------------------------------------------
real_compressor.py's decoder feeds the model the *source* tokens:

    ctx_chunk = ids[ci:ci + chunk_len + 1]

so its "identical logits" check is vacuous and the archive is not self-contained
(see data/sota_loop.json -> ledger-wording-correction).

Here the decoder owns nothing but the .zllm file and the model:
  * token 0 of every segment is coded with uniform(V) inside the stream;
  * every later token's distribution is produced from the tokens THAT SEGMENT has
    already emitted (KV cache built from decoded tokens only);
  * segment boundaries are derived from (n_tokens, n_segments) in the header.

K segments are run in lockstep as a batch of K (one token per segment per
forward), so the per-token forward cost is amortised K ways. Segments are coded
independently (own tables, own KV cache) -- that independence is exactly what
makes a lockstep decoder possible.

USAGE (RTX 3060 Ti, SmolLM2-135M, enwik8 at ./data/cloud/enwik8)

    # smoke test first -- ~1 min
    python tools/seg_token_compressor.py --kb 8 --segments 4

    # the two reference points
    python tools/seg_token_compressor.py --kb 100 --segments 1
    python tools/seg_token_compressor.py --kb 100 --segments 4

Results land in data/seg_verify_{K}seg_{kb}kb.json.

EXPECTED COST: one forward per token per pass, two passes (encode + decode), so
K=1 at 100KB is (~30 min x 2). K=4 should be several times faster if the batch
of 4 actually amortises; measure it.

SCALING NOTE (100MB): KV cache for the 135M model is ~23 KB/token, and every
segment holds its own cache, so total cache ~= 23 KB x tokens-per-segment.
100MB (~27M tokens) therefore needs segment lengths of ~200-300K tokens
(K >= 100), not K = 4. The code below is K-agnostic; only memory limits K.
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import seg_codec as sc      # noqa: E402  (numpy is imported lazily inside it)

MAGIC = b"ZLLM"
FORMAT_VERSION = 2


def cfg_dict(args):
    return dict(top_k=args.top_k, floor_frac=args.floor_frac,
                bigram_lambda=args.bigram_lambda, bigram_conf=args.bigram_conf,
                trigram_conf=args.trigram_conf,
                use_trigram=not args.no_trigram)


# ── model plumbing ───────────────────────────────────────────────────
def make_next_logits(model, torch, device, n_segments):
    """Returns next_logits(inp_tokens, step) -> [n_segments][V] probabilities.

    One batched forward per step: row s receives segment s's token at index
    `step`. The KV cache grows by one position per row per step, so position i of
    every segment is forwarded with position_ids == i -- identical on both sides.
    """
    states = [None] * 16  # up to 16 segments

    cache = None
    WINDOW = int(os.environ.get("KV_WINDOW", "8192"))

    def next_logits(inp, step):
        nonlocal cache
        t0 = time.time()
        with torch.inference_mode():
            x = torch.tensor(inp, dtype=torch.long, device=device).unsqueeze(1)  # [K,1]
            kw = dict(input_ids=x, use_cache=True)
            if cache is not None:
                kw["past_key_values"] = cache
            out = model(**kw)
            cache = out.past_key_values
            if cache[0][0].shape[2] > WINDOW:
                cache = tuple((k[:, :, -WINDOW:, :], v[:, :, -WINDOW:, :])
                              for k, v in cache)
            lg = out.logits[:, -1, :].float()
            lg = lg - lg.max(dim=-1, keepdim=True).values
            p = torch.exp(lg)
            p = (p / p.sum(dim=-1, keepdim=True)).cpu().numpy()
        return [p[s] for s in range(len(inp))], time.time() - t0

    return next_logits


def pack_bits(bitstr):
    """Right-pad the final partial chunk before packing (the fix for the bitpack
    bug: bytes(int(chunk, 2)) drops the leading zeros otherwise)."""
    chunks = [bitstr[i:i + 8] for i in range(0, len(bitstr), 8)]
    return bytes(int(c.ljust(8, "0"), 2) for c in chunks)


def unpack_bits(raw, n_bits):
    return "".join(f"{b:08b}" for b in raw)[:n_bits]


def write_zllm(path, meta, bitstr):
    hdr = json.dumps(meta, ensure_ascii=False).encode("utf-8")
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", FORMAT_VERSION))
        f.write(struct.pack("<I", len(hdr)))
        f.write(hdr)
        f.write(pack_bits(bitstr))


def read_zllm(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == MAGIC, f"bad magic {magic!r}"
        version = struct.unpack("<I", f.read(4))[0]
        hlen = struct.unpack("<I", f.read(4))[0]
        meta = json.loads(f.read(hlen).decode("utf-8"))
        payload = f.read()
    return version, meta, unpack_bits(payload, meta["n_bits"])


# ── main ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get(
        "MODEL_OVERRIDE", "./data/cloud/SmolLM2-135M"))
    ap.add_argument("--enwik8", default=os.environ.get(
        "ENWIK8_PATH", "./data/cloud/enwik8"))
    ap.add_argument("--offset-mb", type=int, default=50)
    ap.add_argument("--kb", type=int, default=100)
    ap.add_argument("--segments", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=1024)
    ap.add_argument("--floor-frac", type=float, default=1e-6)
    ap.add_argument("--bigram-lambda", type=float, default=0.99)
    ap.add_argument("--bigram-conf", type=float, default=10.0)
    ap.add_argument("--trigram-conf", type=float, default=3.0)
    ap.add_argument("--no-trigram", action="store_true")
    ap.add_argument("--shared-tables", action="store_true",
                    help="Share n-gram tables across segments (legal in lockstep)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--progress-every", type=int, default=200)
    args = ap.parse_args()

    cfg = cfg_dict(args)
    K = args.segments

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1/6] loading model {args.model} on {device}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.float16
    ).to(device).eval()
    V = int(model.config.vocab_size)
    print(f"      V={V}, loaded in {time.time() - t0:.1f}s")

    print(f"[2/6] reading enwik8 offset {args.offset_mb}MB, {args.kb}KB")
    with open(args.enwik8, "rb") as f:
        f.seek(args.offset_mb * 1024 * 1024)
        raw = f.read(args.kb * 1024)
    text = raw.decode("utf-8", errors="ignore")
    ids = tok.encode(text)
    n_tokens = len(ids)
    plan = sc.seg_plan(n_tokens, K)
    print(f"      {len(raw)} bytes -> {n_tokens} tokens; segments: "
          f"{[L for _, L in plan]}")

    # ---- encode -----------------------------------------------------
    print(f"[3/6] encoding (K={K}, one batched forward per step)")
    enc_uniform = sc.uniform_cum_cached(V, {})
    uni_cache = {}
    tables = sc.SegTables(K, use_trigram=cfg["use_trigram"],
                          shared=args.shared_tables)
    from ac32 import ArithmeticEncoder32
    enc = ArithmeticEncoder32(store=True)
    stats = dict(coded=0, escapes=0, uniform=0)

    nl_enc = make_next_logits(model, torch, device, K)
    maxL = max(L for _, L in plan)
    ticks = {"n": 0}

    def nl_enc_timed(inp, step):
        ticks["n"] += 1
        if args.progress_every and ticks["n"] % args.progress_every == 0:
            el = time.time() - t_enc
            eta = el / max(1, ticks["n"]) * (maxL - 1 - step)
            print(f"      enc step {step}/{maxL - 1}  {el:.0f}s elapsed, "
                  f"~{eta:.0f}s left", flush=True)
        return nl_enc(inp, step)

    t_enc = time.time()
    plan, timings = sc.encode_stream(ids, K, cfg, nl_enc_timed, enc, tables, enc_uniform,
                     sc.build_distribution, uni_cache, stats)
    bitstr, n_bits = enc.finish()
    dt_enc = time.time() - t_enc
    total_steps = maxL - 1
    per_step = dt_enc / max(1, total_steps) * 1000
    print(f"      PHASES: fwd={timings['fwd']:.1f}s dist={timings['dist']:.1f}s "
          f"ac={timings['ac']:.1f}s total={dt_enc:.1f}s")
    print(f"      per-step: {per_step:.1f}ms (target地板: ~0.6ms)")
    assert len(bitstr) == n_bits, (len(bitstr), n_bits)
    del nl_enc, nl_enc_timed
    if device.type == "cuda":
        torch.cuda.empty_cache()

    zpath = args.out or f"test_{args.kb}kb_{K}seg.zllm"
    meta = dict(format_version=FORMAT_VERSION, model=os.path.basename(args.model),
                V=V, n_tokens=n_tokens, n_bits=n_bits, n_segments=K,
                n_bytes=len(raw), offset_mb=args.offset_mb, kb=args.kb,
                cfg={k: v for k, v in cfg.items()}, kind="seg-per-token")
    write_zllm(zpath, meta, bitstr)
    file_bytes = os.path.getsize(zpath)
    print(f"      {n_bits} bits -> {zpath} ({file_bytes} bytes); "
          f"encode {dt_enc:.1f}s ({len(raw) / 1024 / dt_enc:.1f} KB/s)")

    # ---- decode from the FILE only ----------------------------------
    print("[4/6] decoding from the .zllm file (fresh model state)")
    version, meta_r, bitstr_r = read_zllm(zpath)
    assert meta_r["n_tokens"] == n_tokens and meta_r["n_segments"] == K
    from ac32 import ArithmeticDecoder32
    dec = ArithmeticDecoder32(bitstr_r)
    tables_d = sc.SegTables(K, use_trigram=cfg["use_trigram"],
                            shared=args.shared_tables)
    stats_d = dict(coded=0, escapes=0, uniform=0)
    nl_dec = make_next_logits(model, torch, device, K)
    t_dec = time.time()
    rec = sc.decode_stream(n_tokens, K, cfg, nl_dec, dec, tables_d,
                           sc.uniform_cum_cached(V, {}), sc.build_distribution,
                           {}, stats_d)
    dt_dec = time.time() - t_dec
    print(f"      decoded {len(rec)} tokens in {dt_dec:.1f}s")

    # ---- verify -----------------------------------------------------
    print("[5/6] verification")
    tokens_ok = (rec == ids)
    h_orig = hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest()
    h_rec = hashlib.sha256(struct.pack(f"<{len(rec)}I", *rec)).hexdigest()
    text_rec = tok.decode(rec)
    text_orig = tok.decode(ids)
    bytes_ok = (text_rec.encode("utf-8") == raw)
    tok_roundtrip_ok = (text_orig.encode("utf-8") == raw)

    bpb_payload = n_bits / len(raw)
    bpb_file = file_bytes * 8 / len(raw)
    print(f"      tokens identical : {tokens_ok}")
    print(f"      SHA-256 tokens   : {h_orig[:16]} vs {h_rec[:16]} -> "
          f"{h_orig == h_rec}")
    print(f"      bytes identical  : {bytes_ok} (tokenizer self-roundtrip: "
          f"{tok_roundtrip_ok})")
    print(f"      bpb payload={bpb_payload:.4f}  bpb file={bpb_file:.4f}  "
          f"({file_bytes} B for {len(raw)} B)")
    print(f"      seed overhead    : {K} x uniform(V) = ~{K * 16} bits "
          f"({K * 16 / n_bits * 100:.2f}% of stream)")

    result = dict(kind="seg-per-token", model=meta["model"], segments=K,
                  kb=args.kb, offset_mb=args.offset_mb, n_bytes=len(raw),
                  n_tokens=n_tokens, n_bits=n_bits, file_bytes=file_bytes,
                  bpb_payload=round(bpb_payload, 4), bpb_file=round(bpb_file, 4),
                  encode_s=round(dt_enc, 1), decode_s=round(dt_dec, 1),
                  kbs=round(len(raw) / 1024 / max(dt_enc + dt_dec, 1e-9), 2),
                  tokens_identical=bool(tokens_ok),
                  sha256_tokens_match=bool(h_orig == h_rec),
                  bytes_identical=bool(bytes_ok),
                  tokenizer_self_roundtrip=bool(tok_roundtrip_ok),
                  escapes=stats["escapes"], coded=stats["coded"],
                  uniform_coded=stats["uniform"], selftest="tools/test_seg_codec_mirror.py")
    out_json = f"data/seg_verify_{K}seg_{args.kb}kb.json"
    os.makedirs("data", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[6/6] wrote {out_json}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if (tokens_ok and h_orig == h_rec) else 1


if __name__ == "__main__":
    sys.exit(main())
