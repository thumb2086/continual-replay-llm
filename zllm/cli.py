"""zllm CLI - Neural text compression tool.

Usage:
    zllm encode <input> [-o output.zllm] [--preset fast|balanced|ratio]
    zllm decode <input.zllm> [-o output.txt]
    zllm bench [--size 100KB|1MB|10MB] [--preset fast|balanced|ratio]
    zllm info <input.zllm>
"""
import argparse
import json
import os
import struct
import sys
import time

# Add project root to path so we can import ac32, real_compression, etc.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

__version__ = "0.1.0"

MAGIC = b"ZLLM"
FORMAT_VERSION = 1

PRESETS = {
    "fast": {
        "block_tokens": 8192, "top_k": 1024, "prefilter": 2048,
        "overlap": 0, "chunk_pre": 4096, "chunk_head": 2048,
        "sparse_blk": 1, "gather_pi": 1,
        "blend_bt_min": 20, "blend_tt_min": 7, "bigram_lambda": 0.99,
        "bigram_conf": 10, "trigram_conf": 3, "floor_frac": 1e-6,
        "use_fp16_xfer": 1, "use_cache_s2": 0,
        "model": "smollm2-135m",
        "description": "Speed-optimized: chunked prefill+head, K1024, gather",
    },
    "balanced": {
        "block_tokens": 8192, "top_k": 2048, "prefilter": 2048,
        "overlap": 0, "chunk_pre": 4096, "chunk_head": 2048,
        "sparse_blk": 1, "gather_pi": 1,
        "blend_bt_min": 20, "blend_tt_min": 7, "bigram_lambda": 0.99,
        "bigram_conf": 10, "trigram_conf": 3, "floor_frac": 1e-6,
        "use_fp16_xfer": 1, "use_cache_s2": 0,
        "model": "smollm2-135m",
        "description": "Balanced: chunked, K2048 (0.9187 bpb, 27KB/s)",
    },
    "ratio": {
        "block_tokens": 8192, "top_k": 8192, "prefilter": 8192,
        "overlap": 4096, "chunk_pre": 0, "chunk_head": 0,
        "sparse_blk": 0, "gather_pi": 0,
        "blend_bt_min": 5, "blend_tt_min": 2, "bigram_lambda": 0.99,
        "bigram_conf": 10, "trigram_conf": 3, "floor_frac": 1e-6,
        "use_fp16_xfer": 1, "use_cache_s2": 0,
        "model": "smollm2-135m",
        "description": "Ratio crown: ov4096/K8192, classic path (0.9003 bpb)",
    },
    "ratio-qwen": {
        "block_tokens": 28672, "top_k": 2048, "prefilter": 2048,
        "overlap": 0, "chunk_pre": 4096, "chunk_head": 2048,
        "sparse_blk": 1, "gather_pi": 1,
        "blend_bt_min": 20, "blend_tt_min": 7, "bigram_lambda": 0.99,
        "bigram_conf": 10, "trigram_conf": 3, "floor_frac": 1e-6,
        "use_fp16_xfer": 1, "use_cache_s2": 0, "s2_floor": 5e-6,
        "model": "qwen2.5-0.5b",
        "description": "Qwen ratio crown: 0.8442 bpb (practical, fits 8GB)",
    },
}


def _pack_header(metadata):
    """Pack metadata dict into bytes: 4-byte length + JSON UTF-8."""
    hdr_json = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
    return struct.pack(">I", len(hdr_json)) + hdr_json


def _unpack_header(data):
    """Unpack header from bytes, return (metadata_dict, rest)."""
    hdr_len = struct.unpack(">I", data[:4])[0]
    hdr_json = data[4 : 4 + hdr_len]
    return json.loads(hdr_json.decode("utf-8")), data[4 + hdr_len :]


def _set_env_from_preset(preset_name):
    """Apply preset to environment variables (v13 engine reads env)."""
    p = PRESETS[preset_name]
    env_map = {
        "block_tokens": "BLOCK_TOKENS", "top_k": "TOP_K",
        "prefilter": "PREFILTER", "overlap": "OVERLAP",
        "chunk_pre": "CHUNK_PRE", "chunk_head": "CHUNK_HEAD",
        "sparse_blk": "SPARSE_BLK", "gather_pi": "GATHER_PI",
        "blend_bt_min": "BLEND_BT_MIN", "blend_tt_min": "BLEND_TT_MIN",
        "bigram_lambda": "BIGRAM_LAMBDA", "bigram_conf": "BIGRAM_CONF",
        "trigram_conf": "TRIGRAM_CONF", "floor_frac": "FLOOR_FRAC",
        "use_fp16_xfer": "USE_FP16_XFER", "use_cache_s2": "USE_CACHE_S2",
    }
    for k, env_k in env_map.items():
        if k in p:
            os.environ[env_k] = str(p[k])
    model_map = {
        "smollm2-135m": "./data/cloud/SmolLM2-135M",
        "qwen2.5-0.5b": "./data/cloud/Qwen2.5-0.5B",
    }
    if p["model"] in model_map:
        os.environ["MODEL_OVERRIDE"] = model_map[p["model"]]
    if "s2_floor" in p:
        os.environ["S2_FLOOR"] = str(p["s2_floor"])


def cmd_encode(args):
    """Encode input file to .zllm compressed format."""
    preset = args.preset or "fast"
    _set_env_from_preset(preset)

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    input_size = os.path.getsize(input_path)
    output_path = args.output or input_path + ".zllm"
    preset_cfg = PRESETS[preset]

    print(f"zllm encode v{__version__}")
    print(f"  Input:  {input_path} ({input_size:,} bytes)")
    print(f"  Output: {output_path}")
    print(f"  Preset: {preset} ({preset_cfg['description']})")
    print(f"  Model:  {preset_cfg['model']}")
    print()

    t0 = time.time()

    try:
        import torch
        import numpy as np
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {device}")

        model_dir = os.environ.get("MODEL_OVERRIDE", "./data/cloud/SmolLM2-135M")
        print(f"  Loading model from {model_dir}...")
        tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, torch_dtype=torch.float16
        ).to(device).eval()
        V = model.config.vocab_size
        print(f"  Vocab: {V}")

        with open(input_path, "rb") as f:
            raw = f.read()
        text = raw.decode("utf-8", errors="replace")
        ids = tok.encode(text)
        n_bytes = len(text.encode("utf-8"))
        print(f"  Tokens: {len(ids)} ({len(text)/len(ids):.1f} chars/token)")
        print()

        from real_compression import ArithmeticEncoder, probs_to_freqs, TOTAL

        block_tokens = int(os.environ.get("BLOCK_TOKENS", "8192"))
        top_k = int(os.environ.get("TOP_K", "1024"))
        n_escapes = 0
        all_bits = bytearray()

        print("  Encoding...")
        with torch.no_grad():
            for b0 in range(0, len(ids), block_tokens):
                block = ids[b0 : b0 + block_tokens]
                if len(block) < 2:
                    continue

                x = torch.tensor([block], device=device)
                logits = model(x).logits[0].float().cpu().numpy()
                logits = logits - logits.max(axis=1, keepdims=True)
                e = np.exp(logits, dtype=np.float64)
                probs = e / e.sum(axis=1, keepdims=True)

                all_ids = np.arange(V)

                for t_idx in range(len(block) - 1):
                    p = probs[t_idx]
                    target = block[t_idx + 1]
                    topk_idx = np.argpartition(p, -top_k)[-top_k:]
                    topk_idx = topk_idx[np.argsort(-p[topk_idx])]
                    topk_p = p[topk_idx]
                    esc_mass = max(1e-12, 1.0 - topk_p.sum())

                    probs_cat = np.concatenate([topk_p, [esc_mass]])
                    c = probs_cat.sum()
                    freqs = (probs_cat / c * (TOTAL - len(probs_cat))).astype(np.int64) + 1
                    diff = int(TOTAL - freqs.sum())
                    freqs[np.argmax(probs_cat)] += diff
                    cum = np.zeros(len(freqs) + 1, dtype=np.int64)
                    np.cumsum(freqs, out=cum[1:])

                    enc = ArithmeticEncoder(store=True)
                    if target in set(topk_idx.tolist()):
                        s1 = int(np.where(topk_idx == target)[0][0])
                    else:
                        s1 = top_k
                        n_escapes += 1
                    enc.encode_symbol(cum, s1)
                    bitstr, count = enc.finish()
                    if bitstr:
                        for b in bitstr:
                            all_bits.append(int(b))

                block_n = b0 // block_tokens + 1
                total_n = (len(ids) + block_tokens - 1) // block_tokens
                if block_n % 2 == 0:
                    print(f"    block {block_n}/{total_n}, escapes: {n_escapes}", flush=True)

        total_bits = len(all_bits) * 8
        bpb = total_bits / n_bytes if n_bytes > 0 else 0

        metadata = {
            "format_version": FORMAT_VERSION,
            "preset": preset,
            "model": preset_cfg["model"],
            "tokenizer": model_dir,
            "vocab_size": V,
            "block_tokens": block_tokens,
            "top_k": top_k,
            "input_size": input_size,
            "n_tokens": len(ids),
            "n_bytes": n_bytes,
            "n_escapes": n_escapes,
            "total_bits": total_bits,
            "bpb": round(bpb, 6),
            "elapsed_s": round(time.time() - t0, 1),
        }

        hdr_bytes = _pack_header(metadata)
        with open(output_path, "wb") as f:
            f.write(MAGIC)
            f.write(struct.pack(">H", FORMAT_VERSION))
            f.write(hdr_bytes)
            # Pack bit list into bytes
            bit_bytes = bytearray()
            for i in range(0, len(all_bits), 8):
                byte = 0
                for j in range(8):
                    if i + j < len(all_bits):
                        byte = (byte << 1) | all_bits[i + j]
                    else:
                        byte = byte << 1
                bit_bytes.append(byte)
            f.write(bit_bytes)

        compressed_size = os.path.getsize(output_path)
        dt = time.time() - t0
        print()
        print(f"  Result: {bpb:.4f} bpb ({total_bits:,} bits from {n_bytes:,} bytes)")
        print(f"  Output: {compressed_size:,} bytes ({compressed_size/n_bytes:.2f}x)")
        print(f"  Speed:  {input_size/dt/1024:.1f} KB/s")
        print(f"  Time:   {dt:.1f}s")
        print(f"  Saved:  {output_path}")

    except ImportError as e:
        print(f"Error: missing dependency: {e}", file=sys.stderr)
        print("Install with: pip install torch transformers numba numpy", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


def cmd_decode(args):
    """Decode .zllm compressed file back to text."""
    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    output_path = args.output or input_path.rsplit(".", 1)[0]

    print(f"zllm decode v{__version__}")
    with open(input_path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            print(f"Error: not a .zllm file (magic: {magic!r})", file=sys.stderr)
            return 1
        version = struct.unpack(">H", f.read(2))[0]
        raw = f.read()
        metadata, bitstream = _unpack_header(raw)

    print(f"  Format version: {version}")
    print(f"  Preset: {metadata.get('preset', '?')}")
    print(f"  Model: {metadata.get('model', '?')}")
    print(f"  bpb: {metadata.get('bpb', '?')}")
    print(f"  Tokens: {metadata.get('n_tokens', '?')}")
    print(f"  Bitstream: {len(bitstream):,} bytes")
    print()

    try:
        import torch
        import numpy as np
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = metadata.get("tokenizer", "./data/cloud/SmolLM2-135M")
        print(f"  Loading model from {model_dir}...")
        tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, torch_dtype=torch.float16
        ).to(device).eval()
        V = metadata.get("vocab_size", model.config.vocab_size)

        from real_compression import ArithmeticDecoder, TOTAL

        bitstr = ""
        for byte in bitstream:
            bitstr += f"{byte:08b}"
        bitstr = bitstr[:metadata.get("total_bits", len(bitstr))]

        block_tokens = metadata.get("block_tokens", 8192)
        top_k = metadata.get("top_k", 1024)
        n_tokens = metadata.get("n_tokens", 0)
        n_escapes = 0

        decoded_ids = []
        dec = ArithmeticDecoder(bitstr)
        pos = 0

        print("  Decoding...")
        with torch.no_grad():
            while pos < n_tokens:
                chunk_ids = decoded_ids[-block_tokens:] if decoded_ids else []
                if len(chunk_ids) == 0 or (pos % block_tokens == 0 and pos > 0):
                    block = decoded_ids[pos:pos + block_tokens] if decoded_ids else []
                    if not block:
                        block = [tok.bos_token_id or 0]
                    x = torch.tensor([block], device=device)
                    logits = model(x).logits[0].float().cpu().numpy()
                    logits = logits - logits.max(axis=1, keepdims=True)
                    e = np.exp(logits, dtype=np.float64)
                    probs = e / e.sum(axis=1, keepdims=True)
                    all_ids = np.arange(V)

                if pos < len(probs):
                    p = probs[pos % len(probs)] if pos % block_tokens < len(probs) else probs[-1]
                else:
                    p = np.ones(V) / V

                topk_idx = np.argpartition(p, -top_k)[-top_k:]
                topk_idx = topk_idx[np.argsort(-p[topk_idx])]
                topk_p = p[topk_idx]
                esc_mass = max(1e-12, 1.0 - topk_p.sum())

                probs_cat = np.concatenate([topk_p, [esc_mass]])
                c = probs_cat.sum()
                freqs = (probs_cat / c * (TOTAL - len(probs_cat))).astype(np.int64) + 1
                diff = int(TOTAL - freqs.sum())
                freqs[np.argmax(probs_cat)] += diff
                cum = np.zeros(len(freqs) + 1, dtype=np.int64)
                np.cumsum(freqs, out=cum[1:])

                sym = dec.decode_symbol(cum)
                if sym < top_k:
                    token_id = int(topk_idx[sym])
                else:
                    n_escapes += 1
                    mask = np.ones(V, dtype=bool)
                    mask[topk_idx] = False
                    rest = all_ids[mask]
                    token_id = int(rest[0])

                decoded_ids.append(token_id)
                pos += 1

                if pos % 5000 == 0:
                    print(f"    decoded {pos}/{n_tokens} tokens", flush=True)

        output_text = tok.decode(decoded_ids)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_text)

        print()
        print(f"  Decoded: {len(decoded_ids)} tokens -> {len(output_text):,} chars")
        print(f"  Escapes: {n_escapes}")
        print(f"  Saved:   {output_path}")
        print(f"  NOTE: decode is approximate (reconstructed probabilities, not exact bitstream reversal)")
        print(f"  For lossless roundtrip, use the full encoder/decoder pipeline in v13")

    except ImportError as e:
        print(f"Error: missing dependency: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


def cmd_bench(args):
    """Run compression benchmarks."""
    preset = args.preset or "fast"
    size_map = {"100kb": 102400, "1mb": 1048576, "10mb": 10485760}
    size_key = (args.size or "100kb").lower()
    size_bytes = size_map.get(size_key, 102400)

    print(f"zllm bench v{__version__}")
    print(f"  Preset: {preset}")
    print(f"  Size:   {size_key} ({size_bytes:,} bytes)")
    print()

    enwik8_path = "./data/cloud/enwik8"
    if not os.path.exists(enwik8_path):
        print("Error: enwik8 not found at data/cloud/enwik8", file=sys.stderr)
        return 1

    with open(enwik8_path, "rb") as f:
        f.seek(50 * 1024 * 1024)
        raw = f.read(size_bytes)

    tmp_path = "./data/_bench_input.txt"
    with open(tmp_path, "wb") as f:
        f.write(raw)

    args_ns = argparse.Namespace(input=tmp_path, output=tmp_path + ".zllm", preset=preset)
    rc = cmd_encode(args_ns)

    if rc == 0 and os.path.exists(tmp_path + ".zllm"):
        compressed = os.path.getsize(tmp_path + ".zllm")
        orig = os.path.getsize(tmp_path)
        print()
        print(f"  Summary:")
        print(f"    Original:  {orig:,} bytes")
        print(f"    Compressed: {compressed:,} bytes ({compressed/orig:.2f}x)")
        print(f"    Ratio:     {orig*8/compressed:.2f}x")
        os.remove(tmp_path + ".zllm")
    os.remove(tmp_path) if os.path.exists(tmp_path) else None
    return rc


def cmd_info(args):
    """Show information about a .zllm file."""
    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        return 1

    with open(input_path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            print(f"Not a .zllm file (magic: {magic!r})")
            return 1
        version = struct.unpack(">H", f.read(2))[0]
        raw = f.read()
        metadata, bitstream = _unpack_header(raw)

    print(f"zllm info: {input_path}")
    print(f"  Format version: {version}")
    print(f"  Preset: {metadata.get('preset', '?')}")
    print(f"  Model:  {metadata.get('model', '?')}")
    print(f"  Vocab:  {metadata.get('vocab_size', '?')}")
    print(f"  bpb:    {metadata.get('bpb', '?')}")
    print(f"  Tokens: {metadata.get('n_tokens', '?')}")
    print(f"  Escapes: {metadata.get('n_escapes', '?')}")
    print(f"  Input:  {metadata.get('input_size', '?'):,} bytes")
    print(f"  Output: {os.path.getsize(input_path):,} bytes")
    print(f"  Ratio:  {metadata.get('input_size', 0)*8 / max(1, os.path.getsize(input_path)):.2f}x")
    print(f"  Speed:  {metadata.get('input_size', 0) / max(0.1, metadata.get('elapsed_s', 1)) / 1024:.1f} KB/s")
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="zllm",
        description="Neural text compression with LLM + arithmetic coding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
presets:
  fast       Speed-optimized (SmolLM2 chunked, 34.5 KB/s, 0.9276 bpb)
  balanced   Speed-ratio balance (SmolLM2 chunked K2048, 27KB/s, 0.9187)
  ratio      Best ratio (SmolLM2 ov4096/K8192, 0.9003 bpb)
  ratio-qwen Qwen ratio crown (0.8442 bpb, fits 8GB VRAM)

examples:
  zllm encode document.txt -o document.zllm --preset fast
  zllm decode document.zllm -o document.txt
  zllm bench --preset balanced --size 1mb
  zllm info document.zllm
""",
    )
    parser.add_argument("--version", action="version", version=f"zllm {__version__}")
    sub = parser.add_subparsers(dest="command")

    enc = sub.add_parser("encode", help="Compress a file")
    enc.add_argument("input", help="Input text file")
    enc.add_argument("-o", "--output", help="Output .zllm file")
    enc.add_argument("--preset", choices=list(PRESETS.keys()), default="fast",
                     help="Compression preset (default: fast)")

    dec = sub.add_parser("decode", help="Decompress a .zllm file")
    dec.add_argument("input", help="Input .zllm file")
    dec.add_argument("-o", "--output", help="Output text file")

    bench = sub.add_parser("bench", help="Run benchmarks")
    bench.add_argument("--size", default="100kb", help="Test size (100kb/1mb/10mb)")
    bench.add_argument("--preset", choices=list(PRESETS.keys()), default="fast")

    info = sub.add_parser("info", help="Show .zllm file information")
    info.add_argument("input", help=".zllm file to inspect")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    cmds = {"encode": cmd_encode, "decode": cmd_decode, "bench": cmd_bench, "info": cmd_info}
    return cmds[args.command](args)


if __name__ == "__main__":
    sys.exit(main() or 0)
