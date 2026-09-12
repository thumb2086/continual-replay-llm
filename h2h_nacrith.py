"""Head-to-head: Nacrith (third_party) vs our v7 on the SAME 100KB mid-slice.

Runs Nacrith's ParallelNeuralCompressor (1 worker, local BF16 GGUF) on
enwik8[50MB:50MB+100KB] -- the exact bytes behind our smollm2_*_off50
numbers -- and records bpb + KB/s + byte-exact roundtrip.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "third_party", "nacrith"))
from parallel.compressor import ParallelNeuralCompressor

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    with open(os.path.join(HERE, "data", "cloud", "enwik8"), "rb") as f:
        f.seek(50 * 1024 * 1024)
        raw = f.read(100 * 1024)
    print(f"input bytes: {len(raw)}", flush=True)

    gguf = os.path.join(HERE, "third_party", "smollm2-135m-bf16.gguf")
    pc = ParallelNeuralCompressor(n_workers=1, gguf_path=gguf, verbose=True)

    t0 = time.time()
    blob = pc.compress_bytes(raw)
    dt = time.time() - t0
    bpb = len(blob) * 8 / len(raw)
    print(f"compressed bytes: {len(blob)}, bpb={bpb:.4f}, "
          f"time={dt:.1f}s, {len(raw)/1024/dt:.2f} KB/s", flush=True)

    t1 = time.time()
    back = pc.decompress(blob)
    print(f"roundtrip exact: {back == raw} "
          f"(decompress {time.time()-t1:.1f}s)", flush=True)

    out = {"system": "nacrith-third-party", "workers": 1,
           "n_bytes": len(raw), "compressed_bytes": len(blob),
           "bpb": bpb, "elapsed_s": dt,
           "kbs": len(raw) / 1024 / dt,
           "roundtrip_exact": bool(back == raw)}
    with open(os.path.join(HERE, "data", "h2h_nacrith.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("saved data/h2h_nacrith.json", flush=True)


if __name__ == "__main__":
    main()
