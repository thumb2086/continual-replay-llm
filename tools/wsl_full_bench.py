#!/usr/bin/env python3
"""WSL comprehensive benchmark: SmolLM2 all sizes."""
import sys, os, time, json
sys.stdout.reconfigure(encoding="utf-8")

MODEL_DIR = os.path.expanduser("~/models/SmolLM2-135M")
ENWIK8 = os.path.expanduser("~/data/cloud/enwik8")
RESULTS = []

def run_test(name, env_overrides, kb_list):
    """Run a single test configuration."""
    base_env = {
        "BLOCK_TOKENS": "8192",
        "CHUNK_PRE": "4096",
        "CHUNK_HEAD": "2048",
        "SPARSE_BLK": "1",
        "BLEND_BT_MIN": "20",
        "BLEND_TT_MIN": "7",
        "TOP_K": "1024",
        "PREFILTER": "2048",
        "OVERLAP": "0",
        "BIGRAM_LAMBDA": "0.99",
        "ENWIK8_OFFSET_MB": "50",
        "BIGRAM_CONF": "10",
        "TRIGRAM_CONF": "3",
        "FLOOR_FRAC": "1e-6",
        "USE_CACHE_S2": "0",
        "USE_FP16_XFER": "1",
        "MODEL_DIR": MODEL_DIR,
    }
    base_env.update(env_overrides)

    for kb in kb_list:
        os.environ.update(base_env)
        os.environ["ENWIK8_KB"] = str(kb)

        print(f"\n{'='*60}")
        print(f"TEST: {name} | {kb}KB")
        print(f"{'='*60}", flush=True)

        t_start = time.time()
        try:
            exec(open("/mnt/c/Users/CPXru/Desktop/thumb/大拇哥實驗室/online-llm/ensemble/bpe_ensemble_v13.py").read())
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            RESULTS.append({"name": name, "kb": kb, "error": str(e)})
            continue
        elapsed = time.time() - t_start

        # Parse results from the last JSON
        import glob as g
        jsons = sorted(g.glob("data/smollm2_ensemble_v13_*.json"), key=os.path.getmtime)
        if jsons:
            with open(jsons[-1]) as f:
                r = json.load(f)
            bpb = r.get("bits_per_byte", r.get("bpb", 0))
            verified = r.get("verified", 0)
            segments = r.get("segments", 0)
            result = {
                "name": name,
                "kb": kb,
                "bpb": round(bpb, 4),
                "time_s": round(elapsed, 1),
                "kb_s": round(kb / elapsed, 1) if elapsed > 0 else 0,
                "verified": verified,
                "segments": segments,
            }
            RESULTS.append(result)
            print(f"  RESULT: bpb={bpb:.4f} time={elapsed:.1f}s speed={kb/elapsed:.1f}KB/s verified={verified}", flush=True)

        # Save intermediate results
        with open("logs/wsl_all_benchmarks.json", "w") as f:
            json.dump(RESULTS, f, indent=2)

# === SmolLM2 Baseline configs ===
configs = [
    ("SmolLM2 baseline (B8192 K1024)", {}, [100, 1024, 10240, 102400]),
    ("SmolLM2 chunk K2048 (B8192)", {"TOP_K": "2048"}, [100, 1024]),
    ("SmolLM2 chunk K4096 (B8192)", {"TOP_K": "4096"}, [100, 1024]),
    ("SmolLM2 B4096 K1024", {"BLOCK_TOKENS": "4096", "CHUNK_PRE": "2048", "CHUNK_HEAD": "1024"}, [100, 1024]),
    ("SmolLM2 B4096 K2048", {"BLOCK_TOKENS": "4096", "CHUNK_PRE": "2048", "CHUNK_HEAD": "1024", "TOP_K": "2048"}, [100, 1024]),
]

print(f"Starting WSL benchmarks at {time.strftime('%H:%M:%S')}")
print(f"Model: {MODEL_DIR}")
print(f"Data: {ENWIK8}")

for name, overrides, sizes in configs:
    run_test(name, overrides, sizes)

print(f"\n{'='*60}")
print("ALL WSL BENCHMARKS COMPLETE")
print(f"{'='*60}")
for r in RESULTS:
    if "error" not in r:
        print(f"  {r['name']:40s} {r['kb']:6d}KB  bpb={r['bpb']:.4f}  {r['time_s']:7.1f}s  {r['kb_s']:6.1f}KB/s")
    else:
        print(f"  {r['name']:40s} {r['kb']:6d}KB  ERROR: {r['error'][:50]}")

with open("logs/wsl_all_benchmarks.json", "w") as f:
    json.dump(RESULTS, f, indent=2)
print(f"\nSaved to logs/wsl_all_benchmarks.json")
