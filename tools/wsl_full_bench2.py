#!/usr/bin/env python3
"""WSL comprehensive benchmark: SmolLM2 all sizes via subprocess."""
import sys, os, time, json, subprocess
sys.stdout.reconfigure(encoding="utf-8")

MODEL_DIR = os.path.expanduser("~/models/SmolLM2-135M")
ENGINE = "/mnt/c/Users/CPXru/Desktop/thumb/大拇哥實驗室/online-llm/ensemble/bpe_ensemble_v13.py"
RESULTS = []

def run_test(name, env_overrides, kb):
    """Run a single test via subprocess."""
    env = os.environ.copy()
    base = {
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
        "ENWIK8_KB": str(kb),
        "BIGRAM_CONF": "10",
        "TRIGRAM_CONF": "3",
        "FLOOR_FRAC": "1e-6",
        "USE_CACHE_S2": "0",
        "USE_FP16_XFER": "1",
        "MODEL_DIR": MODEL_DIR,
    }
    base.update(env_overrides)
    env.update(base)

    print(f"  {name:40s} {kb:6d}KB ... ", end="", flush=True)
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, "-u", ENGINE],
        capture_output=True, text=True, timeout=600, env=env,
        cwd="/mnt/c/Users/CPXru/Desktop/thumb/大拇哥實驗室/online-llm"
    )
    elapsed = time.time() - t0
    out = r.stdout + r.stderr

    # Parse
    bpb = time_s = verified = segments = 0
    for line in out.split("\n"):
        if "bits/byte:" in line:
            bpb = float(line.split("bits/byte:")[1].strip())
        if "Time:" in line and "s" in line:
            time_s = float(line.split("Time:")[1].split("s")[0].strip())
        if "Verified:" in line:
            verified = int(line.split("Verified:")[1].split()[0])
        if "Segments:" in line:
            segments = int(line.split("Segments:")[1].split(",")[0].strip())

    speed = kb / elapsed if elapsed > 0 else 0
    result = {
        "name": name, "kb": kb, "bpb": round(bpb, 4),
        "time_s": round(time_s if time_s > 0 else elapsed, 1),
        "kb_s": round(speed, 1), "verified": verified, "segments": segments,
    }
    RESULTS.append(result)
    status = f"OK bpb={bpb:.4f} {elapsed:.1f}s {speed:.1f}KB/s v={verified}" if verified > 0 else f"FAIL"
    print(status, flush=True)

    with open("logs/wsl_all_benchmarks.json", "w") as f:
        json.dump(RESULTS, f, indent=2)

# === Config matrix ===
configs = [
    ("SmolLM2 B8192 K1024", {}, [100, 1024, 10240, 102400]),
    ("SmolLM2 B8192 K2048", {"TOP_K": "2048"}, [100, 1024]),
    ("SmolLM2 B8192 K4096", {"TOP_K": "4096"}, [100, 1024]),
    ("SmolLM2 B4096 K1024", {"BLOCK_TOKENS": "4096", "CHUNK_PRE": "2048", "CHUNK_HEAD": "1024"}, [100, 1024]),
    ("SmolLM2 B4096 K2048", {"BLOCK_TOKENS": "4096", "CHUNK_PRE": "2048", "CHUNK_HEAD": "1024", "TOP_K": "2048"}, [100, 1024]),
]

print(f"WSL benchmarks start: {time.strftime('%H:%M:%S')}")
print(f"Model: {MODEL_DIR}\n")

for name, overrides, sizes in configs:
    for kb in sizes:
        run_test(name, overrides, kb)

print(f"\n{'='*70}")
print("WSL BENCHMARK RESULTS")
print(f"{'='*70}")
print(f"{'Config':40s} {'Size':>6s} {'bpb':>7s} {'Time':>7s} {'Speed':>8s} {'Verified':>8s}")
print(f"{'-'*70}")
for r in RESULTS:
    print(f"{r['name']:40s} {r['kb']:5d}KB {r['bpb']:7.4f} {r['time_s']:6.1f}s {r['kb_s']:6.1f}KB/s {r['verified']:>6d}")
