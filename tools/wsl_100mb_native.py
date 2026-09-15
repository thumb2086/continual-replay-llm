#!/usr/bin/env python3
"""WSL 100MB native benchmark - runs inside WSL with ext4 paths."""
import sys, os, time, json
sys.stdout.reconfigure(encoding="utf-8")

# Set all env vars BEFORE importing engine
os.environ["BLOCK_TOKENS"] = "8192"
os.environ["CHUNK_PRE"] = "4096"
os.environ["CHUNK_HEAD"] = "2048"
os.environ["SPARSE_BLK"] = "1"
os.environ["BLEND_BT_MIN"] = "20"
os.environ["BLEND_TT_MIN"] = "7"
os.environ["TOP_K"] = "1024"
os.environ["PREFILTER"] = "2048"
os.environ["OVERLAP"] = "0"
os.environ["BIGRAM_LAMBDA"] = "0.99"
os.environ["ENWIK8_OFFSET_MB"] = "0"
os.environ["ENWIK8_KB"] = "102400"
os.environ["BIGRAM_CONF"] = "10"
os.environ["TRIGRAM_CONF"] = "3"
os.environ["FLOOR_FRAC"] = "1e-6"
os.environ["USE_CACHE_S2"] = "0"
os.environ["USE_FP16_XFER"] = "1"
os.environ["MODEL_DIR"] = os.path.expanduser("~/models/SmolLM2-135M")

print(f"WSL 100MB native test starting at {time.strftime('%H:%M:%S')}")
print(f"Model: {os.environ['MODEL_DIR']}")
print(f"Enwik8: ~/data/cloud/enwik8")
print(flush=True)

# Import and run engine
sys.path.insert(0, "/mnt/c/Users/CPXru/Desktop/thumb/大拇哥實驗室/online-llm")
exec(open("/mnt/c/Users/CPXru/Desktop/thumb/大拇哥實驗室/online-llm/ensemble/bpe_ensemble_v13.py").read())
