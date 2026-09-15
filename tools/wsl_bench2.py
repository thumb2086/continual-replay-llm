#!/usr/bin/env python3
"""Quick speed benchmark: SmolLM2 on WSL CUDA, B8192 CHUNK=4096/2048."""
import sys, os, time
sys.stdout.reconfigure(encoding="utf-8")

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
os.environ["ENWIK8_OFFSET_MB"] = "50"
os.environ["ENWIK8_KB"] = "100"
os.environ["BIGRAM_CONF"] = "10"
os.environ["TRIGRAM_CONF"] = "3"
os.environ["FLOOR_FRAC"] = "1e-6"
os.environ["USE_CACHE_S2"] = "0"
os.environ["USE_FP16_XFER"] = "1"
os.environ["MODEL_DIR"] = os.path.expanduser("~/models/SmolLM2-135M")

# Run v13 engine
exec(open("/mnt/c/Users/CPXru/Desktop/thumb/大拇哥實驗室/online-llm/ensemble/bpe_ensemble_v13.py").read())
