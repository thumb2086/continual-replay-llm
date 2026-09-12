"""Inspect cosmopedia parquet structure (paths handled inside Python)."""

import glob
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

BASE = r"C:\Users\CPXru\Desktop\thumb\大拇哥實驗室"
pattern = os.path.join(BASE, "cortexflow", "data", "cosmopedia-100k", "**", "*.parquet")
files = sorted(glob.glob(pattern, recursive=True))
print("parquet files:", len(files))
for f in files:
    print("  ", round(os.path.getsize(f) / 1024 / 1024, 1), "MB")

import pandas as pd

df = pd.read_parquet(files[0])
print("rows:", len(df))
print("columns:", list(df.columns))
for c in df.columns:
    print(f"  {c}: dtype={df[c].dtype}")
    v = df[c].iloc[0]
    s = str(v)
    print(f"    sample[0:{min(200, len(s))}]: {s[:200]}")
