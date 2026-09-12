"""Inspect podcast directory contents precisely."""

import os
import sys
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

base = Path(r"C:\Users\CPXru\Music\playlist-admin\podcasts")
exts = Counter()
dirinfo = []
for d in sorted(base.iterdir()):
    if not d.is_dir():
        continue
    ffiles = [f for f in d.rglob("*") if f.is_file()]
    for f in ffiles:
        exts[f.suffix.lower()] += 1
    totalsz = sum(f.stat().st_size for f in ffiles)
    dirinfo.append((len(ffiles), totalsz))

print("=== subdirs (file count, total MB) ===")
for (n, sz) in dirinfo:
    print(f"{n:5d} files  {sz/1024/1024:8.1f} MB")
print()
print("=== extensions ===")
for e, c in exts.most_common(15):
    label = e if e else "(none)"
    print(f"{label:8s} {c}")
print()
print("=== total files ===")
print(sum(c for _, c in exts.most_common()))
