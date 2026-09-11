"""Verify downloads: unzip enwik8, locate cosmopedia parquet."""

import zipfile
import glob
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

# verify enwik8
z = zipfile.ZipFile("./data/cloud/enwik8.zip")
for info in z.infolist():
    print("zip contains:", info.filename, info.file_size, "bytes")
z.extractall("./data/cloud/")
print("extracted OK")

# find cosmopedia parquet recursively
fs = glob.glob(
    r"C:\Users\CPXru\Desktop\thumb\大拇哥實驗室\cortexflow\data\cosmopedia-100k\**\*.parquet",
    recursive=True,
)
print("cosmopedia parquet files:", len(fs))
for f in fs:
    mb = os.path.getsize(f) / 1024 / 1024
    print("  ", round(mb, 1), "MB")
