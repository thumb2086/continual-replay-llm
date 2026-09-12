"""Download cloud datasets (within 12GB/day budget)."""

import os
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

BASE = "./data/cloud"


def download_url(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  exists, skip: {dest} ({os.path.getsize(dest)/1024**2:.1f} MB)")
        return dest
    print(f"  downloading {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  saved: {dest} ({os.path.getsize(dest)/1024**2:.1f} MB)")
    return dest


def main():
    os.makedirs(BASE, exist_ok=True)

    # 1. enwik8 (100MB, THE compression benchmark)
    print("[1/3] enwik8...")
    download_url("http://mattmahoney.net/dc/enwik8.zip", f"{BASE}/enwik8.zip")

    # 2. TinyStories valid set (small, eval use)
    print("[2/3] TinyStories valid...")
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download("roneneldan/TinyStories",
                            filename="TinyStoriesV2-GPT4-valid.txt",
                            repo_type="dataset", local_dir=BASE)
        print(f"  saved: {p}")
    except Exception as e:
        print(f"  failed: {str(e)[:200]}")

    # 3. Cosmopedia-100k: already local, just verify
    print("[3/3] Cosmopedia-100k (local copy)...")
    import glob
    local = glob.glob(r"C:\Users\CPXru\Desktop\thumb\大拇哥實驗室\cortexflow\data\cosmopedia-100k\*.parquet")
    print(f"  found {len(local)} parquet files locally, no download needed")
    for f in local:
        print(f"    {os.path.basename(f)} {os.path.getsize(f)/1024**2:.1f} MB")


if __name__ == "__main__":
    main()
