"""Check cosmopedia UNK rate under podcast tokenizer + sample sizes."""

import glob
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

import pandas as pd
from train_v3 import CharTokenizer, load_grouped_texts

BASE = r"C:\Users\CPXru\Desktop\thumb\大拇哥實驗室"
pattern = os.path.join(BASE, "cortexflow", "data", "cosmopedia-100k", "**", "*.parquet")
files = sorted(glob.glob(pattern, recursive=True))

# Rebuild podcast tokenizer identically
base_path = r"C:\Users\CPXru\Music\playlist-admin\podcasts"
texts_a, texts_b, _, _ = load_grouped_texts(base_path, max_files_per_group=150)
tok = CharTokenizer()
tok.fit(texts_a + texts_b)
print("podcast vocab:", tok.vocab_size)

# Sample cosmopedia
total_chars = 0
unk_chars = 0
n_samples = 0
total_len = 0
for f in files:
    df = pd.read_parquet(f, columns=["text"])
    for t in df["text"].tolist():
        if not isinstance(t, str) or len(t) < 100:
            continue
        n_samples += 1
        total_len += len(t)
        # UNK check on first 2000 chars per sample (speed)
        head = t[:2000]
        total_chars += len(head)
        unk_chars += sum(1 for c in head if c not in tok.char2idx)
        if n_samples >= 2000:
            break
    if n_samples >= 2000:
        break

print(f"samples checked: {n_samples}")
print(f"avg sample length: {total_len/n_samples:.0f} chars")
print(f"UNK rate: {unk_chars}/{total_chars} = {unk_chars/total_chars*100:.2f}%")
