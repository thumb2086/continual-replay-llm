"""Distill S from a regularized (weight-decay) M teacher."""

import torch
import numpy as np
import sys
import json

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from train_v3 import OnlineLLMV3, CharTokenizer, load_grouped_texts, create_chunks
from compare_ablation import make_batch
from showdown import distill, measure_frozen, CFG_S, CFG_M


def main():
    torch.manual_seed(123)
    np.random.seed(123)
    device = torch.device("cuda")
    base_path = r"C:\Users\CPXru\Music\playlist-admin\podcasts"
    texts_a, texts_b, _, _ = load_grouped_texts(base_path, max_files_per_group=25)
    tok = CharTokenizer()
    tok.fit(texts_a + texts_b)
    chunks_a = create_chunks(texts_a, chunk_len=128)
    np.random.shuffle(chunks_a)
    with open("./data/frozen_stream.json", encoding="utf-8") as f:
        stream = json.load(f)["chunks"]

    print("--- M teacher WITH weight decay 0.01 ---")
    t = OnlineLLMV3(vocab_size=tok.vocab_size, **CFG_M).to(device)
    opt = torch.optim.AdamW(t.parameters(), lr=3e-4, weight_decay=0.01)
    t.train()
    for epoch in range(10):
        tot, nb = 0.0, 0
        for i in range(0, len(chunks_a) - 8, 8):
            tot += t.pretrain_step(make_batch(tok, chunks_a[i:i + 8], device), opt)
            nb += 1
        print(f"  Epoch {epoch+1}: loss={tot/nb:.4f}", flush=True)

    print("--- Distill S from regularized teacher ---")
    s = OnlineLLMV3(vocab_size=tok.vocab_size, **CFG_S).to(device)
    distill(s, t, tok, chunks_a, device, epochs=10)
    fz = measure_frozen(s, tok, stream, device)
    print(f"S-distilled-v2 frozen: {fz['bpc']:.3f} bpc")
    print("reference: S-scratch 7.466, S-distilled-v1 7.636")

    with open("./data/distill_v2.json", "w") as f:
        json.dump({"s_distilled_v2_frozen_bpc": fz["bpc"]}, f, indent=2)


if __name__ == "__main__":
    main()
