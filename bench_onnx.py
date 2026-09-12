"""ONNX Runtime export benchmark for the S model."""

import torch
import numpy as np
import sys
import time
import os

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from train_v3 import OnlineLLMV3, CharTokenizer, load_grouped_texts

CFG_S = dict(embed_dim=128, hidden_dim=256, n_layers=2, n_heads=4)
ONNX_PATH = "./data/s_model.onnx"


class ExportWrapper(torch.nn.Module):
    """Wrap forward to return logits only (no memory side effects)."""

    def __init__(self, model):
        super().__init__()
        self.embedding = model.embedding
        self.transformer = model.transformer
        self.head = model.head

    def forward(self, x):
        h = self.embedding(x)
        h = self.transformer(h)
        return self.head(h)


def main():
    device = torch.device("cuda")
    torch.manual_seed(0)

    base_path = r"C:\Users\CPXru\Music\playlist-admin\podcasts"
    texts_a, _, _, _ = load_grouped_texts(base_path, max_files_per_group=150)
    tok = CharTokenizer()
    tok.fit(texts_a + texts_a)
    print(f"Vocab: {tok.vocab_size}")

    # Rebuild S-bigdata weights? We only have eval numbers; retrain briefly?
    # For a pure SPEED comparison, random weights are fine (same FLOPs).
    print("\nBuilding S model (random weights OK for speed test)...")
    model = OnlineLLMV3(vocab_size=tok.vocab_size, **CFG_S).to(device)
    model.eval()
    wrapper = ExportWrapper(model).to(device).eval()

    xb = torch.randint(0, tok.vocab_size, (32, 128), device=device)

    # PyTorch fp16 baseline
    for _ in range(5):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            wrapper(xb)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(100):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            wrapper(xb)
    torch.cuda.synchronize()
    pt_time = (time.time() - t) / 100
    print(f"PyTorch fp16 fwd (32x128): {pt_time*1000:.2f} ms")

    # Export to ONNX (dynamo exporter decomposes the fused fastpath op)
    print("\nExporting to ONNX...")
    with torch.no_grad():
        onnx_program = torch.onnx.dynamo_export(
            wrapper,
            torch.randint(0, tok.vocab_size, (32, 128), device=device),
        )
    onnx_program.save(ONNX_PATH)
    print(f"  saved, {os.path.getsize(ONNX_PATH)/1024:.0f} KB")

    import onnxruntime as ort

    for ep_name, opts in [("CUDA", None), ("TensorRT", None)]:
        try:
            if opts is None and ep_name == "TensorRT":
                sess = ort.InferenceSession(
                    ONNX_PATH,
                    providers=[("TensorrtExecutionProvider",
                                {"trt_fp16_enable": True}),
                               "CUDAExecutionProvider"],
                )
            else:
                sess = ort.InferenceSession(ONNX_PATH, providers=["CUDAExecutionProvider"])
            x = np.random.randint(0, tok.vocab_size, (32, 128)).astype(np.int64)
            for _ in range(5):
                sess.run(None, {"input_ids": x})
            t = time.time()
            for _ in range(100):
                sess.run(None, {"input_ids": x})
            dt = (time.time() - t) / 100
            print(f"  ORT {ep_name}: {dt*1000:.2f} ms "
                  f"({pt_time/dt:.2f}x vs PyTorch)")
        except Exception as e:
            print(f"  ORT {ep_name} failed: {str(e)[:200]}")


if __name__ == "__main__":
    main()
