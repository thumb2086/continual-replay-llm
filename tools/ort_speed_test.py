"""Export SmolLM2 to ONNX and benchmark with ORT CUDA EP."""
import sys, time, torch, os, numpy as np
sys.stdout.reconfigure(encoding="utf-8")

os.makedirs("./onnx-smol135m", exist_ok=True)
model_dir = "./data/cloud/SmolLM2-135M"

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, torch_dtype=torch.float16).eval()
V = model.config.vocab_size

# Step 1: Export to ONNX
print("[1] Exporting SmolLM2 to ONNX...")
t0 = time.time()
dummy = torch.randint(0, V, (1, 512), device="cpu")
torch.onnx.export(
    model, (dummy,),
    "./onnx-smol135m/model.onnx",
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {1: "seq_len"}, "logits": {1: "seq_len"}},
    opset_version=17,
    do_constant_folding=True,
)
onnx_time = time.time() - t0
onnx_size = os.path.getsize("./onnx-smol135m/model.onnx") / 1024 / 1024
print(f"  Export: {onnx_time:.1f}s, size: {onnx_size:.1f}MB")

# Step 2: Load with ORT CUDA EP
print("[2] Loading ONNX Runtime session...")
t0 = time.time()
import onnxruntime as ort
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.intra_op_num_threads = 4
sess = ort.InferenceSession("./onnx-smol135m/model.onnx", sess_options=so,
                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print(f"  Loaded in {time.time()-t0:.1f}s")
print(f"  Provider: {sess.get_providers()[0]}")

# Step 3: Warmup
print("[3] Warmup...")
dummy_cu = torch.randint(0, V, (1, 1024), device="cpu").numpy()
for _ in range(3):
    sess.run(None, {"input_ids": dummy_cu})
torch.cuda.synchronize()

# Step 4: Benchmark vs PyTorch
print("[4] Benchmark...")
N = 10
for seq_len in [1024, 2048, 4096, 8192]:
    dummy_onnx = np.random.randint(0, V, (1, seq_len), dtype=np.int64)
    dummy_pt = torch.tensor(dummy_onnx, device="cuda")

    # ORT
    ort_times = []
    for _ in range(N):
        t0 = time.time()
        sess.run(None, {"input_ids": dummy_onnx})
        ort_times.append(time.time() - t0)
    ort_avg = sum(ort_times) / len(ort_times)

    # PyTorch
    model.cuda()
    pt_times = []
    with torch.no_grad():
        for _ in range(N):
            torch.cuda.synchronize()
            t0 = time.time()
            model(dummy_pt)
            torch.cuda.synchronize()
            pt_times.append(time.time() - t0)
    pt_avg = sum(pt_times) / len(pt_times)

    speedup = pt_avg / ort_avg
    print(f"  seq={seq_len:5d}: ORT={ort_avg:.3f}s PyTorch={pt_avg:.3f}s speedup={speedup:.2f}x")
