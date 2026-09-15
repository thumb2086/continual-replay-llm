"""Test torch.compile with Triton on WSL."""
import sys, os, time, torch
sys.stdout.reconfigure(encoding="utf-8")

os.environ["MODEL_DIR"] = os.path.expanduser("~/models/SmolLM2-135M")
model_dir = os.path.expanduser("~/models/SmolLM2-135M")

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, dtype=torch.float16).cuda().eval()
V = model.config.vocab_size

print("[1] torch.compile mode='default'...")
t0 = time.time()
model_compiled = torch.compile(model, mode="default")
x = torch.randint(0, V, (1, 4096), device="cuda")
with torch.no_grad():
    for _ in range(3):
        model_compiled(x)
    torch.cuda.synchronize()
print(f"  Compile: {time.time()-t0:.1f}s")

print("[2] Benchmark compiled vs eager...")
for seq_len in [1024, 2048, 4096, 8192]:
    x = torch.randint(0, V, (1, seq_len), device="cuda")
    # Eager
    eager_times = []
    with torch.no_grad():
        for _ in range(5):
            torch.cuda.synchronize()
            t0 = time.time()
            model(x)
            torch.cuda.synchronize()
            eager_times.append(time.time() - t0)
    eager_avg = sum(eager_times) / len(eager_times)
    # Compiled
    compiled_times = []
    with torch.no_grad():
        for _ in range(5):
            torch.cuda.synchronize()
            t0 = time.time()
            model_compiled(x)
            torch.cuda.synchronize()
            compiled_times.append(time.time() - t0)
    compiled_avg = sum(compiled_times) / len(compiled_times)
    speedup = eager_avg / compiled_avg
    print(f"  seq={seq_len:5d}: eager={eager_avg:.3f}s compiled={compiled_avg:.3f}s speedup={speedup:.2f}x")
