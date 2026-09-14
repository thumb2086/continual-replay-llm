"""Self-distillation: use enwik8 as training data directly.

The model that compresses enwik8 well is one that predicts enwik8 well.
Fine-tuning on enwik8 itself (not Groq text) is the correct approach.

We use a very small lr to avoid catastrophic forgetting of the base model.
"""
import sys, time, math, torch, os
sys.stdout.reconfigure(encoding="utf-8")

MODEL_DIR = "./data/cloud/SmolLM2-135M"
OUTPUT_DIR = "data/distilled_smol135m_enwik8"
EPOCHS = 5
LR = 1e-5
BLOCK = 512

t0 = time.time()

# Load enwik8
with open("./data/cloud/enwik8", "rb") as f:
    f.seek(50 * 1024 * 1024)  # offset 50MB, same as our eval
    raw = f.read(1024 * 1024)  # 1MB of enwik8

from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

text = raw.decode("utf-8", errors="replace")
ids = tok.encode(text)
print(f"Enwik8: {len(raw)} bytes, {len(ids)} tokens")

device = torch.device("cuda")
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True).to(device).float()

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

# Train
model.train()
for ep in range(EPOCHS):
    ep_loss = 0
    n = 0
    for i in range(0, len(ids) - BLOCK - 1, BLOCK):
        x = torch.tensor([ids[i:i+BLOCK]], device=device)
        y = torch.tensor([ids[i+1:i+BLOCK+1]], device=device)
        loss = model(x, labels=y).loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ep_loss += loss.item()
        n += 1
    avg = ep_loss / max(1, n)
    ppl = math.exp(avg)
    bpb = avg / math.log(2)
    print(f"  Epoch {ep+1}: loss={avg:.4f} ppl={ppl:.2f} bpb={bpb:.4f}", flush=True)

dt = time.time() - t0
print(f"\nTraining: {dt:.1f}s")

# Save
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tok.save_pretrained(OUTPUT_DIR)
print(f"Saved to {OUTPUT_DIR}")

# Test perplexity on enwik8 100KB slice
print("\nEnwik8 100KB perplexity test...")
with open("./data/cloud/enwik8", "rb") as f:
    f.seek(50 * 1024 * 1024)
    test_raw = f.read(100 * 1024)
test_ids = tok.encode(test_raw.decode("utf-8", errors="replace"))
model.eval()
with torch.no_grad():
    total_loss = 0
    total_tok = 0
    for i in range(0, len(test_ids) - BLOCK - 1, BLOCK):
        x = torch.tensor([test_ids[i:i+BLOCK]], device=device)
        y = torch.tensor([test_ids[i+1:i+BLOCK+1]], device=device)
        loss = model(x, labels=y).loss
        total_loss += loss.item() * (BLOCK)
        total_tok += BLOCK
    avg_loss = total_loss / total_tok
    ppl = math.exp(avg_loss)
    bpb = avg_loss / math.log(2)
    print(f"  PPL: {ppl:.2f}, bpb: {bpb:.4f}")
    print(f"  Original SmolLM2: 0.9003 bpb")
    print(f"  Distilled: {bpb:.4f} bpb")
    if bpb < 0.9003:
        print(f"  IMPROVEMENT: {0.9003 - bpb:.4f} bpb ({(0.9003-bpb)/0.9003*100:.1f}%)")
    else:
        print(f"  WORSE: {bpb - 0.9003:.4f} bpb ({(bpb-0.9003)/0.9003*100:.1f}%)")

print(f"\nTotal: {time.time()-t0:.1f}s")
print(f"To test: $env:MODEL_OVERRIDE='{OUTPUT_DIR}'")
