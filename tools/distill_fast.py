"""Fast distillation: fine-tune SmolLM2-135M on Groq text (minimal version)."""
import sys, time, math, os, torch
sys.stdout.reconfigure(encoding="utf-8")

MODEL_DIR = "./data/cloud/SmolLM2-135M"
TRAIN_FILE = "data/groq_train.txt"
OUTPUT_DIR = "data/distilled_smol135m"
EPOCHS = 3
LR = 2e-5
BLOCK = 256

t0 = time.time()

# Tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

text = open(TRAIN_FILE, encoding="utf-8").read()
ids = tok.encode(text)
print(f"Text: {len(text)} chars, {len(ids)} tokens")

# Simple dataset
class DS(torch.utils.data.Dataset):
    def __init__(self, ids, bs): self.ids, self.bs = ids, bs
    def __len__(self): return max(0, len(self.ids) - self.bs - 1)
    def __getitem__(self, i):
        x = torch.tensor(self.ids[i:i+self.bs], dtype=torch.long)
        y = torch.tensor(self.ids[i+1:i+self.bs+1], dtype=torch.long)
        return x, y

ds = DS(ids, BLOCK)
loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)
print(f"Dataset: {len(ds)} samples, {len(loader)} batches/epoch")

# Load model
device = torch.device("cuda")
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, dtype=torch.float16).to(device)
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

# Freeze all but last 4 layers + head
for n, p in model.named_parameters():
    p.requires_grad = not any(f"layers.{i}." in n for i in range(26))
nt = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable: {nt:,} ({nt/sum(p.numel() for p in model.parameters())*100:.1f}%)")

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)

# Train
model.train()
for ep in range(EPOCHS):
    ep_loss = 0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        loss = model(x, labels=y).loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ep_loss += loss.item()
        if (i+1) % 10 == 0:
            print(f"  ep{ep+1} step{i+1}: loss={loss.item():.4f}", flush=True)
    print(f"Epoch {ep+1}: avg_loss={ep_loss/len(loader):.4f}")

# Save
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tok.save_pretrained(OUTPUT_DIR)
print(f"\nSaved to {OUTPUT_DIR}")

# Quick perplexity test
with open("./data/cloud/enwik8", "rb") as f:
    f.seek(50 * 1024 * 1024)
    test_ids = tok.encode(f.read(1024).decode("utf-8", errors="replace"))
model.eval()
with torch.no_grad():
    x = torch.tensor([test_ids[:BLOCK]], device=device)
    loss = model(x, labels=x).loss
    ppl = math.exp(loss.item())
    bpb = loss.item() / math.log(2)
print(f"Enwik8 perplexity: {ppl:.2f}, approx bpb: {bpb:.4f}")
print(f"Total time: {time.time()-t0:.1f}s")
