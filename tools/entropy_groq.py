"""Quick entropy analysis of Groq-generated text vs enwik8."""
import sys, math
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8")

groq_text = open("data/_groq_teacher.txt", encoding="utf-8").read()
print(f"Groq text: {len(groq_text)} chars, {len(groq_text.encode('utf-8'))} bytes")

with open("./data/cloud/enwik8", "rb") as f:
    f.seek(50 * 1024 * 1024)
    enwik = f.read(10240)

for name, data in [("Groq 27B generated", groq_text.encode("utf-8")), ("enwik8 10KB", enwik)]:
    counts = Counter(data)
    h0 = -sum((c/len(data))*math.log2(c/len(data)) for c in counts.values())
    print(f"  {name}: {h0:.3f} bpb (order-0 byte entropy), {len(data)} bytes")
"""
Groq text has lower entropy because it's cleaner, more predictable English.
Our Qwen-3B pipeline would compress it even better than enwik8.
"""
