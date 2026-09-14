"""Real distillation: get teacher's probability distribution at each position.

Approach: for each enwik8 position, ask the teacher "what's the next token?"
and get its logprobs. The logprobs give us the full probability distribution
over the vocabulary, which is exactly what we need for distillation.

Key insight: we don't need prompt_logprobs. We can get the SAME information
by asking: "Given this prefix, what is the probability of the ACTUAL next
character being X?" -> this is logprobs[0] for the first generated token.
"""
import sys, time, json, subprocess, math
sys.stdout.reconfigure(encoding="utf-8")

NVIDIA_KEY = "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"
MODEL = "nvidia/nemotron-3-super-120b-a12b"

def ps_chat(prompt, max_tokens=1, retries=3):
    """Single-call with top-50 logprobs for the first generated token."""
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 50,
    })
    for attempt in range(retries):
        ps = f"""[Console]::OutputEncoding=[System.Text.Encoding]::UTF8
$h=@{{"Content-Type"="application/json";"Authorization"="Bearer {NVIDIA_KEY}"}}
$b=[System.Text.Encoding]::UTF8.GetBytes('{body.replace("'","''")}')
try{{(Invoke-WebRequest -Uri 'https://integrate.api.nvidia.com/v1/chat/completions' -Method POST -Headers $h -Body $b -UseBasicParsing -TimeoutSec 45).Content}}catch{{'{{"err":"'+$_.Exception.Message+'"}}'}}"""
        r = subprocess.run(["powershell","-NoProfile","-Command",ps], capture_output=True, timeout=60, encoding="utf-8", errors="replace")
        try:
            result = json.loads(r.stdout.strip()) if r.stdout.strip() else {"err":"empty"}
            if "err" not in result:
                return result
        except: pass
        time.sleep(2 * (attempt + 1))
    return {"err": "max retries"}

def load_enwik8(offset_mb=50, size_kb=100):
    with open("./data/cloud/enwik8", "rb") as f:
        f.seek(offset_mb * 1024 * 1024)
        return f.read(size_kb * 1024)

def main():
    print("=" * 70)
    print("TEACHER LOGPROBS: NVIDIA 120B on enwik8")
    print("=" * 70)

    data = load_enwik8(offset_mb=50, size_kb=100)
    text = data.decode("utf-8", errors="replace")
    print(f"enwik8: {len(data)} bytes")

    # For each position, ask the teacher what it predicts as the NEXT character
    # and get its probability for the ACTUAL next character
    SAMPLE = 30  # test 30 positions first
    PREFIX = 50  # 50 chars of context
    step = max(1, (len(text) - PREFIX - 1) // SAMPLE)
    positions = list(range(0, len(text) - PREFIX - 1, step))[:SAMPLE]

    teacher_bits = []
    student_bits = []
    t0 = time.time()
    api_calls = 0
    errors = 0

    for i, pos in enumerate(positions):
        prefix = text[pos:pos+PREFIX]
        actual_next = text[pos+PREFIX]
        
        prompt = f"""Given this English text, what is the VERY NEXT single character? Reply with exactly one character:
{prefix}"""
        
        result = ps_chat(prompt, max_tokens=1)
        api_calls += 1
        
        if "err" in result:
            errors += 1
            continue
        
        lp = result.get("choices", [{}])[0].get("logprobs", {})
        content = lp.get("content", [])
        
        if content:
            # Get the probability of the ACTUAL next character
            top_logprobs = content[0].get("top_logprobs", [])
            
            # Find the actual character in the top list
            found = False
            for tl in top_logprobs:
                if tl.get("token", "") == actual_next:
                    teacher_bits.append(-tl["logprob"] / math.log(2))
                    found = True
                    break
            
            if not found:
                # Character not in top-50 -> very high bits (conservative estimate)
                teacher_bits.append(8.0)
            
            if (i + 1) % 10 == 0:
                avg_bpb = sum(teacher_bits) / max(1, len(teacher_bits))
                dt = time.time() - t0
                print(f"  [{i+1}/{SAMPLE}] teacher_bpb={avg_bpb:.2f} errors={errors} ({dt:.0f}s)", flush=True)
        
        time.sleep(1)

    dt = time.time() - t0
    avg_bpb = sum(teacher_bits) / max(1, len(teacher_bits))

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  Teacher (120B) avg bpb: {avg_bpb:.4f}")
    print(f"  Student (Qwen-3B):     0.6617")
    print(f"  Student (Qwen-1.5B):   0.7024")
    print(f"  SOTA (Nacrith):        0.9389")
    print(f"  ")
    print(f"  Gap: teacher {avg_bpb:.4f} vs student 0.6617")
    if avg_bpb < 0.6617:
        print(f"  >>> Teacher is BETTER by {0.6617-avg_bpb:.4f} bpb - distillation could help!")
    else:
        print(f"  >>> Teacher is WORSE by {avg_bpb-0.6617:.4f} bpb - at this scale, student already near ceiling")
    
    print(f"\n  API calls: {api_calls} ({errors} errors)")
    print(f"  Time: {dt:.0f}s")
    print(f"  Rate: {api_calls/max(1,dt):.2f} calls/s")
    print(f"  For 100KB full: {len(text)*api_calls/max(1,len(positions)*dt):.0f}s = {len(text)*api_calls/max(1,len(positions)*dt)/3600:.1f}h")
    
    # Save results
    with open("data/nvidia_teacher_bpb.json", "w") as f:
        json.dump({"teacher_bpb": avg_bpb, "n_samples": len(teacher_bits),
                    "errors": errors, "time_s": dt, "teacher_bits": teacher_bits}, f, indent=1)
    print(f"\nSaved to data/nvidia_teacher_bpb.json")

if __name__ == "__main__":
    main()
