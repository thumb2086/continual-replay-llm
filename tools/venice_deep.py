"""Deep test: stability + prompt_logprobs on Venice winners."""
import sys, time, json, subprocess, math
sys.stdout.reconfigure(encoding="utf-8")

KEY = "sk-OO1h9qlD63ALWiHasX37Jg4xZMoQnz2IbjSM2h0P4VoHdG1h"

def ps_chat(model, messages, max_tokens=5, logprobs=True, top_logprobs=10, prompt_logprobs=False):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = top_logprobs
    if prompt_logprobs:
        body["prompt_logprobs"] = True
    body_str = json.dumps(body)
    ps = f"""[Console]::OutputEncoding=[System.Text.Encoding]::UTF8
$h=@{{"Authorization"="Bearer {KEY}";"Content-Type"="application/json"}}
try {{ (Invoke-WebRequest -Uri 'https://api.banana2556.com/v1/chat/completions' -Method POST -Headers $h -Body ([System.Text.Encoding]::UTF8.GetBytes('{body_str.replace("'","''")}')) -UseBasicParsing -TimeoutSec 30).Content }} catch {{ '{{"err":"'+$_.Exception.Message+'"}}' }}"""
    r = subprocess.run(["powershell","-NoProfile","-Command",ps], capture_output=True, timeout=45, encoding="utf-8", errors="replace")
    try: return json.loads(r.stdout.strip()) if r.stdout.strip() else {"err":"empty"}
    except: return {"err":r.stdout[:200]}

WINNERS = [
    "meta/llama-3.2-11b-vision-instruct",
    "nvidia/riva-translate-4b-instruct-v2",
    "openai/gpt-oss-20b",
    "poolside/laguna-xs-2.1",
    "deepseek-ai/deepseek-v4-flash-0731",
]

print("=" * 70)
print("VENICE: Deep test on 5 winners")
print("=" * 70)

# Test 1: Stability (10 consecutive calls)
print("\n--- Test 1: Stability (10 calls each) ---")
for model in WINNERS:
    ok = 0
    for i in range(10):
        r = ps_chat(model, [{"role":"user","content":"hi"}], max_tokens=2)
        if "err" not in r and r.get("choices"):
            ok += 1
        time.sleep(2)
    print(f"  {model}: {ok}/10 success")

# Test 2: Full logprobs with top-10
print("\n--- Test 2: Full logprobs (top-10) ---")
for model in WINNERS:
    r = ps_chat(model, [{"role":"user","content":"The quick brown fox jumps over the lazy dog. What comes next?"}], max_tokens=5, top_logprobs=10)
    if "err" not in r:
        lp = (r.get("choices",[{}])[0].get("logprobs") or {})
        content = lp.get("content", [])
        if content:
            print(f"\n  {model}:")
            for tok in content:
                prob = math.exp(tok.get("logprob", 0))
                print(f"    '{tok.get('token','?')}' logprob={tok.get('logprob',0):.4f} prob={prob*100:.1f}%")
                top = tok.get("top_logprobs", [])
                for t in top[:5]:
                    tp = math.exp(t.get("logprob", 0))
                    print(f"      '{t.get('token','?')}': {t.get('logprob',0):.4f} ({tp*100:.1f}%)")
        else:
            print(f"  {model}: logprobs present but empty content")
    else:
        print(f"  {model}: {r['err'][:50]}")
    time.sleep(3)

# Test 3: prompt_logprobs (input token logprobs)
print("\n--- Test 3: prompt_logprobs (input tokens) ---")
for model in WINNERS:
    r = ps_chat(model, [{"role":"user","content":"The quick brown fox"}], max_tokens=1, prompt_logprobs=True, top_logprobs=5)
    if "err" not in r:
        pplp = r.get("prompt_logprobs")
        olp = r.get("choices",[{}])[0].get("logprobs")
        print(f"  {model}: prompt_logprobs={'YES' if pplp else 'NO'}, output_logprobs={'YES' if olp else 'NO'}")
        if pplp:
            print(f"    prompt_logprobs type: {type(pplp)}, len: {len(pplp)}")
    else:
        print(f"  {model}: {r['err'][:50]}")
    time.sleep(3)
