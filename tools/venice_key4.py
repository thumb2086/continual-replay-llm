"""Test Key 4's powerful models for logprobs."""
import sys, time, json, subprocess, math
sys.stdout.reconfigure(encoding="utf-8")

KEY4 = "L9qpITtjzCXHcdzEzut3lozhNgch5zgAcuMVgYjY8SgQAmnE"

MODELS = [
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-v3.2-speciale",
    "deepseek-v4-flash",
    "gemini-3.1-pro",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gpt-5.6-reasoning",
    "gpt-5.6-sol",
    "grok-4.5",
    "grok-4.6",
    "openai/gpt-5.4-mini",
    "openai/gpt-5.4-nano",
    "openai/gpt-oss-120b",
    "qwen/qwen3-vl-235b-a22b-instruct",
    "minimaxai/minimax-m3",
    "z-ai/glm-4-32b",
    "z-ai/glm-4.5-air",
    "z-ai/glm-4.7-flash",
    "xiaomi/mimo-v2-flash",
    "xiaomi/mimo-v2.5",
    "Olmo-3.1-32B-Instruct",
]

def ps(method, url, headers, body=None, timeout=25):
    h = ";".join(f'"{k}"="{v}"' for k, v in headers.items())
    body_json = json.dumps(body) if body else ""
    if body_json:
        ps_cmd = f"""[Console]::OutputEncoding=[System.Text.Encoding]::UTF8
$h=@{{{h}}}
$b=[System.Text.Encoding]::UTF8.GetBytes('{body_json.replace("'","''")}')
try{{(Invoke-WebRequest -Uri '{url}' -Method {method} -Headers $h -Body $b -UseBasicParsing -TimeoutSec {timeout}).Content}}catch{{'{{"err":"'+$_.Exception.Message+'"}}'}}"""
    else:
        ps_cmd = f"""[Console]::OutputEncoding=[System.Text.Encoding]::UTF8
$h=@{{{h}}}
try{{(Invoke-WebRequest -Uri '{url}' -Method {method} -Headers $h -UseBasicParsing -TimeoutSec {timeout}).Content}}catch{{'{{"err":"'+$_.Exception.Message+'"}}'}}"""
    r = subprocess.run(["powershell","-NoProfile","-Command",ps_cmd], capture_output=True, timeout=timeout+15, encoding="utf-8", errors="replace")
    try: return json.loads(r.stdout.strip()) if r.stdout.strip() else {"err":"empty"}
    except: return {"err":r.stdout[:150]}

print("=" * 70)
print("VENICE Key 4: Test all powerful models for logprobs")
print("=" * 70)

winners = []

for model in MODELS:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "The quick brown fox jumps over the lazy"}],
        "max_tokens": 5,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 10,
    }
    h = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY4}"}
    r = ps("POST", "https://api.banana2556.com/v1/chat/completions", h, body, timeout=25)
    
    if "err" in r:
        print(f"  {model[:50]:50s} FAIL: {r['err'][:40]}")
    else:
        choices = r.get("choices", [])
        if choices:
            lp = choices[0].get("logprobs")
            reply = (choices[0].get("message") or {}).get("content") or ""
            has_lp = lp is not None
            n = len(lp.get("content", [])) if lp else 0
            
            # Compute avg entropy
            avg_ent = 0
            if lp and lp.get("content"):
                for t in lp["content"]:
                    probs = [math.exp(x.get("logprob", 0)) for x in t.get("top_logprobs", []) if x.get("logprob", 0) > -50]
                    if probs:
                        avg_ent += -sum(p * math.log2(max(p, 1e-12)) for p in probs if p > 0)
                avg_ent /= max(1, len(lp["content"]))
            
            status = f"OK+LP E={avg_ent:.2f}" if has_lp else f"OK (no LP)"
            print(f"  {model[:50]:50s} {status:15s} reply=[{(reply or '')[:25]}]")
            
            if has_lp and avg_ent > 0.05:
                winners.append({"model": model, "entropy": avg_ent, "tokens": n, "reply": (reply or "")[:30]})
                # Show first token's top-5
                if lp.get("content"):
                    print(f"    First token top-5:")
                    for t in lp["content"][0].get("top_logprobs", [])[:5]:
                        print(f"      '{t.get('token','?')}': {t.get('logprob',0):.4f} ({math.exp(t.get('logprob',0))*100:.1f}%)")
        else:
            print(f"  {model[:50]:50s} FAIL: no choices")
    
    time.sleep(3)

print(f"\n{'='*70}")
print(f"WINNERS (with logprobs + entropy > 0.05): {len(winners)}")
print(f"{'='*70}")
for w in sorted(winners, key=lambda x: -x["entropy"]):
    print(f"  {w['model']:50s} E={w['entropy']:.2f} tokens={w['tokens']}")

# Save
with open("data/venice_key4_survey.json", "w") as f:
    json.dump(winners, f, indent=1)
print(f"\nSaved {len(winners)} winners to data/venice_key4_survey.json")
