"""Full Venice model survey: all models, all keys, find best logprobs teacher."""
import sys, time, json, subprocess
sys.stdout.reconfigure(encoding="utf-8")

# The 2 actual keys (user pasted; the string has 2 keys separated by a space)
KEYS = [
    "sk-OO1h9qlD63ALWiHasX37Jg4xZMoQnz2IbjSM2h0P4VoHdG1h",
    "sk-v5l8xjgZ5AVSSiDdbvINca1b55jk5O7R85fuNhHP932AwB63sk-8GKvvhcKogUGAqabeeTDaRJTgnTvrvYnaVkRRRHBvaJr8042sk-L9qpITtjzCXHcdzEzut3lozhNgch5zgAcuMVgYjY8SgQAmnE",
]

def ps(method, url, headers, body=None, timeout=20):
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

def test_model(model, key):
    body = {"model": model, "messages": [{"role": "user", "content": "The quick brown fox jumps over"}],
            "max_tokens": 5, "temperature": 0, "logprobs": True, "top_logprobs": 10}
    h = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    r = ps("POST", "https://api.banana2556.com/v1/chat/completions", h, body, timeout=25)
    if "err" in r:
        return {"model": model, "ok": False, "error": r["err"][:60]}
    choices = r.get("choices", [])
    if not choices:
        return {"model": model, "ok": False, "error": "no choices"}
    lp = choices[0].get("logprobs")
    reply = (choices[0].get("message") or {}).get("content") or ""
    n_tok = len((lp or {}).get("content", [])) if lp else 0
    # Compute avg entropy
    avg_ent = 0
    if lp and lp.get("content"):
        for t in lp["content"]:
            probs = [math.exp(x.get("logprob", 0)) for x in t.get("top_logprobs", [])]
            if probs:
                avg_ent += -sum(p * math.log2(max(p, 1e-12)) for p in probs if p > 0)
        avg_ent /= max(1, len(lp["content"]))
    return {"model": model, "ok": True, "logprobs": bool(lp), "tokens": n_tok,
            "reply": reply[:40] if reply else "(empty)", "avg_entropy": round(avg_ent, 2)}

import math

print("=" * 70)
print("VENICE: Complete model survey with logprobs")
print("=" * 70)

# Step 1: Get all models from all keys
all_models = {}
for i, key in enumerate(KEYS):
    h = {"Authorization": f"Bearer {key}"}
    r = ps("GET", "https://api.banana2556.com/v1/models", h)
    if "data" in r:
        for m in r["data"]:
            if m["id"] not in all_models:
                all_models[m["id"]] = key
        print(f"  Key {i+1} (...{key[-8:]}): {len(r['data'])} models")
    else:
        print(f"  Key {i+1}: {r.get('err', '?')[:60]}")

models = sorted(all_models.keys())
print(f"\nTotal unique: {len(models)}")
for m in models:
    print(f"  {m}")

# Step 2: Test each
print(f"\n{'='*70}")
print(f"Testing {len(models)} models for logprobs + quality...")
print(f"{'='*70}")

results = []
for i, model in enumerate(models):
    key = all_models[model]
    r = test_model(model, key)
    results.append(r)
    status = "OK+LP" if r.get("logprobs") else ("OK" if r.get("ok") else "FAIL")
    ent = f" E={r.get('avg_entropy',0):.1f}" if r.get("logprobs") else ""
    err = f" {r.get('error','')[:40]}" if not r.get("ok") else ""
    print(f"  [{i+1:2d}/{len(models)}] {model[:45]:45s} {status:8s}{ent}{err}")
    time.sleep(3)

# Summary
print(f"\n{'='*70}")
print("RESULTS")
print(f"{'='*70}")
ok_lp = [r for r in results if r.get("logprobs")]
ok_no_lp = [r for r in results if r.get("ok") and not r.get("logprobs")]
failed = [r for r in results if not r.get("ok")]

print(f"  Total: {len(results)}")
print(f"  With logprobs: {len(ok_lp)}")
print(f"  OK, no logprobs: {len(ok_no_lp)}")
print(f"  Failed: {len(failed)}")

if ok_lp:
    print(f"\n*** WINNERS (sorted by entropy = information content) ***")
    for r in sorted(ok_lp, key=lambda x: -x.get("avg_entropy", 0)):
        print(f"  {r['model']:50s} E={r.get('avg_entropy',0):.2f} tokens={r['tokens']}")

# Save results
with open("data/venice_survey.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"\nSaved to data/venice_survey.json")
