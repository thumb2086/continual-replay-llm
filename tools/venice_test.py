"""Test ALL Venice API models for logprobs support across all key groups."""
import sys, time, json, subprocess
sys.stdout.reconfigure(encoding="utf-8")

KEYS = [
    "sk-OO1h9qlD63ALWiHasX37Jg4xZMoQnz2IbjSM2h0P4VoHdG1h",
    "sk-v5l8xjgZ5AVSSiDdbvINca1b55jk5O7R85fuNhHP932AwB63sk-8GKvvhcKogUGAqabeeTDaRJTgnTvrvYnaVkRRRHBvaJr8042sk-L9qpITtjzCXHcdzEzut3lozhNgch5zgAcuMVgYjY8SgQAmnE",
]

def ps_json(method, url, headers, body=None, timeout=20):
    h = ";".join(f'"{k}"="{v}"' for k,v in headers.items())
    if body:
        ps = f"""
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$h = @{{{h}}}
$b = [System.Text.Encoding]::UTF8.GetBytes('{json.dumps(body).replace("'","''")}')
try {{ (Invoke-WebRequest -Uri '{url}' -Method {method} -Headers $h -Body $b -UseBasicParsing -TimeoutSec {timeout}).Content }} catch {{ '{{"error":"' + $_.Exception.Message + '"}}' }}
"""
    else:
        ps = f"""
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$h = @{{{h}}}
try {{ (Invoke-WebRequest -Uri '{url}' -Method {method} -Headers $h -UseBasicParsing -TimeoutSec {timeout}).Content }} catch {{ '{{"error":"' + $_.Exception.Message + '"}}' }}
"""
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       capture_output=True, timeout=timeout+10, encoding="utf-8", errors="replace")
    try:
        return json.loads(r.stdout.strip()) if r.stdout.strip() else {"error": "empty"}
    except:
        return {"error": r.stdout[:200]}

# Step 1: Get all models from all keys
print("=" * 70)
print("VENICE API: Full model survey + logprobs test")
print("=" * 70)

all_models = set()
for i, key in enumerate(KEYS):
    h = {"Authorization": f"Bearer {key}"}
    r = ps_json("GET", "https://api.banana2556.com/v1/models", h)
    if "data" in r:
        for m in r["data"]:
            all_models.add(m["id"])
        print(f"  Key {i+1}: {len(r['data'])} models")
    else:
        print(f"  Key {i+1}: {r.get('error', 'unknown')[:80]}")

models = sorted(all_models)
print(f"\nTotal unique models: {len(models)}")
for m in models:
    print(f"  {m}")

# Step 2: Test each model for logprobs
print(f"\n{'='*70}")
print(f"Testing {len(models)} models for logprobs...")
print(f"{'='*70}")

results = []
key_idx = 0

for model in models:
    key = KEYS[key_idx % len(KEYS)]
    key_idx += 1
    
    h = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "The quick brown fox"}],
        "max_tokens": 3,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 3,
    }
    
    r = ps_json("POST", "https://api.banana2556.com/v1/chat/completions", h, body, timeout=25)
    
    if "error" in r:
        status = r["error"][:40]
        print(f"  {model}: {status}")
        results.append({"model": model, "status": "error", "error": status})
    else:
        reply = r.get("choices", [{}])[0].get("message", {}).get("content", "")
        has_lp = r.get("choices", [{}])[0].get("logprobs") is not None
        n_tokens = len(r.get("choices", [{}])[0].get("logprobs", {}).get("content", [])) if has_lp else 0
        print(f"  {model}: OK reply=[{reply[:30]}] logprobs={has_lp} tokens={n_tokens}")
        results.append({"model": model, "status": "ok", "reply": reply[:50], "logprobs": has_lp, "tokens": n_tokens})
    
    time.sleep(3)

# Summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
ok = [r for r in results if r["status"] == "ok"]
with_lp = [r for r in results if r.get("logprobs")]
no_lp = [r for r in results if r["status"] == "ok" and not r.get("logprobs")]
failed = [r for r in results if r["status"] == "error"]

print(f"Total: {len(results)}")
print(f"  OK: {len(ok)}")
print(f"  OK + logprobs: {len(with_lp)}")
print(f"  OK, no logprobs: {len(no_lp)}")
print(f"  Failed: {len(failed)}")

if with_lp:
    print(f"\n*** WINNERS (logprobs supported) ***")
    for r in with_lp:
        print(f"  {r['model']} ({r['tokens']} tokens)")

if no_lp:
    print(f"\nWorks but no logprobs: {[r['model'] for r in no_lp]}")
