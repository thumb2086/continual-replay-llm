"""Test remaining Venice models for logprobs."""
import sys, time, json, subprocess
sys.stdout.reconfigure(encoding="utf-8")

KEY = "sk-OO1h9qlD63ALWiHasX37Jg4xZMoQnz2IbjSM2h0P4VoHdG1h"
BODY = json.dumps({"model":"PLACEHOLDER","messages":[{"role":"user","content":"The quick brown fox"}],"max_tokens":3,"temperature":0,"logprobs":True,"top_logprobs":3})

def ps_chat(model):
    body = BODY.replace("PLACEHOLDER", model)
    ps = f"""[Console]::OutputEncoding=[System.Text.Encoding]::UTF8
$h=@{{"Authorization"="Bearer {KEY}";"Content-Type"="application/json"}}
try {{ (Invoke-WebRequest -Uri 'https://api.banana2556.com/v1/chat/completions' -Method POST -Headers $h -Body ([System.Text.Encoding]::UTF8.GetBytes('{body.replace("'","''")}')) -UseBasicParsing -TimeoutSec 20).Content }} catch {{ '{{"err":"'+$_.Exception.Message+'"}}' }}"""
    r = subprocess.run(["powershell","-NoProfile","-Command",ps], capture_output=True, timeout=35, encoding="utf-8", errors="replace")
    try: return json.loads(r.stdout.strip()) if r.stdout.strip() else {"err":"empty"}
    except: return {"err":r.stdout[:100]}

remaining = [
    "openai/gpt-oss-120b","openai/gpt-oss-20b","poolside/laguna-xs-2.1",
    "qwen/qwen3-next-80b-a3b-instruct","qwen/qwen3.5-122b-a10b",
    "qwen/qwen3.5-397b-a17b","stepfun-ai/step-3.5-flash",
    "stepfun-ai/step-3.7-flash","thinkingmachines/inkling","z-ai/glm-5.2",
    "deepseek-ai/deepseek-v4-flash-0731","meta/llama-3.3-70b-instruct",
    "nvidia/nemotron-4-340b-instruct","nvidia/nemotron-3-ultra-550b-a55b",
    "moonshotai/kimi-k3","moonshotai/kimi-k2.6",
]

print("=== Venice API: remaining model logprobs test ===")
winners = []
for m in remaining:
    r = ps_chat(m)
    if "err" in r:
        print(f"  {m}: {r['err'][:50]}")
    else:
        choices = r.get("choices", [{}])
        reply = ((choices[0].get("message") or {}).get("content") or "") if choices else ""
        has_lp = (choices[0].get("logprobs") is not None) if choices else False
        n = len((choices[0].get("logprobs") or {}).get("content", [])) if has_lp else 0
        print(f"  {m}: reply=[{(reply or '')[:30]}] logprobs={has_lp} tokens={n}")
        if has_lp:
            winners.append({"model": m, "tokens": n, "reply": (reply or "")[:30]})
    time.sleep(4)

print(f"\n{'='*50}")
print(f"WINNERS: {len(winners)}")
for w in winners:
    print(f"  {w['model']} ({w['tokens']} tokens)")
