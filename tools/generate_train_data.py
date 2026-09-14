"""Generate training data from Groq Qwen3.8-27B for distillation.

Strategy: generate diverse English text across 50 topics.
Each topic: ~2KB of text. Total: ~100KB (one batch at a time to respect rate limits).
Uses PowerShell subprocess (urllib blocked by WAF on Windows).
"""
import os, sys, time, json, subprocess
sys.stdout.reconfigure(encoding="utf-8")

API_KEY = open(".env", encoding="utf-8-sig").read().split("=", 1)[1].strip()

TOPICS = [
    "history of computing", "quantum mechanics explained", "evolution of mammals",
    "ancient Rome daily life", "how vaccines work", "deep sea creatures",
    "history of mathematics", "climate change mechanisms", "biography of Newton",
    "the printing press revolution", "how nuclear reactors work", "ancient Egypt",
    "the Renaissance period", "machine learning basics", "the Industrial Revolution",
    "plate tectonics", "history of photography", "how antibiotics work",
    "the Silk Road trade", "astronomy and galaxies", "the French Revolution",
    "how the internet works", "ancient Greece philosophy", "the Black Death plague",
    "DNA and genetics basics", "the Age of Exploration", "how bridges are built",
    "the Cold War period", "ocean currents and climate", "the Scientific Revolution",
    "how engines work", "the Silk Road cultures", "metamorphic rocks explained",
    "the history of writing", "how telescopes work", "the Agricultural Revolution",
    "electricity generation methods", "the Victorian era", "how earthquakes happen",
    "the history of medicine", "deforestation and ecosystems", "the Space Race",
    "how lasers work", "medieval castles and fortifications", "the theory of relativity",
    "how the eye sees", "the Bronze Age collapse", "entropy and thermodynamics",
    "the history of music", "how volcanoes form", "democracy in ancient Athens",
]

def ps_chat(prompt, max_tokens=300, retries=3):
    body = json.dumps({"model":"qwen/qwen3.8-27b","messages":[{"role":"user","content":prompt}],"max_tokens":max_tokens,"temperature":0.7})
    ps = f"""
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$h = @{{"Authorization"="Bearer {API_KEY}";"Content-Type"="application/json"}}
try {{
    $r = Invoke-WebRequest -Uri 'https://api.groq.com/openai/v1/chat/completions' -Headers $h -Method POST -Body ([System.Text.Encoding]::UTF8.GetBytes('{body.replace("'","''")}')) -UseBasicParsing -TimeoutSec 60
    $r.Content
}} catch {{
    '{{"error":"' + $_.Exception.Message + '"}}'
}}
"""
    for attempt in range(retries):
        try:
            r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                               capture_output=True, timeout=90, encoding="utf-8", errors="replace")
            result = json.loads(r.stdout.strip()) if r.stdout.strip() else {"error": "empty"}
            if "error" not in result or "429" not in str(result.get("error", "")):
                return result
            time.sleep(5 * (attempt + 1))  # exponential backoff
        except Exception as e:
            time.sleep(3)
    return {"error": "max retries"}

def generate_dataset(target_bytes=100*1024, output_path="data/groq_train.txt"):
    """Generate diverse text until target_bytes reached."""
    all_text = []
    total_bytes = 0
    
    for i, topic in enumerate(TOPICS):
        if total_bytes >= target_bytes:
            break
        
        prompt = (
            f"Write a detailed, factual encyclopedia-style article about '{topic}'. "
            f"Write at least 500 words. Use plain text, no markdown formatting. "
            f"Start directly with content, no headers."
        )
        
        result = ps_chat(prompt, max_tokens=400)
        
        if "error" in result:
            print(f"  [{i+1}] ERROR: {result['error'][:80]}")
            time.sleep(3)
            continue
        
        text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        text_bytes = len(text.encode("utf-8"))
        total_bytes += text_bytes
        all_text.append(text)
        
        print(f"  [{i+1}/{len(TOPICS)}] {topic[:30]}: {text_bytes} bytes (total: {total_bytes})", flush=True)
        time.sleep(3)  # rate limit: Groq free tier ~30 req/min
    
    # Write to file
    combined = "\n\n".join(all_text)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(combined)
    
    print(f"\nGenerated {total_bytes} bytes to {output_path}")
    return total_bytes

if __name__ == "__main__":
    print("=" * 60)
    print("Groq Training Data Generation")
    print("=" * 60)
    t0 = time.time()
    nbytes = generate_dataset(target_bytes=100*1024)
    dt = time.time() - t0
    print(f"Time: {dt:.1f}s ({nbytes/dt:.0f} bytes/s)")
