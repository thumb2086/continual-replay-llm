"""Test HF Inference free tier logprobs."""
import sys, math
sys.stdout.reconfigure(encoding="utf-8")
from huggingface_hub import InferenceClient

client = InferenceClient()
tests = [
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "microsoft/Phi-3-mini-128k-instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "HuggingFaceTB/SmolLM-135M-Instruct",
    "google/gemma-2-2b-it",
]

for model in tests:
    try:
        r = client.chat_completion(
            model=model,
            messages=[{"role":"user","content":"The quick brown fox"}],
            max_tokens=5, temperature=0, logprobs=True, top_logprobs=5
        )
        has_lp = r.choices[0].logprobs is not None
        reply = r.choices[0].message.content or ""
        
        if has_lp:
            n = len(r.choices[0].logprobs.content)
            ent = 0
            for t in r.choices[0].logprobs.content:
                probs = [math.exp(x.logprob) for x in (t.top_logprobs or [])]
                ent += -sum(p*math.log2(max(p,1e-12)) for p in probs if p > 0)
            ent /= max(1, n)
            print(f"  {model}: OK! logprobs={n} tokens E={ent:.2f} reply=[{reply[:30]}]")
        else:
            print(f"  {model}: reply=[{reply[:30]}] no logprobs")
    except Exception as e:
        print(f"  {model}: {str(e)[:80]}")
