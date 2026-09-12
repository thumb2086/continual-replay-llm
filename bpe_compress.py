"""SmolLM2-135M + arithmetic coding on enwik8 (frozen baseline).

Handles 49K BPE vocab via top-k + escape:
- Stage 1: code top-K token or ESCAPE among K+1 symbols.
- Stage 2 (escape only): uniform coding over remaining V-K tokens.
Both stages use the verified 14-bit arithmetic coder.
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import time
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from transformers import AutoModelForCausalLM, AutoTokenizer
from real_compression import (
    probs_to_freqs, ArithmeticEncoder, ArithmeticDecoder, TOTAL,
)

MODEL_DIR = "./data/cloud/SmolLM2-135M"
TOP_K = int(os.environ.get("TOP_K", "2048"))
BLOCK_TOKENS = 8192

# ---- Ensemble config: blend LLM probs with causal unigram cache ----
# p_mix = LAMBDA * p_llm + (1-LAMBDA) * p_cache, all on GPU.
# Cache counts only previously coded blocks (causal; decoder mirrors).
# LAMBDA=1.0 disables the ensemble (pure LLM).
ENSEMBLE_LAMBDA = 0.9
ENSEMBLE_ALPHA = 1.0

# ---- Phase 2: online adaptation config ----
# "none": frozen baseline. "bias": BitFit-style bias-only updates.
# "full": all parameters. Updates run AFTER coding each block (causal:
# the decoder reproduces them from decoded tokens), in eval mode
# (no dropout) so encoder/decoder stay bit-identical.
ADAPT_MODE = "none"
ADAPT_LR = 1e-5
ADAPT_DTYPE = "fp32"  # fp32 master for update stability (fp16 diverges)
ADAPT_EVERY = 1  # update every N blocks


def stage1_cum(topk_probs, escape_mass):
    """Build K+1 cum array: top-K probs + escape."""
    p = np.concatenate([topk_probs, [escape_mass]])
    return probs_to_freqs(p)[1]


def uniform_cum(n):
    """Uniform cum over n symbols."""
    p = np.ones(n) / n
    return probs_to_freqs(p)[1]


TAIL_GROUP = 16384  # stage-2 groups; each fits the 14-bit precision


def tail_group_rank(rank):
    """Map a tail rank to (group, offset) with fixed group sizes."""
    g = rank // TAIL_GROUP
    return g, rank - g * TAIL_GROUP


def n_tail_groups(n_rest):
    return (n_rest + TAIL_GROUP - 1) // TAIL_GROUP


def main():
    print("=" * 70)
    print("SMOLLM2-135M FROZEN BASELINE ON ENWIK8")
    print("=" * 70)
    device = torch.device("cuda")
    t0 = time.time()

    print("\n[1/4] Loading model + tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, torch_dtype=torch.float16
    ).to(device).eval()
    # For stable online updates, optionally run the whole model in fp32.
    # fp16 updates diverged (overflow -> NaN poisoning), so fp32 master
    # weights are the robust choice; decoder mirrors the same dtype.
    if ADAPT_DTYPE == "fp32" and ADAPT_MODE in ("bias", "full"):
        model = model.float()
        print("  Update dtype: fp32 master weights")
    V = model.config.vocab_size
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Vocab: {V}, params: {n_params:,}")
    print(f"  TOP_K: {TOP_K}, ESC id scheme: stage-2 uniform over rest")
    print(f"  ADAPT: {ADAPT_MODE}, lr={ADAPT_LR}, every={ADAPT_EVERY}")

    # Freeze everything except the adaptation target. LLaMA-style models
    # have no bias params, so "bias" mode targets norm scales instead
    # (same BitFit spirit: tiny trainable count).
    if ADAPT_MODE == "bias":
        for p in model.parameters():
            p.requires_grad_(False)
        for n, p in model.named_parameters():
            if "bias" in n or "norm" in n.lower():
                p.requires_grad_(True)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable: {n_train:,} ({n_train/n_params*100:.2f}%)")
    elif ADAPT_MODE == "full":
        n_train = n_params
        print(f"  Trainable: all {n_train:,} (100%)")
    else:
        n_train = 0
    adapt_opt = None
    scaler = None
    if ADAPT_MODE in ("bias", "full"):
        import torch.optim as _optim
        adapt_opt = _optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=ADAPT_LR, fused=True)
        scaler = torch.amp.GradScaler("cuda")

    print("\n[2/4] Loading enwik8 sample...")
    import os as _os
    _off_mb = int(_os.environ.get("ENWIK8_OFFSET_MB", "0"))
    with open("./data/cloud/enwik8", "rb") as f:
        f.seek(_off_mb * 1024 * 1024)
        raw = f.read(100 * 1024)
    print(f"  Offset: {_off_mb}MB")
    text = raw.decode("utf-8", errors="ignore")
    ids = tok.encode(text)
    print(f"  Chars: {len(text)}, tokens: {len(ids)} "
          f"({len(text)/len(ids):.2f} chars/token)")
    n_bytes = len(text.encode("utf-8"))

    print("\n[3/4] Coding in 2048-token blocks...")
    total_bits, n_escapes = 0, 0
    n_verify_ok, n_verify_fail = 0, 0
    # Deterministic remaining-id order for stage 2
    all_ids = np.arange(V)

    torch.cuda.synchronize()
    t = time.time()
    # Causal unigram cache over BPE tokens (GPU dense vector)
    cache_counts = torch.zeros(V, dtype=torch.float32, device=device)
    cache_total = 0
    with torch.no_grad():
        for b0 in range(0, len(ids), BLOCK_TOKENS):
            block = ids[b0:b0 + BLOCK_TOKENS]
            if len(block) < 2:
                continue
            x = torch.tensor([block], device=device)
            logits = model(x).logits[0].float()  # (T, V) on GPU
            if ENSEMBLE_LAMBDA < 1.0:
                # Ensemble blend on GPU before transfer (no extra PCIe cost)
                probs = torch.softmax(logits, dim=-1)
                p_cache = (cache_counts + ENSEMBLE_ALPHA) / (
                    cache_total + ENSEMBLE_ALPHA * V)
                probs = ENSEMBLE_LAMBDA * probs + (1.0 - ENSEMBLE_LAMBDA) * p_cache
                probs = probs.cpu().numpy()
            else:
                # Original float64 CPU path (bit-identical to baseline runs)
                logits = logits.cpu().numpy()
                logits = logits - logits.max(axis=1, keepdims=True)
                e = np.exp(logits, dtype=np.float64)
                probs = e / e.sum(axis=1, keepdims=True)
            # NOTE: cache update happens AFTER this block is coded (below),
            # so the decoder (which hasn't seen this block yet) stays in sync.

            for t_idx in range(len(block) - 1):
                p = probs[t_idx]
                target = block[t_idx + 1]
                # Top-k + escape
                topk_idx = np.argpartition(p, -TOP_K)[-TOP_K:]
                # sort top-k descending for determinism
                topk_idx = topk_idx[np.argsort(-p[topk_idx])]
                topk_p = p[topk_idx]
                esc_mass = max(1e-12, 1.0 - topk_p.sum())
                cum1 = stage1_cum(topk_p, esc_mass)

                if target in set(topk_idx.tolist()):
                    s1 = int(np.where(topk_idx == target)[0][0])
                    enc = ArithmeticEncoder(store=True)
                    enc.encode_symbol(cum1, s1)
                    bitstr, nb = enc.finish()
                    bitstr2, nb2 = None, 0
                else:
                    n_escapes += 1
                    s1 = TOP_K  # escape id
                    enc = ArithmeticEncoder(store=True)
                    enc.encode_symbol(cum1, s1)
                    bitstr, nb = enc.finish()
                    # Stage 2: hierarchical uniform over non-topk ids.
                    # Tail (~47K) exceeds 14-bit precision, so split into
                    # fixed 16K groups: code group, then offset in group.
                    mask = np.ones(V, dtype=bool)
                    mask[topk_idx] = False
                    rest = all_ids[mask]
                    rank = int(np.where(rest == target)[0][0])
                    n_groups = n_tail_groups(len(rest))
                    g, off = tail_group_rank(rank)
                    cum_g = uniform_cum(n_groups)
                    enc_g = ArithmeticEncoder(store=True)
                    enc_g.encode_symbol(cum_g, g)
                    bitstr_g, nb_g = enc_g.finish()
                    gsize = TAIL_GROUP if g < n_groups - 1 else len(rest) - g * TAIL_GROUP
                    cum_o = uniform_cum(gsize)
                    enc_o = ArithmeticEncoder(store=True)
                    enc_o.encode_symbol(cum_o, off)
                    bitstr_o, nb_o = enc_o.finish()
                    bitstr2, nb2 = (bitstr_g, bitstr_o), (nb_g, nb_o)
                total_bits += nb + (nb2[0] + nb2[1] if isinstance(nb2, tuple) else nb2)

                # Verify first 200 positions overall (fresh decoder per stream)
                if n_verify_ok + n_verify_fail < 200:
                    dec1 = ArithmeticDecoder(bitstr)
                    d1 = dec1.decode_symbol(cum1)
                    if d1 == TOP_K:
                        assert isinstance(bitstr2, tuple)
                        mask = np.ones(V, dtype=bool)
                        mask[topk_idx] = False
                        rest = all_ids[mask]
                        n_groups = n_tail_groups(len(rest))
                        cum_g = uniform_cum(n_groups)
                        dec_g = ArithmeticDecoder(bitstr2[0])
                        dg = dec_g.decode_symbol(cum_g)
                        gsize = TAIL_GROUP if dg < n_groups - 1 else len(rest) - dg * TAIL_GROUP
                        cum_o = uniform_cum(gsize)
                        dec_o = ArithmeticDecoder(bitstr2[1])
                        got = int(rest[dg * TAIL_GROUP + dec_o.decode_symbol(cum_o)])
                    else:
                        assert bitstr2 is None
                        got = int(topk_idx[d1])
                    if got == target:
                        n_verify_ok += 1
                    else:
                        n_verify_fail += 1
                        print(f"  FAIL at block {b0}, pos {t_idx}")

            # Causal cache update AFTER this block is fully coded.
            # The decoder applies the identical update after decoding,
            # so both sides stay in sync.
            if ENSEMBLE_LAMBDA < 1.0:
                b = torch.tensor(block, dtype=torch.long, device=device)
                cache_counts.scatter_add_(
                    0, b, torch.ones_like(b, dtype=torch.float32))
                cache_total += len(block)

            if (b0 // BLOCK_TOKENS + 1) % 2 == 0:
                print(f"  ... block {b0//BLOCK_TOKENS+1}/{(len(ids)+BLOCK_TOKENS-1)//BLOCK_TOKENS}, "
                      f"escapes so far: {n_escapes}", flush=True)

            # Causal online update AFTER coding this block (decoder mirrors
            # it from decoded tokens; eval mode => deterministic).
            block_idx = b0 // BLOCK_TOKENS
            if adapt_opt is not None and (block_idx + 1) % ADAPT_EVERY == 0:
                import torch.nn.functional as _F
                xb = torch.tensor([block], device=device)
                inp, tgt = xb[:, :-1], xb[:, 1:]
                with torch.set_grad_enabled(True):
                    if ADAPT_DTYPE == "fp32":
                        ulogits = model(xb).logits[:, :-1, :]
                    else:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            ulogits = model(xb).logits[:, :-1, :]
                    uloss = _F.cross_entropy(
                        ulogits.reshape(-1, ulogits.size(-1)),
                        tgt.reshape(-1))
                if not (torch.is_tensor(uloss) and uloss.requires_grad):
                    _ng = sum(1 for p in model.parameters() if p.requires_grad)
                    raise RuntimeError(
                        f"DEBUG no-grad: is_grad_enabled={torch.is_grad_enabled()}, "
                        f"inference_mode={torch.is_inference_mode_enabled()}, "
                        f"n_requires_grad={_ng}, uloss_type={type(uloss)}")
                adapt_opt.zero_grad()
                uloss.backward()
                adapt_opt.step()

    torch.cuda.synchronize()
    dt = time.time() - t
    n_tokens = len(ids) - (len(ids) // BLOCK_TOKENS)  # approx symbols coded
    print(f"\n  Total bits: {total_bits}")
    print(f"  bits/byte: {total_bits/n_bytes:.4f}")
    print(f"  escapes: {n_escapes}")
    print(f"  Verified: {n_verify_ok}/200 lossless, fails: {n_verify_fail}")
    print(f"  Time: {dt:.1f}s")

    results = {
        "model": f"SmolLM2-135M-{ADAPT_MODE}",
        "params": n_params,
        "trainable": n_train,
        "adapt_lr": ADAPT_LR,
        "top_k": TOP_K,
        "n_bytes": n_bytes,
        "total_bits": total_bits,
        "bpb": total_bits / n_bytes,
        "escapes": n_escapes,
        "verified_ok": n_verify_ok,
        "verified_fail": n_verify_fail,
        "elapsed_s": time.time() - t0,
    }
    out_path = f"./data/smollm2_{ADAPT_MODE}_off{_off_mb}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
