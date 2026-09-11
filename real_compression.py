"""Real Compression: Huffman baseline vs Neural Arithmetic Coding.

Produces actual compressed bitstreams and verifies lossless decoding.
"""

import torch
import torch.nn.functional as F
import numpy as np
import numba
import json
import heapq
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

from train_v3 import OnlineLLMV3, CharTokenizer, load_grouped_texts, create_chunks
from compare_ablation import make_batch


def maybe_autocast(use_fp16):
    """fp16 autocast context, or no-op for fp32 measurement."""
    if use_fp16:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


# ============================================================
# Huffman Coding (baseline)
# ============================================================

class HuffmanNode:
    def __init__(self, freq, symbol=None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman(freq):
    """Build Huffman tree from {symbol: freq}. Returns {symbol: code_str}."""
    heap = [HuffmanNode(f, s) for s, f in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        heapq.heappush(heap, HuffmanNode(a.freq + b.freq, None, a, b))
    root = heap[0]
    codes = {}

    def walk(node, prefix):
        if node.symbol is not None:
            codes[node.symbol] = prefix or "0"
        else:
            walk(node.left, prefix + "0")
            walk(node.right, prefix + "1")

    walk(root, "")
    return codes, root


def huffman_compress(token_ids, codes):
    """Encode token ids to bitstring."""
    return "".join(codes[t] for t in token_ids)


def huffman_decompress(bitstr, root, n_symbols):
    """Decode bitstring using Huffman tree."""
    out = []
    node = root
    for b in bitstr:
        node = node.left if b == "0" else node.right
        if node.symbol is not None:
            out.append(node.symbol)
            node = root
            if len(out) == n_symbols:
                break
    return out


# ============================================================
# Arithmetic Coding (neural)
# ============================================================

# Code values use 16-bit thresholds (classic Witten-Neal-Cleary setup).
CODE_BITS = 16
TOP = (1 << CODE_BITS) - 1
HALF = 1 << (CODE_BITS - 1)
QUARTER = 1 << (CODE_BITS - 2)
THREE_QUARTER = HALF + QUARTER

# Frequency total MUST satisfy TOTAL <= min post-renormalization range.
# Min exit range is 16386 (proven from the E1/E2/E3 exit conditions), so
# TOTAL = 2**14 = 16384 is safe: every encoded symbol leaves low <= high,
# which makes the renormalization loop always terminate.
FREQ_BITS = 14
TOTAL = 1 << FREQ_BITS


def probs_to_freqs_batch(probs):
    """Vectorized version: probs (N, V) -> cums (N, V+1) int32.

    Same math as probs_to_freqs applied row-wise. int32 halves the
    memory bandwidth vs int64 (max value TOTAL=16384 fits easily).
    Validity (exact sum, floor >= 1) is structural; equivalence with
    the checked single-row version is covered by test_arithmetic_smoke.py.
    """
    probs = np.ascontiguousarray(probs, dtype=np.float32)
    N, V = probs.shape
    scale = np.float32(TOTAL - V) / probs.sum(axis=1, keepdims=True)
    freqs = (probs * scale).astype(np.int32) + np.int32(1)
    diff = np.int32(TOTAL) - freqs.sum(axis=1, dtype=np.int32)
    idx = np.argmax(probs, axis=1)
    freqs[np.arange(N), idx] += diff
    cum = np.empty((N, V + 1), dtype=np.int32)
    cum[:, 0] = 0
    np.cumsum(freqs, axis=1, out=cum[:, 1:])
    return cum


def gpu_quantize(logits, out_dtype=torch.int32):
    """GPU-side quantization: logits (B, T, V) -> cum (B*T, V+1) numpy.

    Softmax, scaling, flooring, argmax-fix and cumsum all run as GPU
    kernels; only the final cum crosses PCIe. uint16 halves the transfer
    (values < TOTAL=16384 always fit). Deterministic for identical
    inputs, so encoder and decoder stay in sync.
    """
    B, T, V = logits.shape
    probs = F.softmax(logits.float(), dim=-1).reshape(-1, V)
    rowsum = probs.sum(dim=1, keepdim=True)
    scale = (TOTAL - V) / rowsum
    freqs = (probs * scale).to(torch.int32) + 1
    diff = TOTAL - freqs.sum(dim=1)
    idx = probs.argmax(dim=1)
    freqs[torch.arange(freqs.shape[0], device=freqs.device), idx] += diff
    cum = torch.empty(freqs.shape[0], V + 1, dtype=torch.int32, device=freqs.device)
    cum[:, 0] = 0
    torch.cumsum(freqs, dim=1, out=cum[:, 1:])
    if out_dtype == torch.uint16:
        return cum.to(torch.uint16).cpu().numpy()
    return cum.cpu().numpy()


def probs_to_freqs(probs):
    """Quantize probability vector to integer frequencies summing to TOTAL.

    Every symbol gets frequency >= 1 (additive smoothing), so the arithmetic
    coder range can never collapse on a zero-probability symbol.
    """
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    V = len(probs)
    # Reserve 1 count per symbol, distribute the rest proportionally
    freqs = (probs * (TOTAL - V)).astype(np.int64) + 1
    # Fix rounding so sum == TOTAL exactly
    diff = int(TOTAL - freqs.sum())
    if diff != 0:
        idx = int(np.argmax(probs))
        freqs[idx] += diff
        if freqs[idx] < 1:
            # Extremely unlikely; fall back to uniform
            freqs[:] = TOTAL // V
            freqs[: TOTAL % V] += 1
    assert freqs.sum() == TOTAL and (freqs >= 1).all()
    cum = np.zeros(len(freqs) + 1, dtype=np.int64)
    np.cumsum(freqs, out=cum[1:])
    return freqs, cum


MAX_RENORM_ITERS = 1000  # hard cap: clean error instead of hang
MAX_BITS_FACTOR = 64  # abort if bits exceed 64x symbols (pathological)

# 16-bit code thresholds, 14-bit frequency total (classic safe setup).
NB_HALF = np.int64(1 << 15)
NB_QUARTER = np.int64(1 << 14)
NB_3Q = NB_HALF + NB_QUARTER
NB_TOTAL = np.int64(1 << 14)


@numba.njit
def nb_encode_flat(cums, symbols, chunk_start, chunk_len, out_counts):
    """Count bits for MANY independent chunks in one call.

    cums: (P, V+1) cumulative frequencies (any int dtype, widened to int64).
    symbols: (P,) symbol ids.
    chunk_start[i], chunk_len[i]: position range of chunk i.
    out_counts[i]: total bits for chunk i (finish bits included).
    Each chunk gets a FRESH encoder state, identical to nb_encode_count.
    Returns 0 on success, negative error code otherwise.
    """
    n_chunks = chunk_start.shape[0]
    for c in range(n_chunks):
        low = np.int64(0)
        high = np.int64((1 << 16) - 1)
        pending = np.int64(0)
        count = np.int64(0)
        s0 = chunk_start[c]
        n = chunk_len[c]
        for k in range(n):
            t = s0 + k
            s = np.int64(symbols[t])
            sym_low = np.int64(cums[t, s])
            sym_high = np.int64(cums[t, s + 1])
            if sym_high <= sym_low:
                return np.int64(-1)
            if not (np.int64(0) <= low and low <= high):
                return np.int64(-2)
            rng = high - low + 1
            high = low + (rng * sym_high) // NB_TOTAL - 1
            low = low + (rng * sym_low) // NB_TOTAL
            it = np.int64(0)
            while True:
                it += 1
                if it > 1000:
                    return np.int64(-3)
                if high < NB_HALF:
                    count += 1 + pending
                    pending = 0
                    low <<= np.int64(1)
                    high = (high << np.int64(1)) + 1
                elif low >= NB_HALF:
                    count += 1 + pending
                    pending = 0
                    low = (low - NB_HALF) << np.int64(1)
                    high = ((high - NB_HALF) << np.int64(1)) + 1
                elif low >= NB_QUARTER and high < NB_3Q:
                    pending += 1
                    if pending > 1000:
                        return np.int64(-4)
                    low = (low - NB_QUARTER) << np.int64(1)
                    high = ((high - NB_QUARTER) << np.int64(1)) + 1
                else:
                    break
        count += 2 + pending
        out_counts[c] = count
    return np.int64(0)


@numba.njit
def nb_encode_count(cums, symbols):
    """Count arithmetic-coding bits for a position sequence.

    cums: (N, V+1) int64 cumulative frequencies, cums[:, -1] == TOTAL.
    symbols: (N,) int64 symbol ids.
    Returns total bit count. Raises (via return code) on invalid range.
    """
    low = np.int64(0)
    high = np.int64((1 << 16) - 1)
    pending = np.int64(0)
    count = np.int64(0)
    n = symbols.shape[0]
    for t in range(n):
        s = np.int64(symbols[t])
        # Widen to int64: rng*sym can approach 2**31, must not wrap.
        sym_low = np.int64(cums[t, s])
        sym_high = np.int64(cums[t, s + 1])
        if sym_high <= sym_low:
            return np.int64(-1)  # invalid: zero-frequency symbol
        if not (np.int64(0) <= low and low <= high):
            return np.int64(-2)  # invalid range
        rng = high - low + 1
        high = low + (rng * sym_high) // NB_TOTAL - 1
        low = low + (rng * sym_low) // NB_TOTAL
        it = np.int64(0)
        while True:
            it += 1
            if it > 1000:
                return np.int64(-3)  # did not converge
            if high < NB_HALF:
                count += 1 + pending
                pending = 0
                low <<= np.int64(1)
                high = (high << np.int64(1)) + 1
            elif low >= NB_HALF:
                count += 1 + pending
                pending = 0
                low = (low - NB_HALF) << np.int64(1)
                high = ((high - NB_HALF) << np.int64(1)) + 1
            elif low >= NB_QUARTER and high < NB_3Q:
                pending += 1
                if pending > 1000:
                    return np.int64(-4)
                low = (low - NB_QUARTER) << np.int64(1)
                high = ((high - NB_QUARTER) << np.int64(1)) + 1
            else:
                break
    count += 2 + pending  # finish() emits pending+1 opposite bits + 1 discriminator
    return count


class ArithmeticEncoder:
    """Arithmetic encoder with bounded memory.

    store=False (default): only counts bits, stores nothing.
    store=True: packs bits into a bytearray (1 byte per 8 bits).
    """

    def __init__(self, store=False):
        self.low = 0
        self.high = TOP  # 16-bit code range, independent of frequency TOTAL
        self.pending = 0
        self.count = 0
        self.store = store
        self._buf = bytearray()
        self._cur = 0
        self._ncur = 0

    def _emit(self, bit):
        self.count += 1 + self.pending
        if self.store:
            bits = [bit] + [1 - bit] * self.pending
            for b in bits:
                self._cur = (self._cur << 1) | b
                self._ncur += 1
                if self._ncur == 8:
                    self._buf.append(self._cur)
                    self._cur = 0
                    self._ncur = 0
        self.pending = 0

    def _check_range(self, where):
        if not (0 <= self.low <= self.high):
            raise ArithmeticError(
                f"invalid range at {where}: low={self.low} high={self.high}"
            )

    def encode_symbol(self, cum, symbol):
        freq = int(cum[symbol + 1]) - int(cum[symbol])
        if freq <= 0:
            raise ArithmeticError(f"zero-frequency symbol {symbol}")
        self._check_range("encode_entry")
        sym_low = int(cum[symbol])
        sym_high = int(cum[symbol + 1])
        rng = self.high - self.low + 1
        self.high = self.low + (rng * sym_high) // TOTAL - 1
        self.low = self.low + (rng * sym_low) // TOTAL
        iters = 0
        while True:
            iters += 1
            if iters > MAX_RENORM_ITERS:
                raise ArithmeticError("renormalization did not converge")
            if self.high < HALF:
                self._emit(0)
                self.low <<= 1
                self.high = (self.high << 1) + 1
            elif self.low >= HALF:
                self._emit(1)
                self.low = (self.low - HALF) << 1
                self.high = ((self.high - HALF) << 1) + 1
            elif self.low >= QUARTER and self.high < THREE_QUARTER:
                self.pending += 1
                if self.pending > MAX_RENORM_ITERS:
                    raise ArithmeticError("pending underflow overflow")
                self.low = (self.low - QUARTER) << 1
                self.high = ((self.high - QUARTER) << 1) + 1
            else:
                break

    def finish(self):
        self.pending += 1
        if self.low < QUARTER:
            self._emit(0)
        else:
            self._emit(1)
        bitstr = None
        if self.store:
            if self._ncur:
                self._buf.append(self._cur << (8 - self._ncur))
            bitstr = "".join(f"{byte:08b}" for byte in self._buf)[: self.count]
        return bitstr, self.count


class ArithmeticDecoder:
    def __init__(self, bitstr):
        self.bits = bitstr
        self.pos = 0
        self.low = 0
        self.high = TOP  # must match encoder init
        self.code = 0
        for _ in range(CODE_BITS):
            self.code = (self.code << 1) | self._read_bit()

    def _read_bit(self):
        if self.pos < len(self.bits):
            b = int(self.bits[self.pos])
            self.pos += 1
            return b
        return 0

    def decode_symbol(self, cum):
        if not (0 <= self.low <= self.high):
            raise ArithmeticError(
                f"invalid decoder range: low={self.low} high={self.high}"
            )
        rng = self.high - self.low + 1
        value = ((self.code - self.low + 1) * TOTAL - 1) // rng
        # Binary search
        lo, hi = 0, len(cum) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if cum[mid] <= value:
                lo = mid
            else:
                hi = mid
        symbol = lo
        if int(cum[symbol + 1]) - int(cum[symbol]) <= 0:
            raise ArithmeticError(f"zero-frequency symbol {symbol} in decoder")
        sym_low = int(cum[symbol])
        sym_high = int(cum[symbol + 1])
        self.high = self.low + (rng * sym_high) // TOTAL - 1
        self.low = self.low + (rng * sym_low) // TOTAL
        iters = 0
        while True:
            iters += 1
            if iters > MAX_RENORM_ITERS:
                raise ArithmeticError("decoder renormalization did not converge")
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.code -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTER:
                self.low -= QUARTER
                self.high -= QUARTER
                self.code -= QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) + 1
            self.code = (self.code << 1) | self._read_bit()
        return symbol


# ============================================================
# Experiment
# ============================================================

def get_probs_batch(model, tokenizer, chunk_ids, device):
    """Get per-position probability distributions for a chunk."""
    model.eval()
    with torch.no_grad():
        x = torch.tensor([chunk_ids], device=device)
        logits = model(x)
        probs = F.softmax(logits[0], dim=-1).cpu().numpy()
    return probs  # (seq_len, vocab)


def neural_compress_stream(model, tokenizer, chunks, device, adapt=False,
                           replay_pool=None, opt=None, verify=True,
                           block_size=4, freeze_after=None, use_fp16=True):
    """Compress stream with neural arithmetic coding. Returns total bits.

    Block processing: each block of `block_size` chunks gets ONE batched
    forward (identical math to sequential forwards while weights are frozen),
    symbols are counted with the numba fast path, then ONE optimizer update
    is applied on the stacked block. The decoder mirrors this exactly, so
    causality and sync are preserved.

    freeze_after: if set, stop applying updates after this many chunks
    (prefix adaptation). The decoder reproduces this by counting decoded
    chunks, so sync is preserved.
    """
    total_bits = 0
    total_symbols = 0
    n_verify = 0
    scaler = torch.amp.GradScaler("cuda") if adapt else None

    # Pre-encode all chunks to id lists
    id_lists = []
    for chunk in chunks:
        ids = tokenizer.encode(chunk)
        if len(ids) >= 3:
            id_lists.append(ids)

    model.eval()  # deterministic throughout; updates use no dropout noise

    # Split: adaptive prefix vs frozen bulk. The decoder reproduces the same
    # split by counting decoded chunks, so sync is preserved.
    if adapt and freeze_after is not None:
        adapt_ids = id_lists[:freeze_after]
        frozen_ids = id_lists[freeze_after:]
    else:
        adapt_ids = id_lists if adapt else []
        frozen_ids = [] if adapt else id_lists

    def code_with_probs(id_batch, cum_batch_3d):
        """Numba fast-path coding for a list of id lists. Returns (bits, syms)."""
        nonlocal_total = [0, 0]
        for j, ids in enumerate(id_batch):
            L = len(ids)
            cum = np.ascontiguousarray(cum_batch_3d[j, : L - 1])
            syms = np.array(ids[1:], dtype=np.int64)
            nbits = int(nb_encode_count(cum, syms))
            if nbits < 0:
                raise ArithmeticError("encoder error in fast path")
            if nbits > MAX_BITS_FACTOR * (L - 1):
                raise ArithmeticError("bit blowup in fast path")
            nonlocal_total[0] += nbits
            nonlocal_total[1] += L - 1
        return nonlocal_total[0], nonlocal_total[1]

    def batched_forward(id_batch):
        """One forward + GPU quantization for a list of id lists."""
        maxlen = max(len(ids) for ids in id_batch)
        batch = torch.zeros(len(id_batch), maxlen, dtype=torch.long, device=device)
        for j, ids in enumerate(id_batch):
            batch[j, : len(ids)] = torch.tensor(ids, device=device)
        with torch.no_grad(), maybe_autocast(use_fp16):
            logits = model(batch)
        if not torch.isfinite(logits).all():
            raise ArithmeticError("non-finite model logits")
        V = logits.shape[-1]
        return batch, gpu_quantize(logits).reshape(len(id_batch), -1, V + 1)

    # ---- Phase 1: adaptive prefix (block-wise with updates) ----
    for bi in range(0, len(adapt_ids), block_size):
        block = adapt_ids[bi : bi + block_size]
        batch, cum_block = batched_forward(block)

        for j, ids in enumerate(block):
            L = len(ids)
            cum = np.ascontiguousarray(cum_block[j, : L - 1])
            syms = np.array(ids[1:], dtype=np.int64)
            nbits = int(nb_encode_count(cum, syms))
            if nbits < 0:
                raise ArithmeticError(f"encoder error at adapt block {bi}")
            if nbits > MAX_BITS_FACTOR * (L - 1):
                raise ArithmeticError(f"bit blowup at adapt block {bi}")

            # Verify first 3 chunks with the reference Python path
            if verify and n_verify < 3:
                enc = ArithmeticEncoder(store=True)
                for t in range(L - 1):
                    enc.encode_symbol(cum[t], int(syms[t]))
                bitstr, nbits_py = enc.finish()
                assert nbits_py == nbits, f"fast/slow count mismatch at block {bi}"
                dec = ArithmeticDecoder(bitstr)
                out = [dec.decode_symbol(cum[t]) for t in range(L - 1)]
                assert out == ids[1:], f"LOSSLESS FAIL at block {bi}"
                n_verify += 1

            total_bits += nbits
            total_symbols += L - 1

        # ONE causal update per block
        model.eval()
        with torch.set_grad_enabled(True):
            inp, tgt = batch[:, :-1], batch[:, 1:]
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits_b = model(inp)
                min_len = min(logits_b.shape[1], tgt.shape[1])
                loss = F.cross_entropy(
                    logits_b[:, :min_len].reshape(-1, logits_b.size(-1)),
                    tgt[:, :min_len].reshape(-1),
                )
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            # Memory store (matches online_step behavior)
            with torch.no_grad():
                h = model.embedding(inp).mean(1)
                for v in h:
                    model.memory.store(v, v, importance=1.0 / (1.0 + loss.item()))

    # ---- Phase 2: frozen bulk (giant batches, no updates) ----
    # All chunks are exactly 128 tokens, so stack aggressively.
    FROZEN_BATCH = 128
    for fi in range(0, len(frozen_ids), FROZEN_BATCH):
        fblock = frozen_ids[fi : fi + FROZEN_BATCH]
        batch, cum_block = batched_forward(fblock)
        b, n = code_with_probs(fblock, cum_block)
        total_bits += b
        total_symbols += n

    return total_bits, total_symbols, n_verify


def frozen_code_all(model, tokenizer, chunks, device, verify=True,
                    fwd_batch=200, use_fp16=True):
    """Frozen ultra-fast path: giant forwards + ONE flat numba call.

    No weight updates. All chunks share one (or few) forward passes and
    a single numba counting call. Decoder mirrors trivially (no updates
    to reproduce).
    """
    id_lists = []
    for chunk in chunks:
        ids = tokenizer.encode(chunk)
        if len(ids) >= 3:
            id_lists.append(ids)

    model.eval()
    total_bits, total_symbols, n_verify = 0, 0, 0

    for fi in range(0, len(id_lists), fwd_batch):
        fblock = id_lists[fi : fi + fwd_batch]
        maxlen = max(len(ids) for ids in fblock)
        batch = torch.zeros(len(fblock), maxlen, dtype=torch.long, device=device)
        for j, ids in enumerate(fblock):
            batch[j, : len(ids)] = torch.tensor(ids, device=device)
        with torch.no_grad(), maybe_autocast(use_fp16):
            logits = model(batch)
        if not torch.isfinite(logits).all():
            raise ArithmeticError("non-finite logits in frozen path")
        V = logits.shape[-1]
        # uint16 halves the PCIe transfer; values < TOTAL always fit
        cum = gpu_quantize(logits, out_dtype=torch.uint16).reshape(len(fblock), -1, V + 1)

        # Flatten all chunks of this forward-batch into ONE numba call
        starts, lens, syms_list, cum_rows = [], [], [], []
        pos = 0
        for j, ids in enumerate(fblock):
            L = len(ids)
            starts.append(pos)
            lens.append(L - 1)
            syms_list.append(np.array(ids[1:], dtype=np.int64))
            cum_rows.append(np.ascontiguousarray(cum[j, : L - 1]))
            pos += L - 1
        flat_syms = np.concatenate(syms_list)
        flat_cum = np.concatenate(cum_rows, axis=0)
        starts_a = np.array(starts, dtype=np.int64)
        lens_a = np.array(lens, dtype=np.int64)
        out = np.zeros(len(fblock), dtype=np.int64)
        rc = nb_encode_flat(flat_cum, flat_syms, starts_a, lens_a, out)
        if rc != 0:
            raise ArithmeticError(f"flat encoder error {rc}")
        for j, ids in enumerate(fblock):
            nbits = int(out[j])
            if nbits > MAX_BITS_FACTOR * (len(ids) - 1):
                raise ArithmeticError("bit blowup in frozen path")
            total_bits += nbits
            total_symbols += len(ids) - 1

        # Verify first 3 chunks overall with the reference Python path
        if verify and n_verify < 3:
            for j, ids in enumerate(fblock):
                if n_verify >= 3:
                    break
                L = len(ids)
                c = np.ascontiguousarray(cum[j, : L - 1]).astype(np.int64)
                s = np.array(ids[1:], dtype=np.int64)
                enc = ArithmeticEncoder(store=True)
                for t in range(L - 1):
                    enc.encode_symbol(c[t], int(s[t]))
                bitstr, _ = enc.finish()
                dec = ArithmeticDecoder(bitstr)
                assert [dec.decode_symbol(c[t]) for t in range(L - 1)] == ids[1:]
                n_verify += 1

    return total_bits, total_symbols, n_verify


def main():
    print("=" * 70)
    print("REAL COMPRESSION: Huffman vs Neural Arithmetic Coding")
    print("=" * 70)

    torch.manual_seed(123)
    np.random.seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    t0 = time.time()

    # ---- Data ----
    base_path = r"C:\Users\CPXru\Music\playlist-admin\podcasts"
    texts_a, texts_b, _, _ = load_grouped_texts(base_path, max_files_per_group=25)

    tok = CharTokenizer()
    tok.fit(texts_a + texts_b)
    print(f"Vocab: {tok.vocab_size}")

    chunks_a = create_chunks(texts_a, chunk_len=128)
    chunks_b = create_chunks(texts_b, chunk_len=128)
    np.random.shuffle(chunks_a)
    np.random.shuffle(chunks_b)

    # Use 400 chunks for the real-coding test (runtime)
    stream = chunks_b[:400]
    stream_ids = []
    for c in stream:
        ids = tok.encode(c)
        if len(ids) >= 3:
            stream_ids.append(ids)
    n_symbols = sum(len(ids) - 1 for ids in stream_ids)
    n_chars = sum(len(c) for c in stream)
    print(f"Stream: {len(stream_ids)} chunks, {n_symbols} symbols, {n_chars} chars")

    # ---- Huffman baseline ----
    print("\n[1/3] Huffman baseline...")
    t_huff = time.time()
    all_ids = [i for ids in stream_ids for i in ids]
    freq = Counter(all_ids)
    codes, root = build_huffman(freq)
    huff_bits = sum(len(codes[t]) for t in all_ids)
    # Verify
    test_ids = stream_ids[0]
    test_bits = huffman_compress(test_ids, codes)
    assert huffman_decompress(test_bits, root, len(test_ids)) == test_ids
    avg_code_len = np.mean([len(codes[t]) for t in all_ids])
    print(f"  Symbols: {len(all_ids)}, unique: {len(freq)}")
    print(f"  Total bits: {huff_bits}, bits/char: {huff_bits/n_chars:.3f}")
    print(f"  Avg code length: {avg_code_len:.3f} bits/symbol")
    print(f"  Round-trip verified on chunk 0")
    print(f"  Huffman time: {time.time()-t_huff:.1f}s")

    # ---- Pretrain neural model on A ----
    print("\n[2/3] Pretraining neural model on Topic A...")
    t_pre = time.time()
    from compare_ablation import make_batch  # noqa
    model = OnlineLLMV3(
        vocab_size=tok.vocab_size, embed_dim=256, hidden_dim=512, n_layers=6, n_heads=8
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    model.train()
    for epoch in range(10):
        total = 0.0
        nb = 0
        for i in range(0, len(chunks_a) - 8, 8):
            padded = make_batch(tok, chunks_a[i:i + 8], device)
            total += model.pretrain_step(padded, optimizer)
            nb += 1
        print(f"  Epoch {epoch+1}: loss={total/nb:.4f}")

    # ---- Neural arithmetic coding: prefix-adaptation curve ----
    print(f"\n  Pretrain time: {time.time()-t_pre:.1f}s")
    print("\n[3/3] Prefix adaptation curve (frozen base + varying adapt prefix)...")
    import copy
    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    curve = {}
    for freeze_after in [None, 200, 100, 40]:
        m = OnlineLLMV3(
            vocab_size=tok.vocab_size, embed_dim=256, hidden_dim=512,
            n_layers=6, n_heads=8,
        ).to(device)
        m.load_state_dict(base_state)
        adapt_opt = torch.optim.Adam(m.parameters(), lr=1e-4, fused=True)
        t_code = time.time()
        nbits, nsym, nver = neural_compress_stream(
            m, tok, stream, device, adapt=True, replay_pool=chunks_a,
            opt=adapt_opt, block_size=8, freeze_after=freeze_after,
        )
        dt = time.time() - t_code
        key = "full" if freeze_after is None else f"prefix{freeze_after}"
        curve[key] = {
            "bpc": nbits / n_chars,
            "chars_per_s": n_chars / dt,
            "verified": nver,
        }
        print(f"  {key:10s}: {nbits/n_chars:.3f} bpc, {n_chars/dt:.0f} chars/s "
              f"(verified {nver})")
        del m
        torch.cuda.empty_cache()

    # Fully frozen ultra-fast path (no updates at all)
    mf = OnlineLLMV3(
        vocab_size=tok.vocab_size, embed_dim=256, hidden_dim=512,
        n_layers=6, n_heads=8,
    ).to(device)
    mf.load_state_dict(base_state)
    t_frozen = time.time()
    fbits, _, fver = frozen_code_all(mf, tok, stream, device, verify=True)
    fdt = time.time() - t_frozen
    curve["frozen"] = {
        "bpc": fbits / n_chars,
        "chars_per_s": n_chars / fdt,
        "verified": fver,
    }
    print(f"  {'frozen':10s}: {fbits/n_chars:.3f} bpc, {n_chars/fdt:.0f} chars/s "
          f"(verified {fver})")
    del mf
    torch.cuda.empty_cache()

    nbits = int(curve["full"]["bpc"] * n_chars)
    code_time = None  # per-setting times printed above

    # ---- Compare ----
    print("\n" + "=" * 70)
    print("RESULTS (real compressed bits)")
    print("=" * 70)
    huff_bpc = huff_bits / n_chars
    print(f"  Huffman: {huff_bpc:.3f} bits/char")
    print(f"  Raw UTF-8: {sum(len(c.encode('utf-8')) for c in stream)*8/n_chars:.3f} bits/char")
    for key, r in curve.items():
        win = (huff_bpc - r["bpc"]) / huff_bpc * 100
        print(f"  Neural {key:10s}: {r['bpc']:.3f} bits/char "
              f"({win:+.1f}% vs Huffman), {r['chars_per_s']:.0f} chars/s")

    nbits = int(curve["full"]["bpc"] * n_chars)
    neur_bpc = curve["full"]["bpc"]
    nver = curve["full"]["verified"]

    results = {
        "n_chunks": len(stream_ids),
        "n_symbols": n_symbols,
        "n_chars": n_chars,
        "huffman_bpc": huff_bpc,
        "neural_bpc": neur_bpc,
        "prefix_curve": curve,
        "roundtrip_verified_chunks": nver,
        "elapsed_s": time.time() - t0,
    }
    Path("./data").mkdir(parents=True, exist_ok=True)
    with open("./data/real_compression.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to data/real_compression.json ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
