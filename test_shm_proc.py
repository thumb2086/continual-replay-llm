"""Standalone plumbing test for v13 proc-loop (CPU only, no GPU/model).

Validates: shared-memory create/attach across procs, task dispatch,
single-source determinism (thread result == proc result).

Run (only when the user greenlights runs):
    python test_shm_proc.py
NOT part of benchmark runs. Must print MATCH on all lines.
"""
import sys
import time
import numpy as np

sys.path.insert(0, ".")
from bpe_ensemble_v13 import (
    _range_core, _proc_init, _proc_range, _shm_alloc, _SHM_MAIN,
    _shm_cleanup,
)
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

V, K, P, NC, T0 = 3000, 1024, 2048, 400, 3
rng = np.random.default_rng(7)
block = [int(x) for x in rng.integers(0, V, size=NC + 8)]
bi_id_of = {11: 0, 22: 1}
bi_keys = [np.array([5, 6, 7], dtype=np.int64),
           np.array([100, 200], dtype=np.int64)]
bi_vals = [np.array([3.0, 1.0, 2.0]), np.array([9.0, 4.0])]
bi_tot = np.array([6.0, 13.0])
tri_id_of, tri_keys, tri_vals, tri_tot = {}, [], [], np.array([])

specs = [
    ("t_blk", (2, 600, V), "float32"),
    ("t_tv", (2, 600, K), "float32"),
    ("t_ti", (2, 600, K), "int64"),
    ("t_pi", (2, 600, P), "int64"),
    ("t_obt", (600, K), "float64"),
    ("t_ork", (600,), "int64"),
    ("t_opa", (600,), "int8"),
    ("t_oti", (600, K), "int64"),
]
_SHM_MAIN.clear()
for nm, sh, dt in specs:
    _SHM_MAIN[nm] = _shm_alloc(nm, sh, dt)
G = {nm: _SHM_MAIN[nm][1] for nm, _, _ in specs}
G["blk"][0] = rng.random((600, V)).astype(np.float32)
G["tv"][0] = rng.random((600, K)).astype(np.float32)
G["ti"][0] = rng.integers(0, V, size=(600, K)).astype(np.int64)
G["pi"][0] = rng.integers(0, V, size=(600, P)).astype(np.int64)

scratch = [
    np.zeros(V, dtype=np.int32), np.zeros(V, dtype=np.int32),
    np.zeros(V, dtype=np.float64), np.empty(K, dtype=np.float64),
    np.empty(K, dtype=np.int64), np.empty(K, dtype=np.int64),
    np.empty(K, dtype=np.float64), np.zeros(1, dtype=np.int64),
    [0],
]

t = time.time()
with ThreadPoolExecutor(max_workers=2) as pool:
    futs = [pool.submit(_range_core, block, T0, T0, T0 + 200, 0, 999,
                        bi_id_of, tri_id_of, bi_keys, bi_vals, bi_tot,
                        tri_keys, tri_vals, tri_tot,
                        G["blk"][0], G["tv"][0], G["ti"][0], G["pi"][0],
                        G["obt"], G["ork"], G["opa"], G["oti"], scratch),
            pool.submit(_range_core, block, T0, T0 + 200, T0 + NC, 0, 999,
                        bi_id_of, tri_id_of, bi_keys, bi_vals, bi_tot,
                        tri_keys, tri_vals, tri_tot,
                        G["blk"][0], G["tv"][0], G["ti"][0], G["pi"][0],
                        G["obt"], G["ork"], G["opa"], G["oti"], scratch)]
    for f in futs:
        f.result()
print(f"thread: {time.time() - t:.1f}s")
snap = {k: G[k][:NC].copy() for k in ("obt", "ork", "opa", "oti")}
for k in ("obt", "ork", "opa", "oti"):
    G[k][:NC] = 0

t = time.time()
with ProcessPoolExecutor(
        max_workers=2, initializer=_proc_init,
        initargs=([[nm, list(sh), dt] for nm, sh, dt in specs], V, K)) as pool:
    base = (block, T0, 0, 999, bi_id_of, tri_id_of, bi_keys, bi_vals,
            bi_tot, tri_keys, tri_vals, tri_tot)
    futs = [pool.submit(_proc_range, *base, T0, T0 + 200, 0),
            pool.submit(_proc_range, *base, T0 + 200, T0 + NC, 0)]
    for f in futs:
        assert f.result() is True
print(f"proc: {time.time() - t:.1f}s")

ok = True
for k in ("obt", "ork", "opa", "oti"):
    same = bool((G[k][:NC] == snap[k]).all())
    print(("MATCH " if same else "DIFF  ") + k)
    ok = ok and same
_shm_cleanup()
print("ALL-MATCH" if ok else "MISMATCH-FOUND")
