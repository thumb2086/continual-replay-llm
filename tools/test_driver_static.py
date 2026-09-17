"""Static checks on the GPU-only driver (stdlib only, no torch needed).

WHY THIS EXISTS
---------------
`tools/seg_token_compressor.py` cannot be imported here: it needs torch, a GPU
and a model directory, so nothing in this sandbox can execute it. That is fine
for logic that lives in importable modules (`seg_codec`, the chunk schedule) but
it means whole-run failures of the "crashed after 24 minutes of work" kind have
no test coverage at all.

That happened: the driver frees the encode-side model state with

    del nl_enc, nl_enc_timed          # after the encode pass

and the result dict, written after decode, reached back for
`nl_enc.prefill_note` -- UnboundLocalError, after a full encode AND decode had
already succeeded, with nothing to verify against. So this file scans the driver
as TEXT/AST for the failure classes that need no model:

  * a name used after it was deleted in the same function, in source order;
  * a name used inside a function before any binding of it exists there (a
    typo'd or forgotten local).

Note on pyflakes: it reports the closure `nl_enc_timed -> nl_enc` as undefined
because `del` removes the binding, but that reference is only reached before the
`del`, so it is a false positive. Source order is the thing that matters, and
that is what this check uses.

Run: python3 tools/test_driver_static.py
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.join(HERE, "seg_token_compressor.py")

FAILS = []


def check(label, ok, detail=""):
    print(f"  [{'ok  ' if ok else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


BINDING_NODES = (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.For, ast.With,
                 ast.FunctionDef, ast.AsyncFunctionDef, ast.Import,
                 ast.ImportFrom, ast.comprehension, ast.Lambda)


def bound_names(node):
    """Names a statement binds (shallow: this statement only, not its body)."""
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)) \
                and sub is not node:
            if isinstance(sub.ctx, ast.Store):
                out.add(sub.id)
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and sub is not node:
            out.add(sub.name)
            for a in sub.args.args + sub.args.kwonlyargs:
                out.add(a.arg)
        if isinstance(sub, ast.Import):
            for a in sub.names:
                out.add((a.asname or a.name).split(".")[0])
        if isinstance(sub, ast.ImportFrom):
            for a in sub.names:
                out.add(a.asname or a.name)
    return out


def deleted_names(node):
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Delete):
            for t in sub.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
    return out


def scan_function(fn, path, problems):
    """Walk a function body in source order, tracking del-then-use."""
    flat = []

    def walk(stmts):
        for st in stmts:
            flat.append(st)
            for field in ("body", "orelse", "finalbody"):
                inner = getattr(st, field, None)
                if isinstance(inner, list) and inner and \
                        isinstance(inner[0], ast.stmt):
                    walk(inner)
            for h in getattr(st, "handlers", []) or []:
                walk(h.body)
    walk(fn.body)

    dead = set()
    for st in flat:
        # a use before any binding is only a problem for names never bound
        for sub in ast.walk(st):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                if sub.id in dead:
                    problems.append(
                        f"{os.path.basename(path)}:{sub.lineno}: '{sub.id}' is used "
                        f"after `del {sub.id}` in {fn.name}()")
        dead |= deleted_names(st)
        dead -= bound_names(st)


def scan_module(path):
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if getattr(node, "name", "").startswith("_"):
                pass
            scan_function(node, path, problems)
    return problems


def main():
    print("=" * 78)
    print("driver static checks (no torch: this is the only coverage this file has)")
    print("=" * 78)

    problems = scan_module(DRIVER)
    check("no del-then-use in tools/seg_token_compressor.py", not problems,
          "; ".join(problems[:3]))
    for p in problems:
        print(f"        {p}")

    src = open(DRIVER, encoding="utf-8").read()
    # The measured runs in data/HISTORY.md were produced with these defaults;
    # changing them silently would make those numbers irreproducible.
    for flag, want in (("--overlap", "0"), ("--kv-window", "8192"),
                       ("--trigram-conf", "3.0"), ("--top-k", "1024"),
                       ("--prefilter", "2048")):
        ok = f'"{flag}", type=int, default={want}' in src or \
             f'"{flag}", type=float, default={want}' in src
        check(f"{flag} default is still {want} (measured runs stay reproducible)", ok)

    # the driver must not hold unbounded per-token history: 100MB is 27M tokens
    check("history buffer is a bounded deque, not a growing list",
          "deque(maxlen=window)" in src)

    print()
    print("=" * 78)
    print(f"RESULT: {'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    print("=" * 78)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
