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
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return {node.name}           # its parameters are its own scope
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)) \
                and sub is not node:
            if isinstance(sub.ctx, ast.Store):
                out.add(sub.id)
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and sub is not node:
            out.add(sub.name)
        if isinstance(sub, ast.Import):
            for a in sub.names:
                out.add((a.asname or a.name).split(".")[0])
        if isinstance(sub, ast.ImportFrom):
            for a in sub.names:
                out.add(a.asname or a.name)
    return out


def early_bound_names(node):
    """Names bound WITHIN the same statement but visible to its own expressions.

    Comprehension targets and lambda parameters are read back inside the very
    statement that introduces them (`{k: v for k, v in cfg.items()}`), so they
    must count as bound before that statement's loads are scanned. Ordinary
    assignments deliberately do NOT go here: `x = x + 1` before any binding of x
    is a genuine use-before-binding and must stay reportable.
    """
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.comprehension):
            out |= {t.id for t in ast.walk(sub.target)
                    if isinstance(t, ast.Name)}
        if isinstance(sub, ast.Lambda):
            a = sub.args
            out |= {x.arg for x in (a.posonlyargs + a.args + a.kwonlyargs)}
    return out


def shallow_loads(node):
    """Names read by THIS statement only, not by nested statement bodies.

    The flat walk emits nested statements separately, so a Load inside an `if`
    body must not be attributed to the `if` line -- doing that flagged
    variables assigned and used within the same block (`el`, `eta` in the
    progress printer) as use-before-binding. Expressions still descend fully,
    including lambdas and comprehensions.
    """
    out = []

    def visit(n, root):
        for child in ast.iter_child_nodes(n):
            if isinstance(child, ast.stmt) and child is not root:
                continue                      # separate flat entry
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                out.append(child)
            visit(child, root)
    visit(node, node)
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
    """Walk a function body in source order, tracking del-then-use and
    use-before-any-binding.

    The second check exists because this file's first fix introduced its own
    twin: `fingerprint` was defined in the results block but first used ~60
    lines earlier in the output path, which is the same crash at a different
    moment. Anything bound ANYWHERE in the function but read before any binding
    is reported.
    """
    flat = []

    def walk(stmts):
        for st in stmts:
            # Nested defs and lambdas have their OWN scope: their bodies are
            # scanned as separate functions (scan_module walks every def), and
            # descending here would blame the outer function for their local
            # parameters. `bound_names` still records the nested def's NAME.
            if not isinstance(st, ast.FunctionDef):
                flat.append(st)
            else:
                flat.append(st)
            for field in ("body", "orelse", "finalbody"):
                inner = getattr(st, field, None)
                if isinstance(inner, list) and inner and \
                        isinstance(inner[0], ast.stmt):
                    if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue          # separate scope
                    walk(inner)
            for h in getattr(st, "handlers", []) or []:
                walk(h.body)
    walk(fn.body)

    # names that are declared nonlocal/global are bound elsewhere
    declared = set()
    for st in flat:
        if isinstance(st, (ast.Global, ast.Nonlocal)):
            declared.update(st.names)

    own_args = {a.arg for a in (fn.args.posonlyargs + fn.args.args +
                                fn.args.kwonlyargs)}
    all_bound = set(declared) | own_args
    for st in flat:
        all_bound |= bound_names(st)

    live = set(declared) | own_args
    dead = set()
    for st in flat:
        live_here = live | early_bound_names(st)
        for sub in shallow_loads(st):
            if True:
                if sub.id in dead:
                    problems.append(
                        f"{os.path.basename(path)}:{sub.lineno}: '{sub.id}' is used "
                        f"after `del {sub.id}` in {fn.name}()")
                elif sub.id in all_bound and sub.id not in live_here:
                    problems.append(
                        f"{os.path.basename(path)}:{sub.lineno}: '{sub.id}' is read "
                        f"before any binding of it in {fn.name}()")
        live |= bound_names(st)
        dead |= deleted_names(st)
        dead -= bound_names(st)


def scan_source(src, name="<string>"):
    """Report names used after `del`, and names read before any binding."""
    tree = ast.parse(src, filename=name)
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scan_function(node, name, problems)
    return problems


def scan_module(path):
    return scan_source(open(path, encoding="utf-8").read(), path)


# The checker is itself checked: each snippet encodes a failure mode that
# actually happened in this session, and the last one must stay silent. A guard
# that cannot be shown to fire is decoration.
SELF_TESTS = [
    ("del-then-use (the crash after a full encode+decode)", """
def main():
    nl = build()
    plan, timings = run(nl)
    del nl
    result = dict(prefill=nl.note)
    print(result)
"""),
    ("use-before-binding (the fingerprint bug)", """
def main():
    zpath = f"out_{fingerprint}.zllm"
    fingerprint = "k1024"
    write(zpath)
"""),
    ("nested function scopes must not be blamed on the outer function", """
def main():
    x = 1
    def helper(inp, step):
        el = time.time()
        if el > 0:
            eta = el * step
            print(f"{el} {eta} {inp}")
        return x
    helper(1, 2)
    d = {k: v for k, v in {"a": 1}.items()}
    print(d)
"""),
]


def main():
    print("=" * 78)
    print("driver static checks (no torch: this is the only coverage this file has)")
    print("=" * 78)

    print("  -- the checker itself --")
    for label, snippet in SELF_TESTS:
        found = scan_source(snippet)
        must_flag = "must not be blamed" not in label
        good = bool(found) if must_flag else not found
        check(f"self-test: {label}", good,
              (found[0] if found else "") if must_flag else
              "; ".join(found[:2]))
        if must_flag and found:
            print(f"        {found[0]}")

    print()
    print("  -- the real driver --")
    problems = scan_module(DRIVER)
    check("tools/seg_token_compressor.py is clean", not problems,
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
