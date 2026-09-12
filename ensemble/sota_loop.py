"""SOTA loop driver: iterate ideas until bpc < TARGET_BPB with verified lossless.

Usage:
  python sota_loop.py --add <id> "<script> <args>" "<result_json>" "<env KEY=VAL;...>"
  python sota_loop.py --run-next      # run oldest pending idea, record result
  python sota_loop.py --auto          # keep running until target met or queue empty
  python sota_loop.py --status        # show ledger + gaps

Targets (enwik8 100KB mid-slice unless noted):
  RATIO_TARGET_BPB = 0.9389  (public SOTA; stop only when strictly below)
  SPEED_TARGET_KBS = 14.5    (10x our v2 baseline ~1.45KB/s; neural-class SOTA beat)
Stop rule: BOTH met on the same run, verified_ok>0 and verified_fail==0.
"""
import json, os, subprocess, sys, time

LEDGER = "./data/sota_loop.json"
RATIO_TARGET_BPB = 0.9389
SPEED_TARGET_KBS = 14.5

def load():
    if os.path.exists(LEDGER):
        return json.load(open(LEDGER))
    return {"ideas": [], "history": [], "done": False}

def save(d):
    # Atomic write (tmp + replace): a kill mid-write must never leave a
    # 0-byte ledger again (2026-09-12 incident: 50 entries rebuilt by hand).
    tmp = LEDGER + ".tmp"
    json.dump(d, open(tmp, "w"), indent=1)
    os.replace(tmp, LEDGER)

def cmd_add(args):
    _id, script_cmd, result_json, env = args[0], args[1], args[2], (args[3] if len(args) > 3 else "")
    d = load()
    if any(i["id"] == _id for i in d["ideas"]):
        print(f"idea {_id} already queued"); return
    d["ideas"].append({"id": _id, "cmd": script_cmd, "result": result_json,
                       "env": env, "status": "pending"})
    save(d); print(f"queued {_id}")

def run_one(idea):
    env = dict(os.environ)
    for kv in idea["env"].split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1); env[k.strip()] = v.strip()
    t0 = time.time()
    # Stream worker output live (no capture): background logs show progress
    # as it happens instead of going dark for hours. Result still comes
    # from the JSON file the worker writes.
    p = subprocess.run(idea["cmd"], shell=True, env=env)
    wall = time.time() - t0
    if p.returncode != 0:
        return {"id": idea["id"], "status": "error", "wall_s": round(wall, 1)}
    try:
        r = json.load(open(idea["result"]))
    except Exception as e:
        return {"id": idea["id"], "status": "no-result", "wall_s": round(wall, 1),
                "err": str(e)}
    bpb = r.get("bpb") or r.get("bpc")
    n, el = r.get("n_bytes", 0), r.get("elapsed_s", wall)
    kbs = (n / 1024) / el if el else 0
    return {"id": idea["id"], "status": "ok", "bpb": round(bpb, 4),
            "kbs": round(kbs, 2), "verified_ok": r.get("verified_ok"),
            "verified_fail": r.get("verified_fail"), "elapsed_s": round(el, 1)}

def cmd_run_next():
    d = load()
    nxt = next((i for i in d["ideas"] if i["status"] == "pending"), None)
    if not nxt:
        print("queue empty"); return False
    print(f"=== running {nxt['id']}: {nxt['cmd']} [{nxt['env']}] ===")
    nxt["status"] = "running"; save(d)
    res = run_one(nxt)
    nxt["status"] = res["status"]; save(d)
    d["history"].append(res); save(d)
    print(json.dumps(res, indent=1))
    if res.get("status") == "ok" and (res.get("verified_fail") or 0) == 0:
        gap_r = res["bpb"] - RATIO_TARGET_BPB
        gap_s = SPEED_TARGET_KBS - res["kbs"]
        print(f"gap ratio: {gap_r:+.4f} bpb | gap speed: {gap_s:+.2f} KB/s")
        if gap_r < 0 and gap_s <= 0:
            d["done"] = True; save(d)
            print("*** SOTA BEATEN ON BOTH AXES ***")
            return True
    return False

def cmd_auto():
    while True:
        d = load()
        if d["done"]:
            print("done flag set, stopping"); return
        if not any(i["status"] == "pending" for i in d["ideas"]):
            print("queue empty, stopping (target NOT met)"); return
        cmd_run_next()

def cmd_status():
    d = load()
    print(f"done={d['done']} ratio_target={RATIO_TARGET_BPB} speed_target={SPEED_TARGET_KBS}KB/s")
    for h in d["history"]:
        print(" ", json.dumps(h))

def cmd_reset_stuck():
    # A kill leaves ideas stuck in 'running'; --auto only picks 'pending'.
    # Run this after any manual Stop-Process before relaunching.
    d = load()
    n = 0
    for i in d["ideas"]:
        if i["status"] == "running":
            i["status"] = "pending"
            n += 1
    save(d)
    print(f"reset {n} stuck idea(s) to pending")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    if sys.argv[1] == "--add": cmd_add(sys.argv[2:])
    elif sys.argv[1] == "--run-next": cmd_run_next()
    elif sys.argv[1] == "--auto": cmd_auto()
    elif sys.argv[1] == "--status": cmd_status()
    elif sys.argv[1] == "--reset-stuck": cmd_reset_stuck()
    else: print(__doc__)
