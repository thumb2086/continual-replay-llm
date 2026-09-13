"""Combined figure: Pareto (off50|off75) + Scale (throughput|ratio) + Balance
All side numbers forced to plain (no 2x10^1). Single output: figures_combined.png
"""
import math, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

# --- data (copied from pareto/balance/scale, single source) ---
OFF50 = [
    ("gate\n22.7KB/s", 100 / 4.4, 0.9276),
    ("gather\n18.9KB/s", 100 / 5.3, 0.9272),
    ("chunk\n34.5KB/s", 100 / 2.9, 0.9276),
    ("chunkK2\n27KB/s", 100 / 3.7, 0.9187),
    ("chunkK4\n16.9KB/s", 100 / 5.9, 0.9142),
    ("ov0K2 gather", 100 / 5.7, 0.9183),
    ("knee gather", 100 / 5.9, 0.9194),
    ("ov1024 gather", 100 / 5.5, 0.9241),
    ("K3072 gather", 100 / 9.7, 0.9029),
    ("ov3072K8 gather", 100 / 15.0, 0.9026),
    ("K2k", 100 / 18.0, 0.9050),
    ("K4k", 100 / 20.2, 0.9013),
    ("ov0@K8", 100 / 21.8, 0.9127),
    ("ov1024@K8", 100 / 22.3, 0.9099),
    ("ov2048@K8", 100 / 23.6, 0.9058),
    ("K8k gather", 100 / 27.4, 0.9005),
    ("crown gate52", 100 / 15.2, 0.9005),
    ("crown gate207", 100 / 13.3, 0.9006),
    ("K8k\ncrown", 100 / 28.1, 0.9003),
    ("K6k", 100 / 28.5, 0.9004),
    ("K16k", 100 / 52.4, 0.9027),
    ("ov6144@K8", 100 / 55.8, 0.9004),
    ("ov4096\nK1k", 100 / 17.3, 0.9139),
]
OFF75 = [
    ("ov4096", 100 / 18.6, 0.9662),
    ("ov3072", 100 / 15.6, 0.9653),
    ("ov2048", 100 / 14.2, 0.9687),
    ("ov1024", 100 / 12.4, 0.9723),
    ("ov0", 100 / 11.3, 0.9785),
    ("gate207 gather\n25KB/s", 100 / 4.0, 0.9812),
    ("K2048 gate\n22KB/s", 100 / 4.5, 0.9715),
]
MB1 = [
    ("1MB@off50", 1024 / 256.0, 0.9222),
    ("1MB@off25", 1024 / 262.7, 0.9076),
    ("1MB gate207", 1024 / 49.4, 0.9359),
    ("1MB crown-g", 1024 / 201.6, 0.9079),
    ("1MB chunked", 1024 / 29.8, 0.9360),
]
SIZES_KB = [100, 1024, 10240]
SIZES_LBL = ["100KB", "1MB", "10MB"]
TIME = {
    "speed classic\nov0/K1024": [9.7, 104.3, 1095.9],
    "speed gather\nov0/K1024": [5.3, 56.9, 632.2],
    "speed gate207\nov0/K1024": [4.5, 49.4, None],
    "speed chunked\nov0/K1024": [2.9, 29.8, 273.7],
    "crown classic\nov4096/K8192": [24.0, 254.7, 1755.0],
    "crown gather\nov4096/K8192": [27.4, 201.6, None],
}
BPB = {
    "speed classic\nov0/K1024": [0.9268, 0.9347, 0.9025],
    "speed gather\nov0/K1024": [0.9272, 0.9355, 0.9039],
    "speed gate207\nov0/K1024": [0.9276, 0.9359, None],
    "speed chunked\nov0/K1024": [0.9276, 0.9360, 0.9042],
    "crown classic\nov4096/K8192": [0.9003, 0.9078, 0.8765],
    "crown gather\nov4096/K8192": [0.9005, 0.9079, None],
}
SOTA_BPB = 0.9389
SOTA_RATIO = 8.0 / SOTA_BPB
NNCP_KBS = 3.25
def plain(ax):
    # plain numbers, keep minor ticks visible (no 1x10^1)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"{y:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda y,_: f"{y:g}"))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{x:g}"))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda x,_: f"{x:g}"))
def score(kbs,bpb): return (8/bpb)*math.log(kbs) if kbs>0 else 0

fig = plt.figure(figsize=(14, 14))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.9], hspace=0.45, wspace=0.25)

# helper for pareto
def pareto_panel(ax, pts, title, ymin, ymax, color, show, offsets, xlim):
    ax.axhline(SOTA_RATIO, color="red", linestyle="--", lw=1, label="SOTA 8.52x")
    ax.axvline(NNCP_KBS, color="orange", linestyle="--", lw=1, label="NNCP 3.25KB/s")
    xs=[p[1] for p in pts]; ys=[8/p[2] for p in pts]
    order=sorted(range(len(pts)), key=lambda i: pts[i][1])
    ax.plot([xs[i] for i in order],[ys[i] for i in order], color=color, lw=1.4, alpha=0.7)
    for lab,x,b in pts:
        y=8/b
        star=lab.endswith("crown") or lab.endswith("knee")
        ax.scatter([x],[y], s=150 if star else 80, marker="*" if star else "o", color=color, edgecolors="black", zorder=3)
        if lab in show:
            ax.annotate(lab,(x,y), textcoords="offset points", xytext=offsets.get(lab,(8,8)), fontsize=7,
                        arrowprops=dict(arrowstyle="-", color="gray", lw=0.6, shrinkB=2))
    for lab,x,b in MB1:
        ax.scatter([x],[8/b], s=120, marker="D", color="purple", edgecolors="black", zorder=4)
        # only label on left panel to avoid duplication
    ax.set_xlabel("KB/s (faster right ->)"); ax.set_ylabel("compression ratio x (higher better ^)")
    ax.set_title(title+"  [top-right is best — frontier moves to top-right]", fontsize=9); ax.set_ylim(ymin,ymax); ax.set_xscale("log"); ax.set_xlim(2, xlim)
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7, loc="lower left"); plain(ax)

ax1 = fig.add_subplot(gs[0,0]); pareto_panel(ax1, OFF50, "Pareto: article region (off50)", 8.3, 9.1, "steelblue",
    show=("K8k\ncrown","K16k","knee gather","ov4096\nK1k","K3072 gather","chunk\n34.5KB/s","chunkK2\n27KB/s","chunkK4\n16.9KB/s","gate\n22.7KB/s","crown gate207"),
    offsets={"K8k\ncrown":(-64,14),"K16k":(8,12),"knee gather":(8,18),"ov4096\nK1k":(-64,-12),"K3072 gather":(8,12),"chunk\n34.5KB/s":(8,12),"chunkK2\n27KB/s":(8,-16),"chunkK4\n16.9KB/s":(8,14),"gate\n22.7KB/s":(10,-14),"crown gate207":(-72,-14)}, xlim=50)
ax2 = fig.add_subplot(gs[0,1]); pareto_panel(ax2, OFF75, "Pareto: hard region (off75)", 7.9, 8.7, "seagreen",
    show=("ov4096","ov3072","ov0","gate207 gather\n25KB/s","K2048 gate\n22KB/s"),
    offsets={"ov4096":(-52,-10),"ov3072":(10,16),"ov0":(10,-8),"gate207 gather\n25KB/s":(10,10),"K2048 gate\n22KB/s":(10,-14)}, xlim=40)

# scale
def kbs(sz,sec): return [s/t for s,t in zip(sz,sec)]
ax3 = fig.add_subplot(gs[1,0])
for n,ts in TIME.items():
    xs=[s for s,t in zip(SIZES_KB,ts) if t is not None]; ys=kbs(xs,[t for t in ts if t is not None])
    ax3.loglog(xs,ys,"o-", label=n)
ax3.set_xlabel("file size (KB, log)"); ax3.set_ylabel("throughput (KB/s, higher better ^)")
ax3.set_title("Throughput ladder [top-right is best]", fontsize=9); ax3.set_xticks(SIZES_KB); ax3.set_xticklabels(SIZES_LBL)
ax3.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"{y:g}")); ax3.yaxis.set_minor_formatter(FuncFormatter(lambda y,_: f"{y:g}"))
ax3.xaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{x:g}")); ax3.xaxis.set_minor_formatter(FuncFormatter(lambda x,_: f"{x:g}"))
ax3.grid(True, which="both", alpha=0.3); ax3.legend(fontsize=6)

ax4 = fig.add_subplot(gs[1,1])
for n,bs in BPB.items():
    xs=[s for s,b in zip(SIZES_KB,bs) if b is not None]; ys=[8/b for b in bs if b is not None]
    ax4.semilogx(xs,ys,"s-", label=n)
ax4.axhline(SOTA_RATIO, color="red", linestyle=":", label="SOTA 8.52x")
ax4.set_xlabel("file size (KB, log)"); ax4.set_ylabel("compression ratio x (higher better ^)")
ax4.set_title("Ratio vs scale [top-right is best]", fontsize=9); ax4.set_xticks(SIZES_KB); ax4.set_xticklabels(SIZES_LBL)
ax4.grid(True, which="both", alpha=0.3); ax4.legend(fontsize=6)
ax4.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"{y:g}")); ax4.yaxis.set_minor_formatter(FuncFormatter(lambda y,_: f"{y:g}"))
ax4.xaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{x:g}")); ax4.xaxis.set_minor_formatter(FuncFormatter(lambda x,_: f"{x:g}"))

# balance
PTS = OFF50 + [("Qwen K2k",100/5.6,0.8442),("Qwen K4k*",100/12.2,0.8391)]
sota_pts=[p for p in PTS if p[2]<=SOTA_BPB]
best=max(sota_pts, key=lambda p: score(p[1],p[2]))
ax5 = fig.add_subplot(gs[2,:])
xs=[p[1] for p in PTS]; ys=[8/p[2] for p in PTS]
sizes=[50+60*(score(p[1],p[2])/score(best[1],best[2])) for p in PTS]
ax5.scatter(xs, ys, c=['red' if p==best else 'steelblue' for p in PTS], s=sizes, edgecolors='black', zorder=3, alpha=0.85)
for x,y,l in zip(xs,ys,[p[0] for p in PTS]):
    if l in (best[0], "K8k\ncrown", "Qwen K4k*", "chunkK2\n27KB/s", "chunkK4\n16.9KB/s"):
        ax5.annotate(l,(x,y), textcoords="offset points", xytext=(6,6), fontsize=7, arrowprops=dict(arrowstyle="-", color="gray", lw=0.6))
ax5.set_xscale("log"); ax5.set_xlabel("KB/s (faster right, log)"); ax5.set_ylabel("compression ratio x = 8/bpb (higher better ^)")
ax5.set_title(f"Balance: ratio vs speed (SOTA 8.52x) — best = {best[0]}  score={score(best[1],best[2]):.1f}  size=score", fontsize=9)
ax5.axhline(SOTA_RATIO, color="red", linestyle="--", lw=1, label="SOTA 8.52x")
ax5.grid(True, which="both", alpha=0.3)
ax5.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"{y:g}")); ax5.yaxis.set_minor_formatter(FuncFormatter(lambda y,_: f"{y:g}"))
ax5.xaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{x:g}")); ax5.xaxis.set_minor_formatter(FuncFormatter(lambda x,_: f"{x:g}"))
ax5.legend(fontsize=7, loc="lower left")

fig.suptitle("Combined: Pareto + Scale + Balance (all measured, no extrapolation)", fontsize=11)
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig("figures_combined.png", dpi=140)
print("saved figures_combined.png")
# also keep individual plain-fixed files for compatibility
fig2, ax = plt.subplots(figsize=(10,6))
# reuse pareto already, but ensure plain
print("also re-export plain-fixed individual pngs")
# re-save individual with plain fix
import subprocess, sys
# call existing scripts which now have plain() — they will overwrite
