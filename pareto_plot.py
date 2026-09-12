"""Pareto frontier: speed vs ratio (measured, quiet box, 100KB slice).

Run: python pareto_plot.py  (CPU only, reads the table below)
Output: pareto.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, seconds/100KB, bpb) -- all measured 2026-09-13, quiet box,
# same base (floor1e-6/K1024/procs+pipeline), verified 220/220 each.
PTS = [
    ("ov4096\ncrown", 19.4, 0.9139),
    ("BC7", 19.1, 0.9140),
    ("ov6144", 34.4, 0.9141),
    ("TRI0", 18.3, 0.9146),
    ("ov3072", 16.9, 0.9166),
    ("ov2048\nknee", 14.2, 0.9194),
    ("ov1024", 13.5, 0.9237),
    ("ov0", 12.0, 0.9268),
]
SOTA_BPB = 0.9389          # Nacrith paper, full file
NNCP_S = 100 / 3.25        # NNCP v2 3.25 KB/s -> s per 100KB

fig, ax = plt.subplots(figsize=(9, 6))
xs = [p[1] for p in PTS]
ys = [p[2] for p in PTS]
ax.axhline(SOTA_BPB, color="red", linestyle="--", linewidth=1,
           label="SOTA ratio 0.9389 (worse above)")
ax.axvline(NNCP_S, color="orange", linestyle="--", linewidth=1,
           label="NNCP v2 speed 3.25KB/s (slower right)")
order = sorted(range(len(PTS)), key=lambda i: PTS[i][1])
ax.plot([xs[i] for i in order], [ys[i] for i in order],
        color="gray", linewidth=1, alpha=0.6)
for i, (lab, x, y) in enumerate(PTS):
    ms = 11 if i in (0, 5) else 7
    mk = "*" if i in (0, 5) else "o"
    ax.scatter([x], [y], s=ms * 10, marker=mk, color="steelblue",
               edgecolors="black", zorder=3)
    ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 6),
                fontsize=9)
ax.set_xlabel("seconds per 100KB (quiet box)  ->  faster left")
ax.set_ylabel("bits/byte (lower better)")
ax.set_title("Speed vs ratio Pareto (SmolLM2-135M ensemble, enwik8 slice)")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.3)

ax2 = ax.secondary_xaxis("top", functions=(lambda s: 100 / s, lambda k: 100 / k))
ax2.set_xlabel("KB/s (faster right)")

fig.tight_layout()
fig.savefig("pareto.png", dpi=120)
print("saved pareto.png")

# balance verdict (printed, also the ledger source)
print("knee: ov2048 0.9194 @ 14.2s (7.0KB/s)")
print("crown: ov4096 0.9139 @ 19.4s (5.1KB/s)")
print("fastest sub-SOTA: ov0 0.9268 @ 12.0s (8.3KB/s)")
