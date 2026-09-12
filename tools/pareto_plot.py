"""Pareto frontier: speed vs ratio (measured, 100KB slices).

Run from repo root: python tools/pareto_plot.py  (CPU only)
Output: pareto.png (repo root, referenced by README)
X axis is KB/s so 1MB runs are directly comparable.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, KB/s, bpb). All 100KB slices verified (220-ish lossless each).
# off50 = representative article region; off75 = hard region.
OFF50 = [
    ("ov4096\ncrown", 100 / 19.4, 0.9139),
    ("BC7", 100 / 19.1, 0.9140),
    ("ov6144", 100 / 34.4, 0.9141),
    ("TRI0", 100 / 18.3, 0.9146),
    ("ov5120", 100 / 24.6, 0.9164),
    ("ov3072", 100 / 16.9, 0.9166),
    ("ov2560", 100 / 16.1, 0.9169),
    ("ov2048\nknee", 100 / 14.2, 0.9194),
    ("ov1536", 100 / 13.8, 0.9237),
    ("ov512", 100 / 13.1, 0.9232),
    ("ov1024", 100 / 13.5, 0.9237),
    ("ov0", 100 / 12.0, 0.9268),
]
OFF75 = [
    ("ov4096", 100 / 18.6, 0.9662),
    ("ov3072", 100 / 15.6, 0.9653),
    ("ov2048", 100 / 14.2, 0.9687),
    ("ov1024", 100 / 12.4, 0.9723),
    ("ov0", 100 / 11.3, 0.9785),
]
# 1MB flagships (KB/s comparable on this axis)
MB1 = [
    ("1MB@off50", 1024 / 256.0, 0.9222),
    ("1MB@off25", 1024 / 262.7, 0.9076),
]
SOTA_BPB = 0.9389          # Nacrith paper, full file
NNCP_KBS = 3.25            # NNCP v2 reference speed

fig, ax = plt.subplots(figsize=(10, 6.5))
ax.axhline(SOTA_BPB, color="red", linestyle="--", linewidth=1,
           label="SOTA ratio 0.9389 (worse above)")
ax.axvline(NNCP_KBS, color="orange", linestyle="--", linewidth=1,
           label="NNCP v2 speed 3.25KB/s (slower left)")


def curve(pts, color, label):
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    order = sorted(range(len(pts)), key=lambda i: pts[i][1])
    ax.plot([xs[i] for i in order], [ys[i] for i in order],
            color=color, linewidth=1.2, alpha=0.7, label=label)
    for lab, x, y in pts:
        star = lab.endswith("crown") or lab.endswith("knee")
        ax.scatter([x], [y], s=140 if star else 70,
                   marker="*" if star else "o", color=color,
                   edgecolors="black", zorder=3)
        ax.annotate(lab, (x, y), textcoords="offset points",
                    xytext=(6, 6), fontsize=8)


curve(OFF50, "steelblue", "off50 (article region, 12 pts)")
curve(OFF75, "seagreen", "off75 (hard region, 5 pts)")
for lab, x, y in MB1:
    ax.scatter([x], [y], s=160, marker="D", color="purple",
               edgecolors="black", zorder=4)
    ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, -12),
                fontsize=8)

ax.set_xlabel("KB/s (faster right)")
ax.set_ylabel("bits/byte (lower better)")
ax.set_title("Speed vs ratio (SmolLM2-135M ensemble, measured)")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 10)

fig.tight_layout()
fig.savefig("pareto.png", dpi=120)
print("saved pareto.png")
print(f"points: {len(OFF50) + len(OFF75) + len(MB1)}")
