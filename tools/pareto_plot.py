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
    ("K16k", 100 / 52.4, 0.9027),
    ("ov6144@K8", 100 / 55.8, 0.9004),
    ("K8k\ncrown", 100 / 24.0, 0.9003),
    ("K6k", 100 / 28.5, 0.9004),
    ("ov2048@K8", 100 / 23.6, 0.9058),
    ("ov1024@K8", 100 / 22.3, 0.9099),
    ("ov0@K8", 100 / 21.8, 0.9127),
    ("K4k", 100 / 20.2, 0.9013),
    ("K2k", 100 / 18.0, 0.9050),
    ("ov4096\nK1k", 100 / 17.3, 0.9139),
    ("BC7", 100 / 19.1, 0.9140),
    ("ov6144", 100 / 34.4, 0.9141),
    ("TRI0", 100 / 18.3, 0.9146),
    ("ov5120", 100 / 24.6, 0.9164),
    ("ov3072", 100 / 16.9, 0.9166),
    ("ov2560", 100 / 16.1, 0.9169),
    ("ov2048\nknee", 100 / 12.0, 0.9194),
    ("ov1536", 100 / 13.8, 0.9237),
    ("ov512", 100 / 13.1, 0.9232),
    ("ov1024", 100 / 13.5, 0.9237),
    ("ov0\n10KB/s", 100 / 9.8, 0.9268),
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


def _base(title):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(SOTA_BPB, color="red", linestyle="--", linewidth=1,
               label="SOTA ratio 0.9389 (worse above)")
    ax.axvline(NNCP_KBS, color="orange", linestyle="--", linewidth=1,
               label="NNCP v2 speed 3.25KB/s (slower left)")
    ax.set_xlabel("KB/s (faster right)")
    ax.set_ylabel("bits/byte (lower better)")
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), fontsize=9)
    ax.grid(True, alpha=0.3)
    return fig, ax


def _anno(ax, lab, x, y, dx, dy):
    ax.annotate(lab, (x, y), textcoords="offset points", xytext=(dx, dy),
                fontsize=9,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.7,
                                shrinkB=2))


def _dots(ax, pts, color, show, offsets):
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    order = sorted(range(len(pts)), key=lambda i: pts[i][1])
    ax.plot([xs[i] for i in order], [ys[i] for i in order],
            color=color, linewidth=1.4, alpha=0.7)
    for lab, x, y in pts:
        star = lab.endswith("crown") or lab.endswith("knee")
        ax.scatter([x], [y], s=150 if star else 80,
                   marker="*" if star else "o", color=color,
                   edgecolors="black", zorder=3)
        if lab in show:
            _anno(ax, lab, x, y, *offsets.get(lab, (8, 8)))


def _diamonds(ax):
    for lab, x, y in MB1:
        ax.scatter([x], [y], s=180, marker="D", color="purple",
                   edgecolors="black", zorder=4)
        _anno(ax, lab, x, y, 10, 10 if "off50" in lab else -20)


# ---- plot 1: off50 frontier ----
fig, ax = _base("Speed vs ratio, article region (SmolLM2-135M, measured)")
_dots(ax, OFF50, "steelblue", show=("K8k\ncrown", "K16k", "ov0@K8", "ov2048@K8",
                                    "ov4096\nK1k", "ov2048\nknee", "ov0\n10KB/s"),
      offsets={"K8k\ncrown": (-60, -20), "K16k": (8, -18), "ov0@K8": (8, 8),
               "ov2048@K8": (-64, 8), "ov4096\nK1k": (-64, 10),
               "ov2048\nknee": (10, -22), "ov0\n10KB/s": (8, 10)})
_diamonds(ax)
ax.set_xlim(0, 11)
fig.tight_layout()
fig.savefig("pareto_off50.png", dpi=120)
print("saved pareto_off50.png")

# ---- plot 2: off75 hard region ----
fig, ax = _base("Speed vs ratio, hard region (SmolLM2-135M, measured)")
_dots(ax, OFF75, "seagreen", show=("ov4096", "ov3072", "ov0"),
      offsets={"ov4096": (-52, 10), "ov3072": (10, -16), "ov0": (10, 8)})
_diamonds(ax)
ax.set_xlim(0, 10.5)
fig.tight_layout()
fig.savefig("pareto_off75.png", dpi=120)
print("saved pareto_off75.png")
print(f"points: {len(OFF50) + len(OFF75) + len(MB1)}")
