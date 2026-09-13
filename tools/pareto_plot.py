"""Pareto frontier: speed vs compression ratio (measured, 100KB slices).

Run from repo root: python tools/pareto_plot.py  (CPU only)
Output: pareto_off50.png / pareto_off75.png (repo root, referenced by README)
X = KB/s (faster right), Y = compression ratio 8/bpb (higher better),
so the ideal corner is TOP-RIGHT. X axis is KB/s so 1MB runs comparable.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
def _plain(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"{y:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{x:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())

# (label, KB/s, bpb). Fresh era only: current-code numbers (gather default
# on unless noted). Pre-flash timings (BC7/ov6144/TRI0/ov5120/ov3072/ov2560/
# ov1536/ov512/ov1024/ov0-classic) REMOVED 2026-09-14 (stale scheduler era).
# off50 = representative article region; off75 = hard region.
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
# 1MB flagships (KB/s comparable on this axis)
MB1 = [
    ("1MB@off50", 1024 / 256.0, 0.9222),
    ("1MB@off25", 1024 / 262.7, 0.9076),
    ("1MB gate207", 1024 / 49.4, 0.9359),
    ("1MB crown-g", 1024 / 201.6, 0.9079),
    ("1MB chunked", 1024 / 29.8, 0.9360),
]
SOTA_BPB = 0.9389          # Nacrith paper, full file
SOTA_RATIO = 8.0 / SOTA_BPB  # 8.52x: higher-is-better twin of the SOTA line
NNCP_KBS = 3.25            # NNCP v2 reference speed


def _base(title, ymin, ymax):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(SOTA_RATIO, color="red", linestyle="--", linewidth=1,
               label="SOTA 8.52x (worse below)")
    ax.axvline(NNCP_KBS, color="orange", linestyle="--", linewidth=1,
               label="NNCP v2 speed 3.25KB/s (slower left)")
    ax.set_xlabel("KB/s (faster right ->)")
    ax.set_ylabel("compression ratio x (higher better ^)")
    ax.set_title(title + "  [top-right is best — frontier moves to top-right]")
    ax.set_ylim(ymin, ymax)
    ax.set_xscale("log")
    ax.legend(loc="lower left", bbox_to_anchor=(0.01, 0.01), fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    _plain(ax)
    return fig, ax


def _anno(ax, lab, x, y, dx, dy):
    ax.annotate(lab, (x, y), textcoords="offset points", xytext=(dx, dy),
                fontsize=9,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.7,
                                shrinkB=2))


def _dots(ax, pts, color, show, offsets):
    xs = [p[1] for p in pts]
    ys = [8.0 / p[2] for p in pts]
    order = sorted(range(len(pts)), key=lambda i: pts[i][1])
    ax.plot([xs[i] for i in order], [ys[i] for i in order],
            color=color, linewidth=1.4, alpha=0.7)
    for lab, x, _bpb in pts:
        y = 8.0 / _bpb
        star = lab.endswith("crown") or lab.endswith("knee")
        ax.scatter([x], [y], s=150 if star else 80,
                   marker="*" if star else "o", color=color,
                   edgecolors="black", zorder=3)
        if lab in show:
            _anno(ax, lab, x, y, *offsets.get(lab, (8, 8)))


def _diamonds(ax, xoff=10):
    for lab, x, _bpb in MB1:
        y = 8.0 / _bpb
        ax.scatter([x], [y], s=160, marker="D", color="purple",
                   edgecolors="black", zorder=4)
        # spread 1MB labels: off50 cluster crowded, push up; chunked/gate207 push out
        if "chunked" in lab: dx, dy = 12, 12
        elif "gate207" in lab: dx, dy = 14, -16
        elif "off50" in lab: dx, dy = -30, 18
        elif "off25" in lab: dx, dy = 8, -22
        else: dx, dy = xoff, 14
        _anno(ax, lab, x, y, dx, dy)


# ---- plot 1: off50 frontier ----
fig, ax = _base("Speed vs ratio, article region (SmolLM2-135M, measured)",
                8.3, 9.1)
# reduced labels to avoid crowding; keep frontier extremes + balanced
_dots(ax, OFF50, "steelblue", show=("K8k\ncrown", "K3072 gather",
                                    "chunk\n34.5KB/s", "chunkK2\n27KB/s",
                                    "chunkK4\n16.9KB/s",
                                    "crown gate207"),
      offsets={"K8k\ncrown": (-78, 16), "K3072 gather": (10, 14),
               "chunk\n34.5KB/s": (10, 12), "chunkK2\n27KB/s": (10, 10),
               "chunkK4\n16.9KB/s": (10, -18), "crown gate207": (-82, -16)})
_diamonds(ax, xoff=12)
ax.set_xlim(2, 55)
fig.tight_layout()
fig.savefig("pareto_off50.png", dpi=120)
print("saved pareto_off50.png")

# ---- plot 2: off75 hard region ----
fig, ax = _base("Speed vs ratio, hard region (SmolLM2-135M, measured)",
                7.9, 8.7)
# spread the two close gate labels vertically
_dots(ax, OFF75, "seagreen", show=("ov4096", "ov0",
                                    "gate207 gather\n25KB/s",
                                    "K2048 gate\n22KB/s"),
      offsets={"ov4096": (-40, -14), "ov0": (12, -14),
               "gate207 gather\n25KB/s": (12, 16),
               "K2048 gate\n22KB/s": (12, -20)})
_diamonds(ax, xoff=12)
ax.set_xlim(2, 42)
fig.tight_layout()
fig.savefig("pareto_off75.png", dpi=120)
print("saved pareto_off75.png")
print(f"points: {len(OFF50) + len(OFF75) + len(MB1)}")
