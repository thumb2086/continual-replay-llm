"""Pareto frontier: speed vs compression ratio (all measured).

Fixed: no clipped points, proper log ticks, no label overlap.
X = KB/s (log, faster right), Y = compression ratio (higher better).
Top-right is best.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from adjustText import adjust_text

# ── Data ──────────────────────────────────────────────────────────────
OFF50 = [
    ("gate 22.7", 100/4.4, 0.9276),
    ("gather 18.9", 100/5.3, 0.9272),
    ("chunk 34.5", 100/2.9, 0.9276),
    ("chunkK2 27", 100/3.7, 0.9187),
    ("chunkK4 16.9", 100/5.9, 0.9142),
    ("ov0K2", 100/5.7, 0.9183),
    ("knee", 100/5.9, 0.9194),
    ("ov1024", 100/5.5, 0.9241),
    ("K3072", 100/9.7, 0.9029),
    ("ov3072K8", 100/15.0, 0.9026),
    ("K2k", 100/18.0, 0.9050),
    ("K4k", 100/20.2, 0.9013),
    ("ov0@K8", 100/21.8, 0.9127),
    ("ov1024@K8", 100/22.3, 0.9099),
    ("ov2048@K8", 100/23.6, 0.9058),
    ("K8k gather", 100/27.4, 0.9005),
    ("crown gate52", 100/15.2, 0.9005),
    ("crown gate207", 100/13.3, 0.9006),
    ("K8k crown", 100/28.1, 0.9003),
    ("K6k", 100/28.5, 0.9004),
    ("K16k", 100/52.4, 0.9027),
    ("ov6144@K8", 100/55.8, 0.9004),
    ("ov4096 K1k", 100/17.3, 0.9139),
]
# SmolLM2-only 1MB diamonds (no Qwen here)
MB1_SMOLM2 = [
    ("1MB off50", 1024/256.0, 0.9222),
    ("1MB off25", 1024/262.7, 0.9076),
    ("1MB gate207", 1024/49.4, 0.9359),
    ("1MB crown-g", 1024/201.6, 0.9079),
    ("1MB chunked", 1024/29.8, 0.9360),
]
# Qwen 1MB diamonds (separate, only for all-models chart)
MB1_QWEN = [
    ("1MB qwen1.5b", 1024/59.5, 0.7319),
    ("1MB qwen3b", 1024/79.9, 0.7103),
]
OFF75 = [
    ("ov4096", 100/18.6, 0.9662),
    ("ov3072", 100/15.6, 0.9653),
    ("ov2048", 100/14.2, 0.9687),
    ("ov1024", 100/12.4, 0.9723),
    ("ov0", 100/11.3, 0.9785),
    ("gate207 25KB/s", 100/4.0, 0.9812),
    ("K2048 gate 22KB/s", 100/4.5, 0.9715),
]
OFF75_MB = [
    ("1MB off50", 1024/256.0, 0.9222),
    ("1MB off25", 1024/262.7, 0.9076),
    ("1MB gate207", 1024/49.4, 0.9359),
    ("1MB crown-g", 1024/201.6, 0.9079),
    ("1MB chunked", 1024/29.8, 0.9360),
]
QWEN15B = [
    ("qwen1.5b K1024/B8", 100/5.51, 0.7402),
    ("qwen1.5b K2048/B8", 100/5.95, 0.7319),
    ("qwen1.5b K1024/B16", 100/8.04, 0.7324),
    ("qwen1.5b K4096/B8", 100/9.48, 0.7281),
    ("qwen1.5b K2048/B16", 100/19.79, 0.7208),
    ("qwen1.5b K2048/B28", 100/25.97, 0.7193),
    ("qwen1.5b K4096/16ov", 100/22.8, 0.7005),
    ("qwen1.5b K4096/28*", 100/35.7, 0.6996),
]
QWEN3B = [
    ("qwen3b K1/B2", 100/9.7, 0.6845),
    ("qwen3b K1/B4", 100/15.4, 0.6706),
    ("qwen3b K2/B4", 100/19.8, 0.6650),
    ("qwen3b K2/B8", 100/30.0, 0.6573),
    ("qwen3b K2/B28*", 100/263.4, 0.6450),
]
SOTA_RATIO = 8.0 / 0.9389
NNCP_KBS = 3.25


def _setup(ax, title, xlabel=True):
    """Setup common axes: log x with proper ticks, grid, reference lines."""
    ax.axhline(SOTA_RATIO, color="red", linestyle="--", lw=1, label="SOTA 8.52x")
    ax.axvline(NNCP_KBS, color="orange", linestyle="--", lw=1, label="NNCP 3.25KB/s")
    ax.set_xscale("log")
    if xlabel:
        ax.set_xlabel("KB/s (faster right, log)")
    ax.set_ylabel("compression ratio (higher better)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.15)
    # Force log ticks: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=20))
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs="auto", numticks=50))
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())


def _plot_frontier(ax, pts, color, label_prefix=""):
    """Plot a frontier line + scatter + labels with repel."""
    xs = [p[1] for p in pts]
    ys = [8/p[2] for p in pts]
    order = sorted(range(len(pts)), key=lambda i: pts[i][1])
    ax.plot([xs[i] for i in order], [ys[i] for i in order],
            color=color, lw=1.2, alpha=0.6, zorder=2)
    texts = []
    for lab, x, b in pts:
        y = 8/b
        ax.scatter([x], [y], s=60, color=color, edgecolors="black", lw=0.5, zorder=3)
        txt = ax.text(x, y, lab, fontsize=5.5, ha="center", va="bottom",
                      color=color, fontweight="bold")
        texts.append(txt)
    return texts


def _diamonds(ax, pts, color="purple"):
    """Plot diamond markers for 1MB points."""
    texts = []
    for lab, x, b in pts:
        y = 8/b
        ax.scatter([x], [y], s=100, marker="D", color=color, edgecolors="black", lw=0.5, zorder=4)
        txt = ax.text(x, y, lab, fontsize=5.5, ha="center", va="bottom",
                      color=color, fontweight="bold")
        texts.append(txt)
    return texts


# ═══════════════════════════════════════════════════════════════════════
# Plot 1: off50 (SmolLM2 only, no Qwen)
# ═══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 7))
_setup(ax, "Pareto: speed vs ratio — article region (SmolLM2-135M, all measured)")
texts = _plot_frontier(ax, OFF50, "steelblue")
texts += _diamonds(ax, MB1_SMOLM2)
ax.set_xlim(1.5, 70)
ax.set_ylim(8.3, 9.1)
adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, shrinkB=2),
            expand_points=(1.5, 1.8), force_text=0.5, force_points=0.4, lim=100)
fig.tight_layout()
fig.savefig("pareto_off50.png", dpi=130, bbox_inches="tight")
print("saved pareto_off50.png")

# ═══════════════════════════════════════════════════════════════════════
# Plot 2: off75 (SmolLM2 only, no Qwen)
# ═══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 7))
_setup(ax, "Pareto: speed vs ratio — hard region (SmolLM2-135M, all measured)")
texts = _plot_frontier(ax, OFF75, "seagreen")
texts += _diamonds(ax, OFF75_MB)
ax.set_xlim(2, 50)
ax.set_ylim(7.9, 9.0)
adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, shrinkB=2),
            expand_points=(1.5, 1.8), force_text=0.5, force_points=0.4, lim=100)
fig.tight_layout()
fig.savefig("pareto_off75.png", dpi=130, bbox_inches="tight")
print("saved pareto_off75.png")

# ═══════════════════════════════════════════════════════════════════════
# Plot 3: All models combined
# ═══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 8))
_setup(ax, "Pareto: all models (top-right is best)")
# SmolLM2
order = sorted(range(len(OFF50)), key=lambda i: OFF50[i][1])
sx = [OFF50[i][1] for i in order]
sy = [8/OFF50[i][2] for i in order]
ax.plot(sx, sy, "o-", color="steelblue", lw=1, alpha=0.5, label="SmolLM2-135M", zorder=1)
for lab, x, b in OFF50:
    ax.scatter([x], [8/b], s=40, color="steelblue", edgecolors="black", lw=0.3, zorder=2)
# Qwen-1.5B
order = sorted(range(len(QWEN15B)), key=lambda i: QWEN15B[i][1])
qx = [QWEN15B[i][1] for i in order]
qy = [8/QWEN15B[i][2] for i in order]
ax.plot(qx, qy, "s-", color="darkorange", lw=1.5, label="Qwen-1.5B", zorder=2)
texts = []
for lab, x, b in QWEN15B:
    ax.scatter([x], [8/b], s=60, marker="s", color="darkorange", edgecolors="black", lw=0.5, zorder=3)
    txt = ax.text(x, 8/b, lab, fontsize=5, ha="center", va="bottom", color="darkorange")
    texts.append(txt)
# Qwen-3B
order = sorted(range(len(QWEN3B)), key=lambda i: QWEN3B[i][1])
tx = [QWEN3B[i][1] for i in order]
ty = [8/QWEN3B[i][2] for i in order]
ax.plot(tx, ty, "^-", color="crimson", lw=1.5, label="Qwen-3B", zorder=2)
for lab, x, b in QWEN3B:
    ax.scatter([x], [8/b], s=60, marker="^", color="crimson", edgecolors="black", lw=0.5, zorder=3)
    txt = ax.text(x, 8/b, lab, fontsize=5, ha="center", va="bottom", color="crimson")
    texts.append(txt)
# 1MB diamonds
for lab, x, b in MB1_SMOLM2 + MB1_QWEN:
    y = 8/b
    ax.scatter([x], [y], s=80, marker="D", color="purple", edgecolors="black", lw=0.5, zorder=4)
    txt = ax.text(x, y, lab, fontsize=5, ha="center", va="bottom", color="purple")
    texts.append(txt)
adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.4),
            expand_points=(1.5, 2.0), force_text=0.4, force_points=0.3, lim=120)
ax.set_xlim(1, 70)
ax.set_ylim(8.5, 13)
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig("pareto_all_models.png", dpi=130, bbox_inches="tight")
print("saved pareto_all_models.png")

print(f"total points: {len(OFF50)+len(OFF75)+len(MB1_SMOLM2)+len(MB1_QWEN)+len(QWEN15B)+len(QWEN3B)}")
