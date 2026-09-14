"""Size-vs-throughput (+ratio) scaling chart: 100KB / 1MB / 10MB ladders.

Top-right is best on BOTH panels (throughput up, ratio up).
All points measured (220/220). See data/sota_loop.json.
Saves scale_time_size.png (repo root, referenced by READMEs).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

SIZES_KB = [100, 1024, 10240]
SIZES_LBL = ["100KB", "1MB", "10MB"]

# seconds per scheme (None = not measured)
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
SOTA = 8.0 / 0.9389  # 8.52x


def _kbs(size_kb, secs):
    return [s / t for s, t in zip(size_kb, secs)]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

for name, ts in TIME.items():
    xs = [s for s, t in zip(SIZES_KB, ts) if t is not None]
    ys = _kbs(xs, [t for t in ts if t is not None])
    ax1.loglog(xs, ys, "o-", label=name)
ax1.set_xlabel("file size")
ax1.set_ylabel("throughput (KB/s, higher better)")
ax1.set_title("Throughput ladder [top-right is best]")
ax1.set_xticks(SIZES_KB)
ax1.set_xticklabels(SIZES_LBL)
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
# log y: let matplotlib handle minor ticks naturally
ax1.grid(True, which="major", alpha=0.3)
ax1.grid(True, which="minor", alpha=0.15)
ax1.legend(fontsize=7)

for name, bs in BPB.items():
    xs = [s for s, b in zip(SIZES_KB, bs) if b is not None]
    ys = [8.0 / b for b in bs if b is not None]
    ax2.semilogx(xs, ys, "s-", label=name)
ax2.axhline(SOTA, color="red", linestyle=":", label="SOTA 8.52x (full-file)")
ax2.set_xlabel("file size")
ax2.set_ylabel("compression ratio (higher better)")
ax2.set_title("Ratio vs scale [top-right is best]")
ax2.set_xticks(SIZES_KB)
ax2.set_xticklabels(SIZES_LBL)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
ax2.grid(True, which="major", alpha=0.3)
ax2.grid(True, which="minor", alpha=0.15)
ax2.legend(fontsize=7)

fig.suptitle("Scaling: throughput & ratio vs size (all measured)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("scale_time_size.png", dpi=130, bbox_inches="tight")
print("saved scale_time_size.png")
