"""Balance curve: score = (8/bpb) * log(KB/s), SOTA-constrained.
Highlights the balanced point that beats SOTA on both axes.
"""
import math, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# All 100KB OFF50 points (current 35) + new carpet points
PTS = [
    ("gate 22.7", 100/4.4, 0.9276),
    ("gather 18.9", 100/5.3, 0.9272),
    ("chunk 34.5*", 100/2.9, 0.9276),
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
    # Qwen separate line (different tokenizer, for reference)
    ("Qwen K2k", 100/5.6, 0.8442),
    ("Qwen K4k*", 100/12.2, 0.8391),
]

def score(kbs,bpb): return (8/bpb)*math.log(kbs) if kbs>0 else 0

SOTA_BPB=0.9389
# filter SOTA-constrained (bpb<=SOTA)
sota_pts=[p for p in PTS if p[2]<=SOTA_BPB]
best=max(sota_pts, key=lambda p: score(p[1],p[2]))
print(f"Balanced (SOTA-constrained) = {best[0]} score={score(best[1],best[2]):.2f} {best[1]:.1f}KB/s {best[2]:.4f}")

fig, ax = plt.subplots(figsize=(10,5))
xs=[p[1] for p in PTS]
ys=[score(p[1],p[2]) for p in PTS]
labels=[p[0] for p in PTS]
ax.scatter(xs, ys, c=['red' if p==best else 'steelblue' for p in PTS], s=80, edgecolors='black', zorder=3)
for x,y,l in zip(xs,ys,labels):
    if l in (best[0], "K8k crown", "Qwen K4k*", "chunkK2 27", "chunkK4 16.9"):
        ax.annotate(l, (x,y), textcoords="offset points", xytext=(6,6), fontsize=8,
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.6))
ax.set_xscale("log")
ax.set_xlabel("KB/s (faster right, log)")
ax.set_ylabel("score = (8/bpb) * log(KB/s)  (higher better)")
ax.set_title("Balance curve (SOTA bpb≤0.9389) — top-right is best")
ax.grid(True, which="both", alpha=0.3)
# SOTA fail zone annotation
ax.axhline(score(1,SOTA_BPB), color="orange", linestyle="--", label="SOTA ratio floor")
ax.legend()
fig.tight_layout()
fig.savefig("balance_curve.png", dpi=130)
print("saved balance_curve.png")
