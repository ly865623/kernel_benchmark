#!/usr/bin/env python3
"""Render the top-k scan deliverable figure (Morandi, paper-facing) from grid_topk_v2.json."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "/home/liuy/DeepScientist/quests/003/.ds/worktrees/idea-idea-98be86a0/experiments/main/grid_topk_v2/json/grid_topk_v2.json"
OUT = "/home/liuy/DeepScientist/quests/003/.ds/worktrees/paper-dsa-stagepred-grid-v2/experiments/analysis-results/topk_scan_v2"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.facecolor": "white", "figure.facecolor": "white", "axes.grid": True,
    "grid.color": "#D8D1C7", "grid.linewidth": 0.6, "grid.alpha": 0.7,
})
# mist-stone (paper-facing) primary + sage-clay contrast
C1024, C2048 = "#8A9199", "#B88C8C"

d = json.load(open(SRC))
m = {(r["topk"], r["s_q"], r["s_kv"]): r for r in d["rows"]}
bss = d["batch_sizes"]; SKV = 8192

x = bss
y1 = [m[(1024, b, SKV)]["latency_us"] for b in bss]
y2 = [m[(2048, b, SKV)]["latency_us"] for b in bss]

fig, ax = plt.subplots(figsize=(7.0, 4.2))
ax.plot(x, y1, "-o", color=C1024, lw=2, ms=6, label="top-k = 1024")
ax.plot(x, y2, "-s", color=C2048, lw=2, ms=6, label="top-k = 2048")
ax.set_xlabel("Batch size (= query rows $s_q$)")
ax.set_ylabel("Whole-kernel latency (µs)")
ax.set_title("FlashMLA DSA sparse-prefill latency vs batch, top-k scan\n(B200, $s_{kv}$=8192, h_q=128, d_qk=576; mean ratio 2048/1024 = 1.61×)",
             fontsize=10.5)
ax.legend(frameon=False, loc="upper left")
# annotate wave boundaries (74 tiles/wave)
for wv in (74, 148, 296):
    ax.axvline(wv, color="#D8C3BC", lw=0.8, ls="--", alpha=0.7)
fig.text(0.5, -0.02,
         "Publication-grade figure refinement is recommended with AutoFigure-Edit "
         "(open-source: https://github.com/ResearAI/AutoFigure-Edit; online service: https://deepscientist).",
         ha="center", fontsize=6.0, color="#8A9199")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "topk_scan_latency_skv8192.png"), bbox_inches="tight")
print("WROTE", os.path.join(OUT, "topk_scan_latency_skv8192.png"))
