"""
Plot Zero-Shot vs 4-Shot vs RAG AUC comparison across all four conditions,
with 95% bootstrap CI error bars.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent / "outputs"

def load(task, filename):
    path = ROOT / task / filename
    d = json.loads(path.read_text())
    auc = d["auc"]
    lo, hi = d["auc_ci_95"]
    return auc, lo, hi

# ── data ─────────────────────────────────────────────────────────────────────
# Each entry: (auc, ci_lo, ci_hi)
data = {
    # Zero-shot (no RAG, no prompting)
    "zs_base_dir":   load("direction", "results_base.json"),
    "zs_ft_dir":     load("direction", "results.json"),
    "zs_base_sur":   load("surprise",  "results_base.json"),
    "zs_ft_sur":     load("surprise",  "results.json"),

    # 4-shot prompting (no RAG) — v4 seed42 + ctx (from comparison table)
    "fs_base_dir":   (0.520, 0.429, 0.605),
    "fs_ft_dir":     (0.536, 0.458, 0.612),
    "fs_base_sur":   (0.720, 0.569, 0.858),
    "fs_ft_sur":     (0.748, 0.622, 0.869),

    # RAG only (zero-shot + RAG)
    "rag_base_dir":  load("direction", "results_rag_base.json"),
    "rag_ft_dir":    load("direction", "results_rag.json"),
    "rag_base_sur":  load("surprise",  "results_rag_base.json"),
    "rag_ft_sur":    load("surprise",  "results_rag.json"),

    # RAG + 4-shot prompting — v4 seed42
    "ragfs_base_dir": load("direction", "results_prompt_rag_base_v4_seed42.json"),
    "ragfs_ft_dir":   load("direction", "results_prompt_rag_ft_v4_seed42.json"),
    "ragfs_base_sur": load("surprise",  "results_prompt_rag_base_v4_seed42.json"),
    "ragfs_ft_sur":   load("surprise",  "results_prompt_rag_ft_v4_seed42.json"),
}

# ── layout ────────────────────────────────────────────────────────────────────
groups   = ["Base\nDirection", "FT\nDirection", "Base\nSurprise", "FT\nSurprise"]
n_groups = len(groups)
n_bars   = 4  # conditions per group
width    = 0.18
x = np.arange(n_groups)
offsets  = np.array([-1.5, -0.5, 0.5, 1.5]) * width

colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
labels = ["Zero-shot", "4-shot", "RAG", "RAG + 4-shot"]

# rows: [zero-shot, 4-shot, RAG, RAG+4-shot]
# cols: [Base Dir, FT Dir, Base Sur, FT Sur]
aucs = np.array([
    [data["zs_base_dir"][0],    data["zs_ft_dir"][0],    data["zs_base_sur"][0],    data["zs_ft_sur"][0]],
    [data["fs_base_dir"][0],    data["fs_ft_dir"][0],    data["fs_base_sur"][0],    data["fs_ft_sur"][0]],
    [data["rag_base_dir"][0],   data["rag_ft_dir"][0],   data["rag_base_sur"][0],   data["rag_ft_sur"][0]],
    [data["ragfs_base_dir"][0], data["ragfs_ft_dir"][0], data["ragfs_base_sur"][0], data["ragfs_ft_sur"][0]],
])

ci_lo = np.array([
    [data["zs_base_dir"][1],    data["zs_ft_dir"][1],    data["zs_base_sur"][1],    data["zs_ft_sur"][1]],
    [data["fs_base_dir"][1],    data["fs_ft_dir"][1],    data["fs_base_sur"][1],    data["fs_ft_sur"][1]],
    [data["rag_base_dir"][1],   data["rag_ft_dir"][1],   data["rag_base_sur"][1],   data["rag_ft_sur"][1]],
    [data["ragfs_base_dir"][1], data["ragfs_ft_dir"][1], data["ragfs_base_sur"][1], data["ragfs_ft_sur"][1]],
])

ci_hi = np.array([
    [data["zs_base_dir"][2],    data["zs_ft_dir"][2],    data["zs_base_sur"][2],    data["zs_ft_sur"][2]],
    [data["fs_base_dir"][2],    data["fs_ft_dir"][2],    data["fs_base_sur"][2],    data["fs_ft_sur"][2]],
    [data["rag_base_dir"][2],   data["rag_ft_dir"][2],   data["rag_base_sur"][2],   data["rag_ft_sur"][2]],
    [data["ragfs_base_dir"][2], data["ragfs_ft_dir"][2], data["ragfs_base_sur"][2], data["ragfs_ft_sur"][2]],
])

err_lo = aucs - ci_lo
err_hi = ci_hi - aucs

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 7))

for i, (color, label, offset) in enumerate(zip(colors, labels, offsets)):
    xs   = x + offset
    yerr = np.array([err_lo[i], err_hi[i]])
    bars = ax.bar(xs, aucs[i], width=width, color=color, label=label,
                  zorder=3, alpha=0.88)
    ax.errorbar(xs, aucs[i], yerr=yerr, fmt="none", color="black",
                capsize=4, capthick=1.2, linewidth=1.2, zorder=4)
    for bar, val in zip(bars, aucs[i]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=12)
ax.set_ylabel("AUC", fontsize=13)
ax.set_ylim(0, 1.05)
ax.set_title("Zero-Shot vs 4-Shot vs RAG AUC (95% CI)", fontsize=14, fontweight="bold")
ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=11, loc="upper left", framealpha=0.85)

plt.tight_layout()
out = Path(__file__).parent.parent / "outputs" / "rag_comparison_chart.png"
plt.savefig(out, dpi=150)
print(f"Saved to {out}")
