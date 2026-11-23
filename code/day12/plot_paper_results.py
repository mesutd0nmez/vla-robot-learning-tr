"""Day 12 — Plots the LIBERO and LIBERO-plus (Language) numbers from the VLANeXt paper.
Usage: python plot_paper_results.py [output_folder/]   → 03-libero-results.png, 04-libero-plus-language.png
Numbers: tables of arXiv:2602.18532v2 (VLANeXt); transcribed by hand.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
import os, sys
out = (sys.argv[1] if len(sys.argv) > 1 else "./") ; os.makedirs(out, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# --- 03: LIBERO per-suite, VLANeXt paper Table (arXiv 2602.18532v2)
suites = ["Spatial", "Object", "Goal", "Long", "Avg."]
models = [("OpenVLA (2024)", "#8c6bb1", [84.7, 88.4, 79.2, 53.7, 76.5]),
          ("OpenVLA-OFT (2025)", "#3a9a5c", [97.6, 98.4, 97.9, 94.5, 97.1]),
          ("π0.5 (2025)", "#3d7fc4", [98.8, 98.2, 98.0, 92.4, 96.9]),
          ("VLANeXt-B (2026)", "#e09a00", [99.0, 99.2, 96.6, 94.8, 97.4])]
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(suites)); w = 0.2
for i, (name, c, v) in enumerate(models):
    b = ax.bar(x + (i - 1.5) * w, v, w * 0.92, color=c, label=name, zorder=3)
    for r, val in zip(b, v):
        ax.text(r.get_x() + r.get_width() / 2, val + 0.8, f"{val:.1f}", ha="center", va="bottom", fontsize=8, color="#333")
ax.set_xticks(x); ax.set_xticklabels(suites)
ax.set_ylim(40, 104); ax.set_ylabel("Success rate (%)")
ax.yaxis.grid(True, color="#e5e5e5", zorder=0)
ax.set_title("LIBERO: 4 suites, 500 episodes/suite — source: VLANeXt paper table (arXiv:2602.18532v2)", fontsize=10)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False, fontsize=9)
fig.text(0.5, 0.005, "Each suite is 10 tasks × 50 episodes = 500 episodes. Next to these numbers, the 3 episodes you will run today are only a 'does it work?' test.",
        ha="center", fontsize=8.5, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(out + "03-libero-results.png", dpi=180); plt.close(fig)

# --- 04: LIBERO-plus Language perturbation per suite, OFT vs VLANeXt (same paper, LIBERO-plus table)
suites = ["Spatial", "Object", "Goal", "Long"]
oft   = [80.5, 99.0, 53.2, 87.0]
vla   = [86.9, 98.6, 51.5, 90.3]
clean_oft = [97.6, 98.4, 97.9, 94.5]; clean_vla = [99.0, 99.2, 96.6, 94.8]
fig, ax = plt.subplots(figsize=(10, 5.2))
x = np.arange(4); w = 0.36
b1 = ax.bar(x - w/2, oft, w * 0.94, color="#3a9a5c", label="OpenVLA-OFT, Language perturbation", zorder=3)
b2 = ax.bar(x + w/2, vla, w * 0.94, color="#e09a00", label="VLANeXt-B, Language perturbation", zorder=3)
ax.scatter(x - w/2, clean_oft, marker="_", s=700, color="#222", zorder=4, linewidths=2, label="same model, clean LIBERO (line)")
ax.scatter(x + w/2, clean_vla, marker="_", s=700, color="#222", zorder=4, linewidths=2)
for bars, vals in ((b1, oft), (b2, vla)):
    for r, v in zip(bars, vals):
        ax.text(r.get_x() + r.get_width()/2, v - 6, f"{v:.1f}", ha="center", fontsize=9, color="white", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(suites); ax.set_ylim(0, 105); ax.set_ylabel("Success rate (%)")
ax.yaxis.grid(True, color="#e5e5e5", zorder=0)
ax.set_title("LIBERO-plus 'Language' perturbation: how much does success drop when the command is changed? — source: arXiv:2602.18532v2", fontsize=10)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, fontsize=9)
fig.text(0.5, 0.005, "On Goal both models drop to ~50% (language is really used); on Object they stay at ~99% (the command is almost unnecessary, the shortcut from Lectures 3–4).",
         ha="center", fontsize=8.5, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(out + "04-libero-plus-language.png", dpi=180); plt.close(fig)
print("ok")
