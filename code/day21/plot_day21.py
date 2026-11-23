"""Lecture 21 — figure 05: thesis environment decision matrix (LIBERO+MuJoCo vs Webots).

The scores are SUBJECTIVE (Claude's assessment in this lecture, 0–3); not a source, a starting point for discussion.
After talking to your advisor, change the scores and redraw:  python plot_day21.py --out 05-decision.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CRITERIA = [
    ("Ready-made benchmark + leaderboard (Spatial/Goal, Plus, CF, X)", 3, 0),
    ("VLA eval code available (OpenVLA, VLANeXt, OFT…)",             3, 0),
    ("Training data (50 demos/task, HDF5, RLDS)",                   3, 0),
    ("Ease of installation / running on a Mac M2",                  1, 3),
    ("Room scale, mobile base (RoomSpace-style scene)",              0, 3),
    ("Scene/PROTO authoring, camera placement flexibility",         1, 3),
    ("ROS 2 bridge (transfer to a real robot)",                     0, 3),
    ("Time to the Turkish-command experiment: can it start 'today'", 3, 1),
    ("Comparability of results with the literature",                3, 0),
    ("Same robot (Panda) exists in both environments",              3, 2),
]
LABELS = ["LIBERO + robosuite + MuJoCo", "Webots R2025a"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="05-decision.png")
    a = ap.parse_args()
    M = np.array([[c[1], c[2]] for c in CRITERIA], float)
    fig, ax = plt.subplots(figsize=(11, 6.8))
    cmap = matplotlib.colors.ListedColormap(["#ffd6d6", "#fde9c4", "#d4f5d4", "#9be29b"])
    ax.imshow(M, cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto")
    for i, (name, s1, s2) in enumerate(CRITERIA):
        for j, s in enumerate((s1, s2)):
            ax.text(j, i, ["none", "weak", "medium", "strong"][s], ha="center", va="center", fontsize=11, weight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(LABELS, fontsize=11)
    ax.set_yticks(range(len(CRITERIA))); ax.set_yticklabels([c[0] for c in CRITERIA], fontsize=9.5)
    ax.xaxis.tick_top()
    tot = M.sum(axis=0)
    ax.set_xlabel(f"total (equal weights): {tot[0]:.0f} / {tot[1]:.0f}   —   the thesis question needs a benchmark, Webots is not a benchmark",
                  fontsize=10, labelpad=10)
    ax.set_title("Lecture 21 — Thesis environment decision matrix (SUBJECTIVE scoring 0–3; to be updated with the advisor)", fontsize=12, pad=38)
    for i in range(len(CRITERIA) + 1):
        ax.axhline(i - 0.5, color="white", lw=2)
    ax.axvline(0.5, color="white", lw=2)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("saved:", a.out)


if __name__ == "__main__":
    main()
