"""Lecture 18 — figures 02 and 06: the 'empty command' and 'wrong command' numbers in the literature (transcribed by hand; no model/GPU needed).

  python plot_day18.py <output_folder/>

02-empty-literature.png  Correct English command vs empty command (Ø) on LIBERO-Spatial, per model.
   Sources: Chen et al. 2026 (arXiv:2606.15714) Table 6 — Spatial columns en / Ø (the OpenVLA-OFT row in the paper is the same as
   ABot-M0; it contradicts the paper's own Table 1 → marked 'suspicious'); Zhan et al. 2026 (arXiv:2601.04052) Table 8/9 — π0 and π0.5
   correct / blank; Fang et al. 2026 (arXiv:2602.17659) Table I — π0.5 vision-only (language input removed) Spatial 68.2.
06-wrong-literature.png  LIBERO-CF CF-Spatial (Fang et al. Table III, without CAG): which task does the model perform under a wrong (counterfactual) command?
   Grounding/Success × Faithful (faithful to the command) / Biased (performs the task from training).
"""
import os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out = (sys.argv[1] if len(sys.argv) > 1 else ".").rstrip("/") + "/"; os.makedirs(out, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
Y, B, G, R = "#ffd166", "#cfe8ff", "#d4f5d4", "#ffd6d6"
YD, BD, GD, RD = "#c79a00", "#3b6ea8", "#2e8b57", "#c0392b"

# ---------- 02 ----------
# (label, en, empty, source abbreviation, note)
ROWS = [
    ("OpenVLA-OFT", 93.5, 80.0, "Chen T6", "row suspicious"),
    ("π0.5", 97.5, 66.0, "Chen T6", ""),
    ("π0.5", 95.4, 65.0, "Zhan T8/9", ""),
    ("π0.5 (vision-only)", 97.2, 68.2, "Fang T1", "no language input"),
    ("π0", 96.8, 29.8, "Zhan T8/9", ""),
    ("Cosmos Policy", 96.5, 55.5, "Chen T6", ""),
    ("Qwen3-VL-OFT", 92.5, 34.0, "Chen T6", ""),
    ("Qwen3-VL-FAST", 89.0, 12.5, "Chen T6", ""),
    ("Qwen3-VL-GR00T", 94.0, 64.5, "Chen T6", ""),
    ("Qwen3-VL-π", 93.0, 41.0, "Chen T6", ""),
    ("Qwen2.5-VL-OFT", 91.5, 66.5, "Chen T6", ""),
    ("Qwen2.5-VL-FAST", 86.5, 2.0, "Chen T6", ""),
    ("Qwen2.5-VL-GR00T", 95.5, 62.5, "Chen T6", ""),
    ("Qwen2.5-VL-π", 88.0, 52.5, "Chen T6", ""),
]
fig, ax = plt.subplots(figsize=(15, 6.4))
y = np.arange(len(ROWS))[::-1]
for yi, (lab, en, em, src, note) in zip(y, ROWS):
    ax.plot([em, en], [yi, yi], color="#bbb", lw=3, solid_capstyle="round", zorder=1)
    ax.scatter(en, yi, s=80, color=GD, zorder=3, edgecolor="white", lw=1)
    ax.scatter(em, yi, s=80, color=RD, zorder=3, edgecolor="white", lw=1)
    ax.text(en + 1.2, yi, f"{en:.1f}", va="center", fontsize=9, color=GD)
    ax.text(em - 1.2, yi, f"{em:.1f}", va="center", ha="right", fontsize=9, color=RD)
    ax.text(103, yi, f"Δ −{en - em:.0f}  ·  {src}" + (f"  ({note})" if note else ""), va="center", fontsize=8.5, color="#555")
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in ROWS])
ax.set_xlim(-2, 135); ax.set_xlabel("LIBERO-Spatial success rate (%)")
ax.axvspan(29.8, 68.2, color=R, alpha=0.25, lw=0)
ax.set_ylim(-1.6, len(ROWS) - 0.4)
ax.text(49, -1.1, "share of Spatial 'solvable from the image' with an empty command: 30–68% for most models", ha="center", fontsize=9, color=RD)
ax.scatter([], [], s=80, color=GD, label="correct English command"); ax.scatter([], [], s=80, color=RD, label="empty command (Ø) / no language input")
ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.grid(axis="x", alpha=0.25)
ax.set_title("Lecture 18 — the 'empty command' effect in the literature, LIBERO-Spatial (real paper values; Chen et al. 2606.15714 T6, Zhan et al. 2601.04052 T8–9, Fang et al. 2602.17659 T1)",
             fontsize=10.5)
fig.tight_layout(); fig.savefig(out + "02-empty-literature.png", dpi=160); plt.close(fig)

# ---------- 06 ----------
CF = {  # CF-Spatial, without CAG: (grounding faithful, grounding biased, success faithful, success biased)
    "OpenVLA-OFT": (6.8, 84.7, 1.1, 80.7),
    "π0": (70.9, 37.1, 34.4, 19.1),
    "π0.5": (39.3, 61.3, 24.4, 56.9),
}
fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
for ax, (title, i_f, i_b) in zip(axes, [("Grounding: which object did the gripper touch?", 0, 1), ("Success: which task was completed?", 2, 3)]):
    x = np.arange(len(CF)); w = 0.36
    f = [v[i_f] for v in CF.values()]; b = [v[i_b] for v in CF.values()]
    ax.bar(x - w / 2, f, w, color=B, edgecolor=BD, label="Faithful — target of the wrong command (read the command)")
    ax.bar(x + w / 2, b, w, color=R, edgecolor=RD, label="Biased — the original task from training (went by the image)")
    for xi, (fv, bv) in enumerate(zip(f, b)):
        ax.text(xi - w / 2, fv + 1.5, f"{fv:.1f}", ha="center", fontsize=9, color=BD)
        ax.text(xi + w / 2, bv + 1.5, f"{bv:.1f}", ha="center", fontsize=9, color=RD)
    ax.set_xticks(x); ax.set_xticklabels(list(CF)); ax.set_ylim(0, 100); ax.set_title(title, fontsize=11); ax.grid(axis="y", alpha=0.25)
axes[0].set_ylabel("%"); axes[0].legend(frameon=False, fontsize=9, loc="upper right")
fig.suptitle("Lecture 18 — What happens when a 'wrong command' is given? LIBERO-CF, CF-Spatial (Fang et al. 2026, arXiv:2602.17659, Table III, without CAG)\n"
             "In our 'wrong' condition, success = the Biased column; you can only count Faithful from the video (which bowl?)", fontsize=10.5)
fig.tight_layout(); fig.savefig(out + "06-wrong-literature.png", dpi=160); plt.close(fig)
print("02-empty-literature.png, 06-wrong-literature.png →", out)

# ---------- 07: possible patterns (ILLUSTRATIVE — not real data; the reading key for Lecture 19) ----------
PAT = [
    ("A) Language is ignored\n(pure shortcut)", [90, 85, 85, 85], "clean ≈ empty ≈ wrong ≈ tr\n→ Spatial task 0 is solved from the scene"),
    ("B) Language is read, Turkish is not", [90, 40, 40, 40], "tr ≈ empty: nothing is extracted from Turkish\n(Chen et al.: the OFT/π0.5 pattern)"),
    ("C) Wrong command is 'biased'", [90, 40, 80, 40], "wrong ≈ clean but empty is low:\nthe command says 'pick a bowl', the scene picks which one"),
    ("D) The multilingual backbone helps", [90, 40, 40, 75], "tr > empty: Qwen3-VL extracts at least the\nobject/action from Turkish (hypothesis)"),
]
fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), sharey=True)
cols = [GD, RD, RD, YD]
for ax, (title, vals, note) in zip(axes, PAT):
    ax.bar(["clean", "empty", "wrong", "tr"], vals, color=[G, R, R, Y], edgecolor=cols, lw=1.2)
    ax.set_ylim(0, 128); ax.set_yticks([0, 20, 40, 60, 80, 100]); ax.set_title(title, fontsize=10.5); ax.grid(axis="y", alpha=0.25)
    ax.text(1.5, 124, note, ha="center", va="top", fontsize=8.5, color="#444")
axes[0].set_ylabel("success rate (%) — ILLUSTRATIVE")
fig.suptitle("Lecture 18 — four possible patterns of the 4-row table (ILLUSTRATIVE values; the real numbers will come from your run, interpretation in Lecture 19)", fontsize=11)
fig.tight_layout(); fig.savefig(out + "07-expected-patterns.png", dpi=160); plt.close(fig)
print("07-expected-patterns.png →", out)
