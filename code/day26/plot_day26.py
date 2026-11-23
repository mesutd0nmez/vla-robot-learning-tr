"""Lecture 26 — figures 03 and 05.

  python plot_day26.py <output_folder/>   (default: ../../lectures/images/lecture-26/)

03-rq-coverage.png : Experiment space (block × condition) — which RQ needs which cell; block counts REAL from datasets/04/data/pairs.csv
                     (A 6 / B 6 / C 6 / D 2), no-command upper bound from Lecture 19 (Spatial 9/10, Goal 1/10).
05-literature-map.png : Where the Turkish × VLA cell sits in the literature — each study is placed by the properties it reports
                        (language / command manipulation / loop / empty-command baseline). Sources in the panel; a position, not data.
"""
import csv, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

out = (sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "../../lectures/images/lecture-26/")).rstrip("/") + "/"
os.makedirs(out, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
Y, B, G, R = "#ffd166", "#cfe8ff", "#d4f5d4", "#ffd6d6"
YD, BD, GD, RD = "#c79a00", "#3b6ea8", "#2e8b57", "#c0392b"

# ---------- block counts (real) ----------
pairs = os.path.join(os.path.dirname(__file__), "../../datasets/04-tr-spatial-commands/data/pairs.csv")
blocks = {}
with open(pairs, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        b = row["block"]; blocks.setdefault(b, {"n": 0, "suite": row["suite"], "tier2": 0})
        blocks[b]["n"] += 1
        if row["tier"] == "2": blocks[b]["tier2"] += 1
print("blocks:", blocks)

# ---------- 03: experiment space ----------
conds = ["clean (EN)", "empty", "wrong", "tr_a", "tr_b", "tr_ascii", "tr_bare", "EN-B (new recipe)", "tr + MT", "tr + parser", "tr + LoRA"]
cols = ["A\nSpatial trained", "B\nSpatial new recipe", "C\nGoal trained", "D\nGoal untrained"]
# cell -> (RQ label, color). "" = absent in these RQs.
M = {
    "clean (EN)":        ["a b c", "b", "a b c", "—"],
    "empty":             ["a b c", "b c", "a b c", "—"],
    "wrong":             ["a", "", "a", "—"],
    "tr_a":              ["a b c", "b c", "a b c", "—"],
    "tr_b":              ["b", "b", "b", "—"],
    "tr_ascii":          ["b", "b", "b", "—"],
    "tr_bare":           ["b", "", "b", "—"],
    "EN-B (new recipe)": ["", "b", "", "—"],
    "tr + MT":           ["c", "(=EN-B)", "c", "—"],
    "tr + parser":       ["c", "(=EN-B)", "c", "—"],
    "tr + LoRA":         ["c", "c", "c", "—"],
}
fig, ax = plt.subplots(figsize=(12.5, 7.2))
nr, nc = len(conds), len(cols)
for i, c in enumerate(conds):
    for j in range(nc):
        v = M[c][j]
        if v == "—":
            col, txt = "#eeeeee", "no predicate"
        elif v == "":
            col, txt = "white", ""
        elif v.startswith("(="):
            col, txt = "#f4f4f4", v
        else:
            col = Y if "a" in v.split() else (B if "b" in v.split() else G)
            txt = " · ".join("RQ-" + k for k in v.split())
        ax.add_patch(Rectangle((j, nr - 1 - i), 1, 1, facecolor=col, edgecolor="#999", lw=0.8))
        ax.text(j + 0.5, nr - 1 - i + 0.5, txt, ha="center", va="center", fontsize=8.6, color="#222")
ax.set_xlim(0, nc); ax.set_ylim(0, nr)
ax.set_xticks([j + 0.5 for j in range(nc)])
ax.set_xticklabels([f"{cols[j]}\n({blocks[k]['n']} pairs; ceiling {t})" for j, (k, t) in enumerate([("A", "9/10"), ("B", "9/10"), ("C", "1/10"), ("D", "1/10")])], fontsize=9)
ax.set_yticks([nr - 1 - i + 0.5 for i in range(nr)]); ax.set_yticklabels(conds, fontsize=9)
ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("Lecture 26 — Experiment space: block (datasets/04, real counts) × command condition -> which RQ needs which cell\n"
             "yellow = the core of RQ-a (Lecture 18 conditions), blue = added by RQ-b, green = added by RQ-c; "
             "ceiling = Lecture 19 no-command upper bound", fontsize=10.5)
ax.text(0, -1.25, "In block B, tier 2 = %d pairs need a counterfactual BDDL (bddl_cf/, target bowl_2). MT/parser reduce to English in B -> its result is the EN-B row; "
        "strong reading is measured only in the LoRA row." % blocks["B"]["tier2"], fontsize=8.5, color="#444")
fig.tight_layout(); fig.savefig(out + "03-rq-coverage.png", dpi=160); plt.close(fig)

# ---------- 05: literature map ----------
# x: type of command manipulation, y: language. Marker: closed loop (circle) / open loop (square). Edge: is there an empty-command baseline.
xs = {"translation (training sentence)": 0, "paraphrase / rewrite": 1, "counterfactual in the same scene": 2, "empty / wrong command": 3}
ys = {"English only": 0, "multilingual (no TR)": 1, "Turkish": 2}
studies = [
    # name, x, y, closed loop?, empty baseline?, note
    ("LIBERO-Plus Language\n(2510.13626)", 1, 0, True, False, ""),
    ("LIBERO-PRO\n(2510.03827)", 1, 0, True, False, ""),
    ("LIBERO-CF\n(2602.17659)", 2, 0, True, True, "vision-only"),
    ("LangGap\n(2603.00592)", 2, 0, True, False, "1/k argument"),
    ("Zhan et al.\n(2601.04052)", 3, 0, True, True, "blank"),
    ("Dong et al.\n(2606.11906)", 0, 1, True, False, "9 languages"),
    ("Chen et al.\n(2606.15714)", 0, 1, True, True, "4 languages + Ø"),
    ("Tatar et al. 2026\n(SIU)", 0, 2, False, False, "1 frame, L2"),
    ("Lecture 15 mini\n(open loop)", 2, 2, False, True, "L2/cos"),
    ("RQ-a / RQ-b / RQ-c\n(this thesis)", 2, 2, True, True, "empty + wrong + CF"),
]
fig, ax = plt.subplots(figsize=(12.5, 6.2))
jit = {}
for name, x, y, closed, empty, note in studies:
    k = (x, y); jit[k] = jit.get(k, 0) + 1
    dx = (jit[k] - 1) * 0.0; dy = (jit[k] - 1) * 0.32
    col = Y if "this thesis" in name else (B if y < 2 else R)
    if "this thesis" in name: col = G
    ax.scatter(x + dx, y + dy, s=520, marker="o" if closed else "s", facecolor=col,
               edgecolor=(GD if empty else "#999"), linewidth=(2.6 if empty else 1.0), zorder=3)
    ax.text(x + dx + 0.13, y + dy, name + (f"  — {note}" if note else ""), va="center", fontsize=8.6)
ax.set_xticks(list(xs.values())); ax.set_xticklabels(list(xs.keys()), fontsize=9.5)
ax.set_yticks(list(ys.values())); ax.set_yticklabels(list(ys.keys()), fontsize=9.5)
ax.set_xlim(-0.4, 4.2); ax.set_ylim(-0.5, 2.9)
ax.grid(alpha=0.2)
ax.set_title("Lecture 26 — The Turkish × closed-loop × counterfactual cell is empty in the literature\n"
             "circle = closed loop (success), square = open loop (action distance); thick green edge = empty-command baseline reported", fontsize=10.5)
ax.text(-0.35, 2.75, "Positions come from the properties the studies report; a classification, not a number (sources in Lectures 23–25 and in the lecture).", fontsize=8.5, color="#444")
fig.tight_layout(); fig.savefig(out + "05-literature-map.png", dpi=160); plt.close(fig)
print("written:", out + "03-rq-coverage.png", out + "05-literature-map.png")
