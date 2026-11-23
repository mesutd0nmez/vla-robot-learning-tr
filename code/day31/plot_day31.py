"""Lecture 31 — figure 04: competence radar.
Three lines: (1) lecture coverage = number of lectures covering the competence / the maximum (REAL, the mapping in self_assessment.csv);
(2) is there evidence produced by you (a project folder or a block you filled in; 'filled:' = no '___' left) — REAL scan;
(3) your self-assessment 0–4 (the mesut_0_4 column; not drawn if empty).
  python plot_day31.py [output folder]
"""
import csv, glob, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "lectures", "images", "lecture-31")
rows = list(csv.DictReader(open(os.path.join(HERE, "self_assessment.csv"), encoding="utf-8")))
lab = [r["yetkinlik"] for r in rows]
cov = np.array([len(r["dersler"].split(";")) for r in rows], float); cov = cov / cov.max()
def kanit(spec):  # kanit = evidence
    # "filled:path" (or the Turkish "dolu:path") = the file exists AND has no unfilled '___' left (a block Mesut filled in); otherwise a glob
    if spec.startswith(("filled:", "dolu:")):
        p = os.path.join(ROOT, spec.split(":", 1)[1])
        return float(os.path.exists(p) and "___" not in open(p, encoding="utf-8").read())
    return float(bool(glob.glob(os.path.join(ROOT, spec))))
ev = np.array([kanit(r["kanit"]) for r in rows])
me = [r["mesut_0_4"].strip() for r in rows]
ang = np.linspace(0, 2 * np.pi, len(rows), endpoint=False); close = lambda v: np.r_[v, v[:1]]; A = close(ang)
plt.rcParams.update({"font.size": 9.5})
fig = plt.figure(figsize=(10, 8.5)); ax = fig.add_subplot(111, polar=True)
ax.plot(A, close(cov), color="#c79a00", lw=2); ax.fill(A, close(cov), color="#ffd166", alpha=0.45, label="lecture coverage (number of covering lectures, normalized) — real")
ax.plot(A, close(ev), color="#2e8b57", lw=2, marker="o"); ax.fill(A, close(ev), color="#d4f5d4", alpha=0.5, label="evidence from your hands: project folder / filled-in block present (1) / absent (0) — real scan")
if all(m != "" for m in me):
    v = np.array([float(m) / 4 for m in me]); ax.plot(A, close(v), color="#3b6ea8", lw=2, ls="--", label="self-assessment (0–4) — Mesut")
else:
    ax.text(0.5, -0.12, "Self-assessment line is empty: fill in the 'mesut_0_4' column of self_assessment.csv and run again",
            transform=ax.transAxes, ha="center", fontsize=9, color="#c0392b")
ax.set_xticks(ang); ax.set_xticklabels([l.replace(" (", "\n(") for l in lab], fontsize=8.5); ax.tick_params(axis="x", pad=18)
ax.set_yticks([0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["", "", "", ""]); ax.set_ylim(0, 1.05)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), fontsize=8.5)
fig.suptitle("Lecture 31 — Competence map: the gap between what was taught (yellow) and what you have proven by hand (green) is the work Stage 0 hands over", fontsize=10.5, y=0.99)
fig.tight_layout(); os.makedirs(out, exist_ok=True); fig.savefig(os.path.join(out, "04-competence-radar.png"), dpi=160)
print("evidence:", dict(zip(lab, ev.astype(int)))); print("written:", os.path.join(out, "04-competence-radar.png"))
