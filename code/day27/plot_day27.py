"""Lecture 27 — figure 06: the real measurement of sim_step_bench.py (results/sim_step.csv)."""
import csv, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
out = sys.argv[1] if len(sys.argv) > 1 else "../../lectures/images/lecture-27"
rows = list(csv.DictReader(open("results/sim_step.csv", encoding="utf-8")))
tasks = sorted({(r["suite"], int(r["task_id"])) for r in rows}, key=lambda t: (t[0] != "libero_spatial", t[1]))
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(12, 5))
w = 0.26
cols = {"0": ("#cfe8ff", "#3b6ea8", "physics only (render off)"), "128": ("#ffd6d6", "#c0392b", "physics + 2 cameras 128 px (software EGL)"),
        "256": ("#ffd6d6", "#8e2b20", "physics + 2 cameras 256 px (software EGL)")}
for i, res in enumerate(("0", "128", "256")):
    ys = [float(next(r["step_ms_median"] for r in rows if (r["suite"], int(r["task_id"])) == t and r["res"] == res)) for t in tasks]
    fc, ec, lab = cols[res]
    b = ax.bar(np.arange(len(tasks)) + (i - 1) * w, ys, w, color=fc, edgecolor=ec, label=lab, hatch="//" if res == "256" else None)
    for x, y in zip(np.arange(len(tasks)) + (i - 1) * w, ys):
        ax.text(x, y + 3, f"{y:.0f}", ha="center", fontsize=8)
ax.set_xticks(range(len(tasks)))
ax.set_xticklabels([f"{s.replace('libero_', '')} {t}\n(n_init = {next(r['n_init'] for r in rows if r['suite']==s and int(r['task_id'])==t)})" for s, t in tasks])
ax.set_ylabel("ms / env.step (median, 60 steps)")
host = rows[0]["host"]
ax.set_title(f"Lecture 27 — LIBERO step time, REAL measurement ({host}, no GPU, mujoco 3.3.0, robosuite 1.4.1): "
             "physics 12–23 ms; rendering without a GPU adds 150–240 ms", fontsize=10)
ax.legend(fontsize=8.5); ax.grid(axis="y", alpha=0.25)
fig.tight_layout(); os.makedirs(out, exist_ok=True); fig.savefig(os.path.join(out, "06-sim-measured.png"), dpi=160)
print("written:", os.path.join(out, "06-sim-measured.png"))
