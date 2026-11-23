"""Lecture 27 — Budget of the experiment matrix and the inverse power calculation.

  python budget.py                         # ../../thesis/experiment-matrix.csv → table + results/budget.csv
  python budget.py --png ../../lectures/images/lecture-27
  python budget.py --calls-sv 2.5 --calls-ov 0.8   # recompute with your own T4 measurement (s/call)

Episode-time model (every term is written separately so you can substitute measurements as you get them):
  T_ep = t_reset + (W + L) · t_sim + ceil(L / chunk) · t_call
    W = 10 wait steps (num_steps_wait; the model is not called)
    L = mean successful length L_s if the episode succeeds, otherwise max_steps (Spatial 220, Goal 300;
        openvla c8f03f4 run_libero_eval.py:173-180)
    chunk = S-VN 8 (Lecture 12: num_steps_execute 8), S-OV 1 (OpenVLA emits one action per step)
    t_sim = physics + render; physics 22 ms MEASURED (sim_step_bench.py, 2 cores, render off)
  Expected T_ep = p · T_ep(success) + (1 − p) · T_ep(failure); p = the condition's expected success (ESTIMATE).
Every number that was not measured is in SCEN and marked "hypothesis". Power: the same model as rq_power.py
(McNemar exact, inverse noise r), but vectorized with the decomposition b+c ~ Binom(n, Δ+2r), b | m ~ Binom(m, (Δ+r)/(Δ+2r))
(seconds at n=300). n=50, r=0.05 → 23 points (same as Lecture 26; the selftest checks it).
"""
import argparse, csv, math, os
import numpy as np
from scipy.stats import binom

HERE = os.path.dirname(os.path.abspath(__file__))
MATRIX = os.path.join(HERE, "..", "..", "thesis", "experiment-matrix.csv")
MAX_STEPS = {"libero_spatial": 220, "libero_goal": 300}   # real code
L_SUCC = {"libero_spatial": 100, "libero_goal": 120}       # hypothesis (Lecture 18: Spatial ~96; Goal is an estimate)
WAIT = 10
T_RESET = 2.5                                              # measured: 2.4–3.2 s (sim_step_bench.py; first load 5.3)
T_PHYS = 0.022                                             # measured: physics, render off
# expected success per condition — cost mix only (hypothesis); NOT a result
P_SUCC = {"clean": 0.8, "empty": 0.3, "wrong": 0.3, "tr_a": 0.5, "tr_ascii": 0.5, "tr_b": 0.5, "tr_bare": 0.4,
          "en_B": 0.3, "tr_B": 0.25, "tr_mt": 0.7, "tr_parse": 0.75}
# scenarios: (render s/step, S-VN s/call T4, S-OV 4-bit s/call T4, S-OV bf16 MPS s/call)
# the SCEN keys (and the h_* result columns derived from them) stay as they are: other scripts read them; NAME is for display only
NAME = {"iyimser": "optimistic", "orta": "mid", "kötümser": "pessimistic", "ölçüm": "measured"}
SCEN = {"iyimser": (0.005, 1.0, 0.5, 1.5), "orta": (0.010, 2.5, 0.8, 2.0), "kötümser": (0.030, 5.0, 1.6, 3.0)}
CHUNK = {"S-VN": 8, "S-OV": 1, "S-OV-mac": 1, "S-OV-lora": 1}
# the fraction of a cell that really has to run (string cache: if MT/parser output == clean, the episode is the clean episode)
RUN_FRACTION = {"tr_parse": 1 / 80, "tr_mt": 0.6}   # parser: 79/80 identical within the set (Lecture 24, real); MT: hypothesis
LEAN = (20, 0.0)
KAGGLE_WEEK_H = 30.0                                   # user report; check the quota on your own account


def t_episode(system, suite, cond, scen):
    rend, sv, ov, mac = SCEN[scen]
    t_call = {"S-VN": sv, "S-OV": ov, "S-OV-lora": ov, "S-OV-mac": mac}[system]
    t_sim = T_PHYS + rend
    def one(L):
        return T_RESET + (WAIT + L) * t_sim + math.ceil(L / CHUNK[system]) * t_call
    p = P_SUCC.get(cond, 0.5)
    return p * one(L_SUCC[suite]) + (1 - p) * one(MAX_STEPS[suite])


def expand(row):
    """Expands one matrix row into (system, suite, condition, episode) parts."""
    systems = row["system"].split(";")
    suites = row["suite"].split("+")
    conds = [c for c in row["conditions"].split(";") if c != "train"]
    n_tasks, n_init, rep = int(row["n_tasks"]), int(row["n_init"]), int(row["repeats"])
    out = []
    for s in systems:
        for su in suites:
            tasks = n_tasks / len(suites)
            for c in conds:
                ep = tasks * n_init * rep
                out.append((s, su, c, ep, ep * RUN_FRACTION.get(c, 1.0)))
    return out


def budget(rows, n_init_core=None):
    res = []
    for r in rows:
        if n_init_core and r["tier"] == "core" and r["rq"] in ("a", "b", "c") and int(r["n_init"]) == 50:
            r = {**r, "n_init": str(n_init_core)}
        parts = expand(r)
        ep = sum(p[3] for p in parts); ep_run = sum(p[4] for p in parts)
        h = {sc: sum(p[4] * t_episode(p[0], p[1], p[2], sc) for p in parts) / 3600 for sc in SCEN}
        where = "Mac" if parts and all(p[0] == "S-OV-mac" for p in parts) else "T4"
        if r["cell_id"] == "C3":
            h = {"iyimser": 65.0, "orta": 80.0, "kötümser": 98.0}  # ESTIMATE from Lecture 24 qlora_budget.py (training)
        res.append(dict(cell=r["cell_id"], rq=r["rq"], tier=r["tier"], where=where, status=r["status"],
                        episodes=round(ep), episodes_run=round(ep_run),
                        **{f"h_{k}": round(v, 1) for k, v in h.items()}))
    return res


# ---------- power (vectorized, same model as rq_power.py) ----------
def power(n, delta, r, alpha=0.05):
    pb, pc = delta + r, r
    if pb + pc > 1:
        return float("nan")
    m = np.arange(n + 1)
    pm = binom.pmf(m, n, pb + pc)
    tot = 0.0
    th = pb / (pb + pc) if pb + pc > 0 else 0.5
    for mi in m[1:]:
        b = np.arange(mi + 1)
        k = np.minimum(b, mi - b)
        p = np.minimum(1.0, 2 * binom.cdf(k, mi, 0.5))
        tot += pm[mi] * binom.pmf(b, mi, th)[p < alpha].sum()
    return tot


def mde(n, r=0.05, target=0.8):
    for d in np.arange(0.01, 0.96, 0.01):
        if power(n, float(d), r) >= target:
            return float(d)
    return float("nan")


def selftest():
    assert abs(mde(50, 0.05) - 0.23) < 1e-9, mde(50, 0.05)   # Lecture 26 table
    assert abs(mde(100, 0.05) - 0.15) < 1e-9, mde(100, 0.05)
    assert abs(mde(50, 0.0) - 0.16) < 1e-9
    print("selftest OK: MDE(50; r=0.05)=23, MDE(100)=15, MDE(50; r=0)=16 — same as rq_power.py")


def inverse(hours_per_system, scen="orta", suite="libero_spatial", tasks=6, conds=("clean", "tr_a")):
    """Budget (T4 hours, one system, one block, two conditions) → init per task → pooled n → MDE."""
    t = sum(t_episode("S-OV", suite, c, scen) for c in conds)  # the two conditions of one init state (S-OV is the expensive one)
    n_init = min(50, int(hours_per_system * 3600 / (t * tasks)))
    return n_init, tasks * n_init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=MATRIX); ap.add_argument("--png", default=None)
    ap.add_argument("--lean", type=int, default=20, help="lean matrix: init per task in the core cells")
    ap.add_argument("--calls-sv", type=float); ap.add_argument("--calls-ov", type=float)
    a = ap.parse_args()
    if a.calls_sv or a.calls_ov:
        r, sv, ov, mac = SCEN["orta"]; SCEN["ölçüm"] = (r, a.calls_sv or sv, a.calls_ov or ov, mac)
    selftest()
    rows = list(csv.DictReader(open(a.matrix, encoding="utf-8")))
    res = budget(rows)
    keys = list(res[0])
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "budget.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(res)
    print("\nSeconds/episode (mid scenario, expected):")
    for s in ("S-VN", "S-OV", "S-OV-mac"):
        print(f"  {s:9s} Spatial clean {t_episode(s,'libero_spatial','clean','orta'):6.0f}  empty {t_episode(s,'libero_spatial','empty','orta'):6.0f}"
              f" | Goal clean {t_episode(s,'libero_goal','clean','orta'):6.0f}  empty {t_episode(s,'libero_goal','empty','orta'):6.0f}")
    print(f"\n{'cell':5s} {'rq':3s} {'tier':5s} {'at':3s} {'episode':>7s} {'run':>6s} " + " ".join(f"{NAME[k]:>11s}" for k in SCEN) + "  status")
    for r in res:
        print(f"{r['cell']:5s} {r['rq']:3s} {r['tier']:5s} {r['where']:3s} {r['episodes']:7d} {r['episodes_run']:6d} "
              + " ".join(f"{r['h_'+k]:11.1f}" for k in SCEN) + "  " + r["status"])
    print("\nTotals (hours):")
    summ = {}
    for tier in ("pilot", "core", "ext"):
        for where in ("T4", "Mac"):
            sel = [r for r in res if r["tier"] == tier and r["where"] == where]
            if sel:
                tot = {k: sum(r["h_" + k] for r in sel) for k in SCEN}
                summ[(tier, where)] = tot
                print(f"  {tier:5s} {where}: " + "  ".join(f"{NAME[k]} {v:6.1f}" for k, v in tot.items())
                      + (f"   → Kaggle weeks (mid, single T4): {tot['orta']/KAGGLE_WEEK_H:4.1f}; T4×2: {tot['orta']/KAGGLE_WEEK_H/2:4.1f}" if where == "T4" else ""))
    for rq in ("cal", "a", "b", "c"):
        sel = [r for r in res if r["rq"] == rq and r["tier"] != "ext"]
        print(f"  RQ-{rq:3s} (pilot+core): mid {sum(r['h_orta'] for r in sel):6.1f} h, episode {sum(r['episodes_run'] for r in sel)}")
    lean = budget([dict(r) for r in rows], n_init_core=a.lean)
    lt = sum(r["h_orta"] for r in lean if r["tier"] in ("pilot", "core") and r["where"] == "T4")
    ll = {k: sum(r["h_" + k] for r in lean if r["tier"] in ("pilot", "core") and r["where"] == "T4") for k in SCEN}
    print(f"\nLean matrix ({a.lean} init per task in the core, block pool n={6*a.lean}, MDE {100*mde(6*a.lean):.0f} points): "
          + "  ".join(f"{NAME[k]} {v:6.1f}" for k, v in ll.items()) + f"  → Kaggle weeks (mid): {lt/KAGGLE_WEEK_H:.1f}; T4×2: {lt/KAGGLE_WEEK_H/2:.1f}")
    with open(os.path.join(HERE, "results", "budget_lean.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(lean)
    global LEAN
    LEAN = (a.lean, lt)
    print("\nInverse: RQ-a Spatial, S-OV, two conditions (clean, tr_a), 6 tasks — budget → n → MDE (r=0.05):")
    inv = []
    for H in (5, 10, 20, 30, 40, 60):
        ni, npool = inverse(H)
        m1, mp = mde(ni) if ni >= 5 else float("nan"), mde(npool) if npool >= 5 else float("nan")
        inv.append((H, ni, npool, m1, mp))
        print(f"  {H:3d} T4 h → {ni:2d} init per task, pooled n={npool:3d} → MDE task {100*m1:4.0f} / block {100*mp:4.0f} points")
    with open(os.path.join(HERE, "results", "inverse.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t4_hours", "n_init_per_task", "n_pooled", "mde_task", "mde_block"]); w.writerows(inv)
    if a.png:
        plots(a.png, res)


def plots(out, res):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    Y, B, G, R = "#ffd166", "#cfe8ff", "#d4f5d4", "#ffd6d6"
    YD, BD, GD, RD = "#c79a00", "#3b6ea8", "#2e8b57", "#c0392b"
    os.makedirs(out, exist_ok=True)
    # 02 — components of the episode time
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6))
    for ax, suite in zip(axes, ("libero_spatial", "libero_goal")):
        labels, sim, call, rst = [], [], [], []
        for s, nm in (("S-VN", "S-VN\nT4"), ("S-OV", "S-OV\n4-bit T4"), ("S-OV-mac", "S-OV\nbf16 Mac")):
            for sc in SCEN:
                if sc == "ölçüm": continue
                rend, sv, ov, mac = SCEN[sc]; tc = {"S-VN": sv, "S-OV": ov, "S-OV-mac": mac}[s]
                L = MAX_STEPS[suite]  # failed episode = the most expensive case
                labels.append(f"{nm}\n{NAME[sc][:4]}"); rst.append(T_RESET); sim.append((WAIT + L) * (T_PHYS + rend))
                call.append(math.ceil(L / CHUNK[s]) * tc)
        x = np.arange(len(labels))
        ax.bar(x, rst, color="#ddd", edgecolor="#666", label="reset (measured)")
        ax.bar(x, sim, bottom=rst, color=B, edgecolor=BD, label="simulation: physics measured + render (hypothesis)")
        ax.bar(x, call, bottom=np.array(rst) + np.array(sim), color=Y, edgecolor=YD, label="model calls (hypothesis)")
        for xi, tot in zip(x, np.array(rst) + np.array(sim) + np.array(call)):
            ax.text(xi, tot + 8, f"{tot/60:.1f} min", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylabel("seconds / FAILED episode"); ax.set_title(f"{suite} — max_steps {MAX_STEPS[suite]} (+10 wait)", fontsize=10.5)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Lecture 27 — What does an episode spend its time on? S-OV is called every step (chunk 1), S-VN every 8 steps. "
                 "Physics 22 ms and reset 2.5 s were measured in this session; call times are ESTIMATES (scenario)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, "02-episode-cost.png"), dpi=160); plt.close(fig)
    # 03 — T4 hours per cell
    sel = [r for r in res if r["where"] == "T4"]
    fig, ax = plt.subplots(figsize=(14, 5.6))
    x = np.arange(len(sel))
    col = {"pilot": B, "core": Y, "ext": R}
    edge = {"pilot": BD, "core": YD, "ext": RD}
    mid = np.array([r["h_orta"] for r in sel]); lo = np.array([r["h_iyimser"] for r in sel]); hi = np.array([r["h_kötümser"] for r in sel])
    bars = ax.bar(x, mid, color=[col[r["tier"]] for r in sel], edgecolor=[edge[r["tier"]] for r in sel],
                  hatch=None)
    for b_, r in zip(bars, sel):
        if r["status"].startswith("blocked"):
            b_.set_hatch("//")
    ax.errorbar(x, mid, yerr=[mid - lo, hi - mid], fmt="none", ecolor="#444", capsize=3, lw=1)
    for xi, r in zip(x, sel):
        ax.text(xi, hi[list(x).index(xi)] + 2, f"{r['h_orta']:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"{r['cell']}\nRQ-{r['rq']}" for r in sel], fontsize=8)
    ax.axhline(KAGGLE_WEEK_H, color="#555", ls="--", lw=1); ax.text(-0.5, KAGGLE_WEEK_H + 2, "Kaggle ~30 h/week (reported)", ha="left", fontsize=8.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=B, edgecolor=BD, label="pilot"), Patch(facecolor=Y, edgecolor=YD, label="core"),
                       Patch(facecolor=R, edgecolor=RD, label="extension"), Patch(facecolor="white", edgecolor="#444", hatch="//", label="blocked (prerequisite missing)")],
              fontsize=8.5, loc="upper left")
    core = sum(r["h_orta"] for r in sel if r["tier"] in ("pilot", "core"))
    ln, lt = LEAN
    ext = sum(r["h_orta"] for r in sel if r["tier"] == "ext")
    ax.set_ylabel("T4 hours (bar: mid, whisker: optimistic–pessimistic)")
    ax.set_title(f"Lecture 27 — T4 budget per cell (model estimate): pilot+core ≈ {core:.0f} h ≈ {core/KAGGLE_WEEK_H:.1f} Kaggle weeks (with T4×2 ≈ {core/KAGGLE_WEEK_H/2:.1f}); "
                 f"extension +{ext:.0f} h\nlean matrix ({ln} init per task, block n={6*ln}): ≈ {lt:.0f} h ≈ {lt/KAGGLE_WEEK_H:.1f} weeks (T4×2 ≈ {lt/KAGGLE_WEEK_H/2:.1f})", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(out, "03-cell-hours.png"), dpi=160); plt.close(fig)
    # 04 — inverse power
    ns = np.array([10, 20, 30, 50, 60, 90, 120, 150, 180, 240, 300])
    m = np.array([100 * mde(int(n)) for n in ns])
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.plot(ns, m, "-o", color=YD, lw=2, ms=5, label="MDE, 80% power, r = 0.05 (McNemar exact)")
    ax.axvline(50, color=RD, ls="--", lw=1); ax.text(52, 70, "ceiling of a single task:\n50 init states in LIBERO\n(counted on 4 tasks)", fontsize=8.5, color=RD)
    ax.axvline(300, color=GD, ls="--", lw=1); ax.text(296, 70, "block pool:\n6 tasks × 50", fontsize=8.5, color=GD, ha="right")
    ax.axhline(15, color="#555", ls=":", lw=1); ax.text(305, 15.5, "RQ-a threshold 15", fontsize=8.5, ha="right")
    sec = ax.secondary_xaxis("top", functions=(lambda n: n * sum(t_episode("S-OV", "libero_spatial", c, "orta") for c in ("clean", "tr_a")) / 3600,
                                                  lambda h: h * 3600 / sum(t_episode("S-OV", "libero_spatial", c, "orta") for c in ("clean", "tr_a"))))
    sec.set_xlabel("S-OV 4-bit T4 hours (clean + tr_a, mid scenario)")
    for n_, mm in zip(ns, m):
        if n_ in (50, 120, 300): ax.annotate(f"n={n_}: {mm:.0f} points", (n_, mm), textcoords="offset points", xytext=(6, 8), fontsize=8.5)
    ax.set_xlabel("number of paired items n (task × init state)"); ax.set_ylabel("smallest detectable difference (points)")
    ax.set_ylim(0, 85); ax.grid(alpha=0.25); ax.legend(fontsize=9)
    ax.set_title("Lecture 27 — Inverse power: no more than 50 inits per task → 15 points is visible only in the block pool (n≥~100)", fontsize=10.5)
    fig.tight_layout(); fig.savefig(os.path.join(out, "04-inverse-power.png"), dpi=160); plt.close(fig)
    print("written: 02, 03, 04 →", out)


if __name__ == "__main__":
    main()
