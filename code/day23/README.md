# day23

Lecture 23 — Seven papers of the Cohn line (abstract level). A reading day; two small scripts: one adapts the **prediction interval** of Blackwell, Barry, Cohn (arXiv:2410.03492) to LIBERO runs, the other draws the lecture's figures from the papers' **real table values**.

| File | What | Status |
|---|---|---|
| `prediction_interval.py` | PI = x̄ ± t(α/2, n−1)·s·√(1/n + 1/n') (same as the formula in the paper's appendix); with `--scores` an interval from the success rates of repeat runs; with `--runs-csv` an interval from the repeats of the same cell (model × suite × task × condition) in `results/runs.csv`; with `--simulate` a width-vs-repeats curve with binomial noise only (REPRESENTATIVE) | `--selftest` passed; `--simulate` ran |
| `plot_day23.py` | Figure 02 (StepGame Table 4 + DecompSR Table 6), 03 (DecompSR Tables 1–2 + cardinal direction 2024→2025), 04 (`datasets/04` relation families × Cohn benchmark coverage), 05 (PI width simulation) | ran |

On the Mac (`vla` env; no model, ~10 s):
```bash
cd code/day23
python prediction_interval.py --selftest
python prediction_interval.py --scores 0.7 0.8 0.6            # 3 repeat runs → PI95
python prediction_interval.py --simulate --p 0.9 --episodes 10 50 --max-repeats 10
# when the runs are done:
python prediction_interval.py --runs-csv ../../projects/05-spatial0-4kosul/results/runs.csv --group model,suite,task_id,condition
python plot_day23.py
```

## prediction_interval.py

```python
"""Lecture 23 — the prediction interval of Blackwell, Barry, Cohn (arXiv:2410.03492), adapted to LIBERO runs.

The formula in the paper (section "Prediction interval", same as the Python in its appendix):
    PI = x̄ ± t(α/2, n−1) · s · sqrt(1/n + 1/n')      α = 0.05, n' = n  ->  t_crit · s · sqrt(2/n)
x̄, s : mean / sample std (ddof=1) of the scores from n independent REPEAT RUNS of the same benchmark.
Interpretation: "the score of the next single run falls into this interval with 95% probability" — this is DIFFERENT from the Wilson interval (Lecture 18):
Wilson measures the episode sampling WITHIN a single run, the PI measures the run-to-run variability (seed, flow-matching noise, API).

Usage:
  python prediction_interval.py --selftest
  python prediction_interval.py --scores 0.7 0.8 0.6              # success rates of 3 repeat runs
  python prediction_interval.py --runs-csv ../../projects/05-spatial0-4kosul/results/runs.csv \
         --group model,suite,task_id,condition                    # repeat runs of the same cell -> PI
  python prediction_interval.py --simulate --p 0.7 --episodes 10 50 --max-repeats 10 --trials 2000
"""
import argparse
import csv
import math
import sys
from collections import defaultdict

try:
    from scipy import stats
except ImportError:  # a small fallback from the t table if scipy is missing
    stats = None

# t(0.975, df) — df = 1..30 (if scipy is missing); source: standard t table
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
         11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
         20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
         29: 2.045, 30: 2.042}


def t_crit(df, alpha=0.05):
    if stats is not None:
        return float(stats.t.ppf(1 - alpha / 2, df))
    return _T975[min(df, 30)]


def prediction_interval(scores, alpha=0.05, n_future=None):
    """The PI of the paper. Returns: (mean, half width, lower, upper). If n<2 the half width is inf."""
    xs = [float(x) for x in scores]
    n = len(xs)
    if n < 2:
        return (xs[0] if xs else float("nan")), float("inf"), float("-inf"), float("inf")
    m = sum(xs) / n
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    n2 = n if n_future is None else n_future
    half = t_crit(n - 1, alpha) * s * math.sqrt(1.0 / n + 1.0 / n2)
    return m, half, m - half, m + half


def width(scores, alpha=0.05):
    """Interval width (upper − lower) = 2·half width; the paper sets its '≤ 0.01' threshold on this."""
    return 2 * prediction_interval(scores, alpha)[1]


def simulate_widths(p, episodes, max_repeats, trials, seed=0):
    """REPRESENTATIVE: each 'run' = the success rate of `episodes` independent Bernoulli(p) episodes (binomial noise).
    In a real run seed/flow-matching/env variability is added on top, so the real widths cannot be NARROWER than this.
    Returns: {n_repeats: mean width}."""
    import random
    rng = random.Random(seed)
    out = {}
    for n in range(2, max_repeats + 1):
        ws = []
        for _ in range(trials):
            sc = [sum(rng.random() < p for _ in range(episodes)) / episodes for _ in range(n)]
            ws.append(width(sc))
        out[n] = sum(ws) / len(ws)
    return out


def selftest():
    # 1) closed form: n=3, scores 0.6/0.7/0.8 -> x̄=0.7, s=0.1, t(0.975,2)=4.303, half = 4.303·0.1·sqrt(2/3)=0.3513
    m, h, lo, hi = prediction_interval([0.6, 0.7, 0.8])
    assert abs(m - 0.7) < 1e-12 and abs(h - 4.303 * 0.1 * math.sqrt(2 / 3)) < 2e-3, (m, h)
    # 2) identical scores -> width 0 (deterministic run)
    assert width([0.9, 0.9, 0.9, 0.9]) == 0.0
    # 3) as n' grows the interval narrows but does not go to zero: sqrt(1/n + 1/n') > sqrt(1/n)
    h_n = prediction_interval([0.6, 0.7, 0.8])[1]
    h_inf = prediction_interval([0.6, 0.7, 0.8], n_future=10**9)[1]
    assert h_inf < h_n and h_inf > 0
    # 4) a single run -> infinite (the paper's main warning: the uncertainty of a single score cannot be computed)
    assert prediction_interval([0.7])[1] == float("inf")
    # 5) the threshold in the paper: 3 near-identical repeats (0.590, 0.592, 0.591) -> width ≤ 0.01
    assert width([0.590, 0.592, 0.591]) <= 0.01
    print("selftest OK: PI formula (t·s·sqrt(1/n+1/n')), zero width, effect of n', single run = inf, 0.01 threshold")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--scores", nargs="+", type=float, help="scores of the repeat runs (0–1 or 0–100)")
    ap.add_argument("--runs-csv", help="results/runs.csv (run_id, model, checkpoint, suite, task_id, lang, condition, n_episodes, success_rate)")
    ap.add_argument("--group", default="model,suite,task_id,condition", help="columns that define the cell")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--p", type=float, default=0.7)
    ap.add_argument("--episodes", nargs="+", type=int, default=[10, 50])
    ap.add_argument("--max-repeats", type=int, default=10)
    ap.add_argument("--trials", type=int, default=2000)
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return
    if a.scores:
        sc = [x / 100 if x > 1 else x for x in a.scores]
        m, h, lo, hi = prediction_interval(sc)
        print(f"n={len(sc)}  x̄={m:.3f}  PI95 = {m:.3f} ± {h:.3f}  [{lo:.3f}, {hi:.3f}]  width={2*h:.3f}"
              + ("  -> ≤0.01: enough" if 2 * h <= 0.01 else "  -> add repeats (paper threshold 0.01)"))
        return
    if a.runs_csv:
        cols = a.group.split(",")
        cells = defaultdict(list)
        with open(a.runs_csv, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                try:
                    sr = float(r["success_rate"])
                except (KeyError, ValueError):
                    continue
                cells[tuple(r.get(c, "") for c in cols)].append(sr / 100 if sr > 1 else sr)
        print("cell | n_runs | x̄ | PI95 half | width")
        for key, sc in sorted(cells.items()):
            m, h, lo, hi = prediction_interval(sc)
            print(" / ".join(key), "|", len(sc), "|", f"{m:.3f}", "|", f"{h:.3f}" if h != float("inf") else "inf (single run)", "|",
                  f"{2*h:.3f}" if h != float("inf") else "—")
        return
    if a.simulate:
        for ep in a.episodes:
            w = simulate_widths(a.p, ep, a.max_repeats, a.trials)
            print(f"p={a.p} episodes={ep}: " + "  ".join(f"n={n}:{v:.3f}" for n, v in w.items()))
        return
    ap.print_help()


if __name__ == "__main__":
    sys.exit(main())
```

The file in the repo is the commented/full version of this (its `--runs-csv` output also has a width column).

## plot_day23.py

```python
"""Lecture 23 figures — REAL table values from the seven papers of the Cohn line + the relation labels of datasets/04 + a simulation.

  python plot_day23.py --pairs ../../datasets/04-tr-spatial-commands/data/pairs.csv --out ../../lectures/images/lecture-23

02-depth-collapse.png  collapse as hops/depth increase: corrected StepGame (Li, Hogg, Cohn AAAI-24, Table 4) and DecompSR (McPheat et al. 2025, Table 6)
03-surfaces.png        same question, different surface: DecompSR language (Table 2) + nonce (Table 1); cardinal direction 2024->2025 (Cohn & Blackwell) + the T4 template
04-relation-map.png    relation family of the 20 pairs in datasets/04 × the families covered by the Cohn benchmarks (from the `relation` column of pairs.csv)
05-pi-width.png        REPRESENTATIVE: width of the Blackwell prediction interval, number of repeats × number of episodes (binomial simulation; prediction_interval.py)
Color language: yellow #ffd166 model/key, blue #cfe8ff input, green #d4f5d4 output, red #ffd6d6 problem.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from prediction_interval import simulate_widths

YEL, BLU, GRN, RED = "#ffd166", "#cfe8ff", "#d4f5d4", "#ffd6d6"
YEL_D, BLU_D, GRN_D, RED_D, GREY = "#c99a1e", "#4a7fb5", "#3f9d5a", "#c0504d", "#777777"

# ---- Li, Hogg, Cohn (AAAI-24) arXiv:2401.03991, Table 4 — corrected StepGame, GPT-4 (%; "/" = not reported) ----
K10 = list(range(1, 11))
GPT4_BASE = [100, 70, 55, 45, 40, 25, 40, 35, 35, 25]
GPT4_COT = [None, 80, 75, 95, 85, 85, 90, 80, 60, 65]
GPT4_TOT = [None, None, 85, 85, 90, 90, 85, 90, 100, 95]
TURBO_BASE = [62, 43, 30, 35, 29, 25, 29, 31, 16, 20]

# ---- McPheat et al. 2025 arXiv:2511.02627 (DecompSR), Table 6 — 5-shot, accuracy 0–1 ----
KD = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]
DECOMP = {
    "GPT-3.5":          [0.73, 0.20, 0.16, 0.12, 0.12, 0.15, 0.14, 0.12, 0.11, 0.12, 0.15, 0.12, 0.12],
    "GPT-4o":           [0.99, 0.60, 0.35, 0.29, 0.29, 0.27, 0.20, 0.22, 0.19, 0.21, 0.15, 0.18, 0.15],
    "Claude Sonnet 3.7":[0.98, 0.77, 0.53, 0.41, 0.37, 0.26, 0.22, 0.22, 0.20, 0.20, 0.12, 0.10, 0.09],
    "DeepSeek-R1":      [0.99, 0.67, 0.67, 0.65, 0.62, 0.61, 0.54, 0.48, 0.37, 0.34, 0.14, 0.17, 0.14],
    "o4-mini":          [0.99, 0.83, 0.75, 0.70, 0.65, 0.62, 0.59, 0.58, 0.52, 0.52, 0.36, 0.17, 0.07],
}
# Table 2 (languages, 5-shot) and Table 1 (nonce) — k = 1, 2, 5, 10
KL = [1, 2, 5, 10]
LANG = {
    "GPT-4o":  {"English": [0.99, 0.60, 0.29, 0.21], "Hindi": [0.96, 0.47, 0.22, 0.18], "Swedish": [0.82, 0.43, 0.26, 0.20]},
    "o4-mini": {"English": [0.99, 0.83, 0.65, 0.52], "Hindi": [0.95, 0.73, 0.63, 0.48], "Swedish": [0.30, 0.23, 0.09, 0.07]},
}
NONCE = {"GPT-4o": ([0.99, 0.60, 0.29, 0.21], [0.37, 0.14, 0.14, 0.10]),
         "o4-mini": ([0.99, 0.83, 0.65, 0.52], [0.51, 0.38, 0.34, 0.26])}

# ---- Cohn & Blackwell: COSIT-24 arXiv:2406.16528 ve "Revisited" arXiv:2507.12059 — `large` (5,760 questions), values given in the text ----
CARD = [("random\n8 directions", 0.125, GREY), ("OLMo 2\n'25 worst", 0.16, RED_D), ("gemini-1.0\n'24 worst", 0.25, RED_D),
        ("Mistral 24B\n'25 local", 0.41, BLU_D), ("gpt-3.5-t.\n'24 best", 0.595, YEL_D), ("o1\n'25 best", 0.92, YEL_D)]
T4 = [("gpt-3.5\n'24", 0.41), ("o1\n'25", 0.55), ("GPT-4.5\n'25", 0.0)]

# ---- datasets/04 `relation` -> family; the families covered by the Cohn benchmarks ----
FAMILIES = ["direction (CDC)", "topology (RCC)", "distance", "region / 2 references"]
BENCH_COVER = {  # source: the relation definition of the respective paper (table in lecture §2)
    "StepGame / DecompSR\n(8 directions, grid)":          ["direction (CDC)"],
    "RoomSpace\n(9 directions, TPP/NTPP, near/far)":    ["direction (CDC)", "topology (RCC)", "distance", "region / 2 references"],
    "Cardinal direction\n(8 directions, text)":                ["direction (CDC)"],
    "QSTRBench\n(CDC, RCC-5/8/22, 9IM, STAR)":     ["direction (CDC)", "topology (RCC)"],
}


def family(rel):
    r = rel.lower()   # the relation column of pairs.csv is Turkish data (yön = direction, mesafe = distance); English labels are accepted too
    if r.startswith(("yön", "direction")):
        return "direction (CDC)"
    if r.startswith(("ec", "ntpp", "tpp")):
        return "topology (RCC)"
    if r.startswith(("mesafe", "distance")):
        return "distance"
    return "region / 2 references"


def fig02(out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    def clean(ys):
        return [(k, y) for k, y in zip(K10, ys) if y is not None]
    for ys, col, lab, ls in ((GPT4_BASE, RED_D, "GPT-4, plain prompt", "-"), (GPT4_COT, BLU_D, "GPT-4 + CoT", "--"),
                             (GPT4_TOT, GRN_D, "GPT-4 + ToT(CoT)", "-."), (TURBO_BASE, GREY, "GPT-3.5 Turbo, plain", ":")):
        pts = clean(ys)
        a1.plot([p[0] for p in pts], [p[1] for p in pts], ls, color=col, marker="o", lw=2, label=lab)
    a1.set_xticks(K10); a1.set_ylim(0, 105); a1.set_xlabel("k (number of hops)"); a1.set_ylabel("accuracy (%)")
    a1.set_title("Corrected StepGame — Li, Hogg, Cohn (AAAI-24), Table 4", fontsize=11)
    a1.grid(alpha=0.3); a1.legend(fontsize=9, loc="lower left")
    a1.annotate("100 → 25:\nthe model reads,\nbut cannot carry the chain", xy=(10, 25), xytext=(6.2, 8), fontsize=9,
                arrowprops=dict(arrowstyle="->", color=RED_D), color=RED_D)
    cols = {"GPT-3.5": GREY, "GPT-4o": RED_D, "Claude Sonnet 3.7": YEL_D, "DeepSeek-R1": BLU_D, "o4-mini": GRN_D}
    for name, ys in DECOMP.items():
        a2.plot(KD, ys, marker="o", lw=2, color=cols[name], label=name)
    a2.axhline(0.125, color=GREY, ls=":", lw=1); a2.text(1.05, 0.14, "1/8 = chance (inference)", fontsize=8, color=GREY)
    a2.set_xscale("log"); a2.set_xticks(KD); a2.set_xticklabels([str(k) for k in KD], fontsize=8)
    a2.set_ylim(0, 1.05); a2.set_xlabel("k (depth, log)"); a2.set_ylabel("accuracy (0–1)")
    a2.set_title("DecompSR — McPheat et al. (2025), Table 6 (5-shot)", fontsize=11)
    a2.grid(alpha=0.3, which="both"); a2.legend(fontsize=9, loc="upper right")
    a2.annotate("k=100: everyone\nat chance level", xy=(100, 0.12), xytext=(22, 0.45), fontsize=9,
                arrowprops=dict(arrowstyle="->", color=RED_D), color=RED_D)
    fig.suptitle("Collapse as depth increases: the same curve in 2024 and in 2025 (real paper values)", fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def fig03(out):
    fig = plt.figure(figsize=(14, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.9, 1.15])
    a1, a2, a3 = (fig.add_subplot(gs[0, i]) for i in range(3))
    # (a) languages
    w = 0.13; xs = list(range(len(KL)))
    lang_cols = {"English": YEL_D, "Hindi": BLU_D, "Swedish": RED_D}
    for mi, (model, d) in enumerate(LANG.items()):
        for li, (lang, ys) in enumerate(d.items()):
            off = (mi * 3 + li - 2.5) * w
            a1.bar([x + off for x in xs], ys, w, color=lang_cols[lang], alpha=1.0 if mi == 0 else 0.55,
                   edgecolor="black" if mi == 1 else "none", lw=0.6)
    a1.set_xticks(xs); a1.set_xticklabels([f"k={k}" for k in KL]); a1.set_ylim(0, 1.05)
    a1.set_title("DecompSR Table 2 — the same questions, three languages\n(filled: GPT-4o · outlined: o4-mini)", fontsize=10)
    a1.legend(handles=[Patch(color=c, label=l) for l, c in lang_cols.items()], fontsize=9, loc="upper right")
    a1.set_ylabel("accuracy"); a1.grid(axis="y", alpha=0.3)
    a1.annotate("o4-mini Swedish k=1: 0.30\n(English 0.99)", xy=(0 + (3 + 2 - 2.5) * w, 0.30), xytext=(0.9, 0.78), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color=RED_D), color=RED_D)
    # (b) nonce
    for mi, (model, (en, nn)) in enumerate(NONCE.items()):
        c = YEL_D if mi == 0 else GRN_D
        a2.plot(KL, en, "-o", color=c, lw=2, label=f"{model} English")
        a2.plot(KL, nn, "--s", color=c, lw=2, label=f"{model} nonce direction names")
    a2.set_xticks(KL); a2.set_ylim(0, 1.05); a2.set_xlabel("k"); a2.grid(alpha=0.3)
    a2.set_title("DecompSR Table 1 — direction words\nreplaced with made-up words", fontsize=10)
    a2.legend(fontsize=8, loc="upper right")
    # (c) cardinal
    xs3 = list(range(len(CARD)))
    a3.bar(xs3, [c[1] for c in CARD], color=[c[2] for c in CARD], edgecolor="black", lw=0.5)
    for x, (lab, v, _) in zip(xs3, CARD):
        a3.text(x, v + 0.02, f"{v:.2f}" if v != 0.125 else "0.125", ha="center", fontsize=9)
    a3.set_xticks(xs3); a3.set_xticklabels([c[0] for c in CARD], fontsize=7.5)
    a3.set_ylim(0, 1.08); a3.set_title("Cardinal-direction benchmark (`large`, 5,760 questions)\nCohn & Blackwell 2024 → 2025", fontsize=10)
    a3.grid(axis="y", alpha=0.3)
    ins = a3.inset_axes([0.05, 0.50, 0.40, 0.42])
    ins.bar(range(3), [t[1] for t in T4], color=[BLU_D, GRN_D, RED_D], edgecolor="black", lw=0.5)
    ins.set_xticks(range(3)); ins.set_xticklabels([t[0] for t in T4], fontsize=6.5); ins.set_ylim(0, 0.7)
    ins.set_title("T4 'path' template", fontsize=8); ins.tick_params(axis="y", labelsize=7)
    for i, t in enumerate(T4):
        ins.text(i, t[1] + 0.02, f"{t[1]:.2f}", ha="center", fontsize=7)
    fig.suptitle("Same question, different surface: accuracy changes when the language, the words or the template change (real paper values)", fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def fig04(pairs_csv, out):
    with open(pairs_csv, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    fams = [family(r["relation"]) for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.6, 1]})
    fam_col = {"direction (CDC)": YEL, "topology (RCC)": BLU, "distance": GRN, "region / 2 references": RED}
    # left: 20 pairs × family grid
    for i, (r, fm) in enumerate(zip(rows, fams)):
        j = FAMILIES.index(fm)
        a1.add_patch(plt.Rectangle((j, i), 1, 1, color=fam_col[fm], ec="white"))
        a1.text(j + 0.5, i + 0.5, r["rel_span"], ha="center", va="center", fontsize=8)
        a1.text(-0.1, i + 0.5, f"{r['pair_id']} ({r['block']})", ha="right", va="center", fontsize=8,
                color=RED_D if r["frame_dependent"].strip() in ("evet", "yes") else "black")
    a1.set_xlim(-2.2, len(FAMILIES)); a1.set_ylim(len(rows), 0)
    a1.set_xticks([j + 0.5 for j in range(len(FAMILIES))]); a1.set_xticklabels(FAMILIES, fontsize=9)
    a1.set_yticks([]); a1.xaxis.tick_top()
    for sp in a1.spines.values():
        sp.set_visible(False)
    a1.set_title("datasets/04 — relation family of the 20 pairs (`relation` column; red id = frame-dependent)", fontsize=10, pad=28)
    counts = {fm: fams.count(fm) for fm in FAMILIES}
    a1.text(len(FAMILIES) / 2, len(rows) + 0.9, "   ".join(f"{k}: {v}" for k, v in counts.items()), ha="center", fontsize=9)
    # right: benchmark × family coverage
    for bi, (b, cov) in enumerate(BENCH_COVER.items()):
        for j, fm in enumerate(FAMILIES):
            on = fm in cov
            a2.add_patch(plt.Rectangle((j, bi), 1, 1, color=fam_col[fm] if on else "#f2f2f2", ec="white"))
            a2.text(j + 0.5, bi + 0.5, "✓" if on else "–", ha="center", va="center", fontsize=12, color="black" if on else "#aaaaaa")
        a2.text(-0.1, bi + 0.5, b, ha="right", va="center", fontsize=8.5)
    a2.set_xlim(-3.4, len(FAMILIES)); a2.set_ylim(len(BENCH_COVER), 0)
    a2.set_xticks([j + 0.5 for j in range(len(FAMILIES))]); a2.set_xticklabels([f.replace(" (", "\n(").replace(" / ", " /\n") for f in FAMILIES], fontsize=8.5)
    a2.set_yticks([]); a2.xaxis.tick_top()
    for sp in a2.spines.values():
        sp.set_visible(False)
    a2.set_title("Which family do the Cohn benchmarks test?", fontsize=10, pad=28)
    a2.text(len(FAMILIES) / 2, len(BENCH_COVER) + 0.5, "In LIBERO the gold label comes from the simulator (BDDL predicate);\n"
            "in RoomSpace from an ASP solver, in DecompSR from the construction rule.", ha="center", fontsize=8.5, color=GREY)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def fig05(out):
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    cfg = [(0.7, 10, RED_D, "-"), (0.9, 10, RED_D, "--"), (0.7, 50, BLU_D, "-"), (0.9, 50, BLU_D, "--")]
    for p, ep, col, ls in cfg:
        w = simulate_widths(p, ep, 10, 1500, seed=int(p * 100) + ep)
        ns = sorted(w)
        ax.plot(ns, [w[n] for n in ns], ls, marker="o", color=col, lw=2, label=f"p={p}, {ep} episodes/run")
    ax.axhline(0.01, color=GRN_D, ls=":", lw=2); ax.text(2.1, 0.012, "Blackwell threshold 0.01 (text benchmark, 5,760 questions)", fontsize=8.5, color=GRN_D)
    ax.axhline(0.38, color=GREY, ls="-.", lw=1); ax.text(6.2, 0.40, "cf.: Lecture 18 Wilson 9/10 → [0.60, 0.98] width 0.38", fontsize=8.5, color=GREY)
    ax.set_yscale("log"); ax.set_ylim(0.007, 4); ax.set_xticks(range(2, 11))
    ax.set_xlabel("number of repeat runs n"); ax.set_ylabel("PI95 width (log)")
    ax.set_title("REPRESENTATIVE — prediction interval width, with binomial noise only (prediction_interval.py --simulate)\n"
                 "in a real run seed / flow-matching / env variability is added -> the real width cannot be narrower than this", fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="../../datasets/04-tr-spatial-commands/data/pairs.csv")
    ap.add_argument("--out", default="../../lectures/images/lecture-23")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig02(os.path.join(a.out, "02-depth-collapse.png"))
    fig03(os.path.join(a.out, "03-surfaces.png"))
    fig04(a.pairs, os.path.join(a.out, "04-relation-map.png"))
    fig05(os.path.join(a.out, "05-pi-width.png"))
    print("written:", a.out)


if __name__ == "__main__":
    main()
```

`plot_day23.py` in the repo produces all four figures; this block shows the data constants and the skeleton.
