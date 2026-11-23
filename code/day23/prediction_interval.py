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
