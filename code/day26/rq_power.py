"""Lecture 26 — tying the criteria of the RQs to numbers: in a paired design (the same init state, two conditions)
(a) with how many init states which size of difference can be caught (McNemar exact, exact power),
(b) how many init states are needed to say "NO difference" (TOST / equivalence).

  python rq_power.py                 # prints the table
  python rq_power.py --png <folder>  # produces 02-mde-tost.png

Model (the same logic as Lecture 18, a little more realistic):
  Two conditions are run on the same init state. The discordant-pair probabilities are
    P(clean succeeds, x fails) = Δ + r        (Δ = true difference, r = "reverse" noise rate)
    P(clean fails, x succeeds) = r
  (b, c) is multinomial with these two probabilities; the McNemar exact p-value is a two-sided
  binomial(0.5) test within the b+c discordant pairs. Power = total probability of the (b, c) pairs with p < 0.05 — an exact calculation.
  MDE = the smallest Δ that gives 80% power.
  Equivalence: half width of the TOST 90% confidence interval (for the paired difference of proportions Wald SE =
  sqrt(b + c − (b − c)²/n)/n; in planning b = c = q·n/2, q = total discordance rate) -> half width
  ≈ 1.645·sqrt(q/n). The n needed for an equivalence bound δ is ≈ 1.645²·q/δ².
Source: McNemar exact (classical); TOST: Lakens 2017 (SPPS 8:355) — the method; the numbers are this script's calculation.
"""
import argparse, math, os
import numpy as np
from scipy.stats import binom, multinomial

ALPHA = 0.05
Z90 = 1.6449  # TOST: two one-sided 5% tests = 90% CI


def mcnemar_p(b, c):
    """Two-sided exact McNemar p-value (binomial(0.5) within the discordant pairs)."""
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * binom.cdf(k, m, 0.5))


def power_paired(n, delta, r):
    """Exact power: the sum over the multinomial (b, c) ~ (n; Δ+r, r, 1−Δ−2r)."""
    p_b, p_c = delta + r, r
    p_rest = 1 - p_b - p_c
    if p_rest < 0:
        return float("nan")
    total = 0.0
    for b in range(n + 1):
        for c in range(n - b + 1):
            if mcnemar_p(b, c) < ALPHA:
                total += multinomial.pmf([b, c, n - b - c], n, [p_b, p_c, p_rest])
    return total


def mde(n, r, target=0.8, grid=np.arange(0.01, 0.96, 0.01)):
    for d in grid:
        if power_paired(n, float(d), r) >= target:
            return float(d)
    return float("nan")


def tost_halfwidth(n, q):
    return Z90 * math.sqrt(q / n)


def n_for_equivalence(delta, q):
    return math.ceil(Z90 ** 2 * q / delta ** 2)


def table():
    ns = [10, 20, 30, 50, 100]
    print("(a) McNemar exact — the smallest catchable paired difference for 80% power (MDE, points)")
    print("     n   r=0 (optimistic)   r=0.05   r=0.10")
    for n in ns:
        row = [mde(n, r) for r in (0.0, 0.05, 0.10)]
        print(f"  {n:4d}   " + "   ".join(f"{100*x:6.0f}" if x == x else "   nan" for x in row))
    print("\n(a') Power: Δ=15 and Δ=30 points, r=0.05")
    for n in ns:
        print(f"  n={n:3d}: Δ=15 → {100*power_paired(n,0.15,0.05):5.1f}%   Δ=30 → {100*power_paired(n,0.30,0.05):5.1f}%")
    print("\n(b) TOST — half width of the 90% CI when the true difference is 0 (points); q = discordant-pair rate")
    print("     n    q=0.10   q=0.20   q=0.40")
    for n in ns + [200]:
        print(f"  {n:4d}   " + "   ".join(f"{100*tost_halfwidth(n,q):6.1f}" for q in (0.1, 0.2, 0.4)))
    print("\n(b') n needed for an equivalence bound δ (q=0.20): " +
          ", ".join(f"δ={int(100*d)} → n≈{n_for_equivalence(d,0.2)}" for d in (0.05, 0.10, 0.15, 0.20, 0.25)))


def png(out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    YD, BD, GD, RD = "#c79a00", "#3b6ea8", "#2e8b57", "#c0392b"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ns = np.array([10, 15, 20, 30, 40, 50, 75, 100])
    ax = axes[0]
    for r, col, lab in [(0.0, GD, "r = 0 (optimistic: the condition can only lose)"),
                        (0.05, YD, "r = 0.05"), (0.10, RD, "r = 0.10 (noisy policy)")]:
        ys = [100 * mde(int(n), r) for n in ns]
        ax.plot(ns, ys, "-o", color=col, lw=2, ms=5, label=lab)
    ax.axhline(15, color="#555", ls="--", lw=0.9); ax.text(11, 12.5, "Lecture 19 --near threshold (15 points)", ha="left", fontsize=8.5, color="#555")
    ax.axhline(25, color="#999", ls=":", lw=0.9); ax.text(11, 22.5, "--far threshold (25)", ha="left", fontsize=8.5, color="#777")
    ax.set_xlabel("number of paired init states n"); ax.set_ylabel("smallest difference caught with 80% power (points)")
    ax.set_title("(a) RQ-a 'there is a difference' — McNemar exact, exact power", fontsize=11)
    ax.set_ylim(0, 100); ax.grid(alpha=0.25); ax.legend(fontsize=8.5, loc="upper right")
    ax = axes[1]
    ns2 = np.array([10, 20, 30, 50, 75, 100, 150, 200])
    for q, col, lab in [(0.1, GD, "q = 0.10 discordant pairs"), (0.2, YD, "q = 0.20"), (0.4, RD, "q = 0.40")]:
        ax.plot(ns2, [100 * tost_halfwidth(int(n), q) for n in ns2], "-o", color=col, lw=2, ms=5, label=lab)
    for d, txt in [(15, "δ = 15: 'tr ≈ clean' acceptance bound (proposal)"), (10, "δ = 10: 'tr ≈ empty' bound (Lecture 19)")]:
        ax.axhline(d, color="#555", ls="--", lw=0.9); ax.text(201, d + 1, txt, ha="right", fontsize=8.5, color="#555")
    ax.set_xlabel("number of paired init states n"); ax.set_ylabel("90% CI half width (points), true difference 0")
    ax.set_title("(b) RQ-a 'no difference' — TOST equivalence (Wald approximation)", fontsize=11)
    ax.set_ylim(0, 40); ax.grid(alpha=0.25); ax.legend(fontsize=8.5)
    fig.suptitle("Lecture 26 — 'There is a difference' and 'no difference' need different n: at r=0.05, n=50 catches only Δ≥23 (n≈100 for Δ=15); "
                 "equivalence (q=0.2) δ=15 -> n≈25, δ=10 -> n≈55", fontsize=10.5)
    fig.tight_layout(); os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "02-mde-tost.png"), dpi=160); plt.close(fig)
    print("written:", os.path.join(out, "02-mde-tost.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--png", default=None)
    a = ap.parse_args()
    table()
    if a.png:
        png(a.png)
