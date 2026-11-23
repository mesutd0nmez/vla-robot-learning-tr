# day26

Lecture 26 — Start of Phase 4: three candidate research questions. No model; three scripts: the power/equivalence calculation for the RQ criteria, the "definition of done" check of the RQ draft, and the figures. The output file is `thesis/rq-candidates.md`.

| File | What | Status |
|---|---|---|
| `rq_power.py` | In a paired design (the same init state, two conditions) the smallest difference caught with 80% power by McNemar exact (MDE) and the half width of the 90% CI / required n for TOST equivalence; figure 02 with `--png` | ran (`results/rq_power.txt`) |
| `rq_lint.py` | In `thesis/rq-candidates.md`, does every `## RQ-x` section have the 7 fields (Question + 6), an H0 label, a numeric threshold, yes/no branches, a ranking table and a predictions block; `--selftest` catches the 7 gaps in a broken draft; `--csv` matrix. Matches the headings in both the Turkish and the English form | ran: 3 RQ × 7 fields **COMPLETE**, selftest OK |
| `plot_day26.py` | Figure 03 (experiment space: block × condition -> RQ; block counts from `pairs.csv`) and 05 (literature map) | ran |
| `results/rq_power.txt`, `results/rq_matrix.csv` | Real outputs | generated |

On the Mac (`vla` env; only `numpy scipy matplotlib`):
```bash
cd code/day26
python rq_power.py                                  # ~1 min (exact multinomial sums)
python rq_lint.py --selftest
python rq_lint.py ../../thesis/rq-candidates.md --csv results/rq_matrix.csv
python plot_day26.py
```
Exit code of `rq_lint.py`: 0 = COMPLETE, 1 = INCOMPLETE, 2 = selftest failed.

## rq_power.py
```python
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
```

## rq_lint.py
```python
"""Lecture 26 — "definition of done" check of the RQ draft (a preregistration linter).

  python rq_lint.py ../../thesis/rq-candidates.md          # table + exit code (0 = complete, 1 = incomplete)
  python rq_lint.py --selftest                              # shows that it catches the gaps in a broken draft
  python rq_lint.py <file> --csv out.csv                   # matrix CSV (for figure 03 of plot_day26.py)

Rule: every `## RQ-<letter>` section must contain the fields — **Question**, **Null hypothesis**, **Variables**,
**Measure/Criterion and threshold**, **Suite / block / sample**, **Systems**, **If yes / if no, what the thesis says**
(Question + 6 = 7 lines).
Extra checks: at least one "H0" label in the null hypothesis section; a numeric threshold (δ/Δ/points) in the criterion section;
both a yes branch and a no branch in the "yes / no" section; a ranking table (`## Ordering rule`) exists; a predictions block exists.
No model/GPU needed.

The headings are matched in BOTH the Turkish form (the original thesis files: Soru, Null hipotez, ...) and the English form
(Question, Null hypothesis, ...), so the linter works on either language version of thesis/rq-candidates.md. The keys of the output
(CSV column names such as `Soru`, `H0_sayisi`) are kept as they were, because other scripts read them.
"""
import argparse, csv, re, sys

# canonical key (kept as the CSV column name) -> accepted heading texts (Turkish original + English variants)
FIELD_ALIASES = {
    "Soru": ["Soru", "Question"],
    "Null hipotez": ["Null hipotez", "Null hypothesis"],
    "Değişkenler": ["Değişkenler", "Variables"],
    "Ölçüt ve eşik": ["Ölçüt ve eşik", "Criterion and threshold", "Measure and threshold", "Measure and criterion"],
    "Suite / blok / örneklem": ["Suite / blok / örneklem", "Suite / block / sample"],
    "Sistemler": ["Sistemler", "Systems"],
    "Evet / hayır çıkarsa tez ne der": ["Evet / hayır çıkarsa tez ne der", "If yes / if no, what the thesis says",
                                        "What the thesis says if the answer is yes / no",
                                        "What the thesis says if yes / no"],
}
FIELDS = list(FIELD_ALIASES)
# English labels used in messages / the printed table
EN = {"Soru": "Question", "Null hipotez": "Null hypothesis", "Değişkenler": "Variables", "Ölçüt ve eşik": "Criterion and threshold",
      "Suite / blok / örneklem": "Suite / block / sample", "Sistemler": "Systems",
      "Evet / hayır çıkarsa tez ne der": "If yes / if no, what the thesis says"}

RANK_RE = r"^## (Sıralama kuralı|(Ordering|Ranking|Priority|Sorting|Selection) rule)"
PRED_RE = r"Önceden yazılan tahminler|[Pp]redictions"


def split_rqs(text):
    parts = re.split(r"^## (RQ-[a-z]) — (.+)$", text, flags=re.M)
    rqs = {}
    for i in range(1, len(parts), 3):
        body = parts[i + 2]
        body = re.split(r"^## ", body, flags=re.M)[0]  # up to the next ## heading
        rqs[parts[i]] = (parts[i + 1].strip(), body)
    return rqs


def field_text(body, field):
    names = "|".join(re.escape(n) for n in FIELD_ALIASES.get(field, [field]))
    m = re.search(r"\*\*(?:" + names + r"):?\*\*:?\s*(.+?)(?=\n\n\*\*|\n\n---|\Z)", body, flags=re.S)
    return m.group(1).strip() if m else None


def check(text):
    rows, problems = [], []
    rqs = split_rqs(text)
    if not rqs:
        problems.append("no `## RQ-x — …` section at all")
    for rq, (title, body) in rqs.items():
        row = {"rq": rq, "title": title}
        for f in FIELDS:
            t = field_text(body, f)
            row[f] = len(t.split()) if t else 0
            if not t:
                problems.append(f"{rq}: field '{EN[f]}' is missing")
        null = field_text(body, "Null hipotez") or ""
        if not re.search(r"H0", null):
            problems.append(f"{rq}: no H0 label in the null hypothesis")
        crit = field_text(body, "Ölçüt ve eşik") or ""
        if not re.search(r"(δ|Δ|κ)\s*[=≥<>]|\d+\s*(puan|points?)|p\s*<\s*0[.,]05", crit):
            problems.append(f"{rq}: no numeric threshold in the criterion")
        yn = field_text(body, "Evet / hayır çıkarsa tez ne der") or ""
        if not (re.search(r"\*(Evet|Yes)", yn) or re.search(r"etkisi var|reddi|rejected|effect exists|κ ≥", yn)) or not (
                re.search(r"\*(Hayır|No)\b", yn) or re.search(r"Yok|reddedilemedi|could not be rejected|None|κ <", yn)):
            problems.append(f"{rq}: one of the yes/no branches is missing")
        row["H0_sayisi"] = len(set(re.findall(r"H0-[a-z]\d", null)))  # number of distinct H0 labels
        rows.append(row)
    if not re.search(RANK_RE, text, flags=re.M):
        problems.append("ranking table (`## Ordering rule`) is missing")
    if not re.search(PRED_RE, text):
        problems.append("predictions block is missing (mandatory for preregistration)")
    return rows, problems


def report(rows, problems):
    if rows:
        head = ["RQ"] + [EN[f][:14] for f in FIELDS] + ["H0"]
        print(" | ".join(h.ljust(14) for h in head))
        for r in rows:
            print(" | ".join([r["rq"].ljust(14)] + [str(r[f]).ljust(14) for f in FIELDS] + [str(r["H0_sayisi"]).ljust(14)]))
        print("(cells: word count of the field; 0 = missing)")
    print(f"\nProblems: {len(problems)}")
    for p in problems:
        print("  -", p)
    print("RESULT:", "COMPLETE" if not problems else "INCOMPLETE")


SELFTEST_DOC_EN = """# test
## RQ-x — Trial
**Question:** Is there something?
**Null hypothesis:** no difference.
**Variables:** condition.
**Systems:** S-OV.
**If yes / if no, what the thesis says:** *Yes:* good.
"""

SELFTEST_DOC_TR = """# test
## RQ-x — Deneme
**Soru:** Bir şey var mı?
**Null hipotez:** fark yok.
**Değişkenler:** koşul.
**Sistemler:** S-OV.
**Evet / hayır çıkarsa tez ne der:** *Evet:* iyi.
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    if a.selftest:
        expected = {"RQ-x: field 'Criterion and threshold' is missing", "RQ-x: field 'Suite / block / sample' is missing",
                    "RQ-x: no H0 label in the null hypothesis", "RQ-x: no numeric threshold in the criterion",
                    "RQ-x: one of the yes/no branches is missing", "ranking table (`## Ordering rule`) is missing",
                    "predictions block is missing (mandatory for preregistration)"}
        ok = True
        for label, doc in (("English headings", SELFTEST_DOC_EN), ("Turkish headings", SELFTEST_DOC_TR)):
            print(f"--- {label}")
            rows, problems = check(doc)
            report(rows, problems)
            ok = ok and expected <= set(problems)
        print("selftest:", "OK (7 gaps caught, English and Turkish headings)" if ok else "FAILED: the expected gaps were not caught")
        sys.exit(0 if ok else 2)
    if not a.path:
        ap.error("give a file path or --selftest")
    text = open(a.path, encoding="utf-8").read()
    rows, problems = check(text)
    report(rows, problems)
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["rq", "title"] + FIELDS + ["H0_sayisi"])
            for r in rows:
                w.writerow([r["rq"], r["title"]] + [r[f] for f in FIELDS] + [r["H0_sayisi"]])
        print("written:", a.csv)
    sys.exit(0 if not problems else 1)
```

## plot_day26.py
```python
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
```
