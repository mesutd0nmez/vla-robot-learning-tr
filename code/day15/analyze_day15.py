"""Lecture 15 — Computes Tatar-style metrics from actions.csv: L2, cosine (7-D / 6-D, normalized / real units),
token mismatch, bin distance, gripper agreement; per-condition summary table + Wilcoxon (n=5) + figure.

    python analyze_day15.py --actions results/actions.csv --out results/        # real run
    python analyze_day15.py --demo --out demo/                                 # test the pipeline with SYNTHETIC data

Reference: the `en` condition of each task. Compared conditions: para, tr, tr2, distractor, empty; extra pair: tr→tr2 (within-Turkish paraphrase).
Metric definitions (n = [-1,1] bin-center vector, a = action in LIBERO units, a[6] ∈ [0,1]):
  l2_n7, cos_n7   : 7 dims, normalized space (gripper ±0.996 → a single flip adds ~2.0 to L2!)
  l2_n6, cos_n6   : first 6 dims, normalized space (gripper reported separately)  ← recommended main metric
  l2_a7, cos_a7   : 7 dims, real units (comparable with Tatar et al.; OpenVLA gripper output is ±1 → flip ≈ 2.0)
  tok_mismatch    : how many of the 7 tokens differ (0..7)
  max_bin         : largest |Δbin| (out of 255 bins)
  within5         : fraction of dims with |Δbin| ≤ 5 (the paired version of Shin 2026's "5-bin accuracy")
  grip_agree      : is the sign of the gripper sent to LIBERO the same (1/0)
"""
import argparse
import csv
import os

import numpy as np

try:
    from scipy.stats import wilcoxon
except Exception:  # if scipy is missing, the p-value is skipped
    wilcoxon = None

N_BINS = 256
BINS = np.linspace(-1, 1, N_BINS)
BIN_CENTERS = (BINS[:-1] + BINS[1:]) / 2.0
VOCAB = 32000
Q01 = np.array([-0.7454732114076613, -0.6616071462631226, -0.9375, -0.1071428582072258, -0.20678570866584778,
                -0.1842857152223587, 0.0])
Q99 = np.array([0.9375, 0.8758928775787354, 0.9321428537368774, 0.1039285734295845, 0.17678570747375488,
                0.14571428298950195, 1.0])
MASK = np.array([True] * 6 + [False])
CONDS = ["para", "tr", "tr2", "distractor", "empty"]
COLORS = {"en": "#ffd166", "para": "#cfe8ff", "tr": "#ffd166", "tr2": "#f3c86a", "distractor": "#ffd6d6", "empty": "#ffd6d6",
          "tr_vs_tr2": "#d4f5d4"}


def bins_to_action(b):
    n = BIN_CENTERS[np.clip(b, 0, 254)]
    return n, np.where(MASK, 0.5 * (n + 1) * (Q99 - Q01) + Q01, n)


def cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu > 0 and nv > 0 else float("nan")


def pair_metrics(ref, oth):
    """ref/oth: dict(bins=np.array(7), n=..., a=..., tok=..., g=±1)"""
    d = np.abs(ref["bins"] - oth["bins"])
    return dict(
        l2_n7=float(np.linalg.norm(ref["n"] - oth["n"])), cos_n7=cos(ref["n"], oth["n"]),
        l2_n6=float(np.linalg.norm(ref["n"][:6] - oth["n"][:6])), cos_n6=cos(ref["n"][:6], oth["n"][:6]),
        l2_a7=float(np.linalg.norm(ref["a"] - oth["a"])), cos_a7=cos(ref["a"], oth["a"]),
        tok_mismatch=int(np.sum(ref["tok"] != oth["tok"])), max_bin=int(d.max()),
        within5=float(np.mean(d <= 5)), grip_agree=int(ref["g"] == oth["g"]),
    )


def read_actions(path):
    recs = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if int(r.get("repeat", 0)) != 0:
            continue
        b = np.array([int(r[f"bin{i}"]) for i in range(7)])
        n, a = bins_to_action(b)
        recs[(int(r["task_id"]), r["condition"])] = dict(
            bins=b, n=n, a=a, tok=np.array([int(r[f"tok{i}"]) for i in range(7)]),
            g=float(r["gripper_libero"]), instruction=r["instruction"], prompt_tokens=int(r["prompt_tokens"]))
    return recs


def make_demo(seed=5):
    """SYNTHETIC data: for testing the pipeline. It is NOT real model output. Bins are random; the conditions
    are generated under the assumption 'para small deviation, tr/tr2 medium + sometimes a gripper flip, distractor different target, empty far'."""
    rng = np.random.default_rng(seed)
    recs = {}
    for t in range(5):
        base = np.concatenate([rng.integers(60, 200, 6), [rng.choice([0, 254])]])
        def mk(b):
            b = np.clip(b, 0, 254)
            n, a = bins_to_action(b)
            return dict(bins=b, n=n, a=a, tok=VOCAB - b - 1, g=-np.sign(2 * a[6] - 1), instruction="(synthetic)", prompt_tokens=0)
        recs[(t, "en")] = mk(base)
        recs[(t, "para")] = mk(base + np.concatenate([rng.integers(-3, 4, 6), [0]]))
        for c in ("tr", "tr2"):
            b = base + np.concatenate([rng.integers(-25, 26, 6), [0]])
            if rng.random() < 0.4:
                b[6] = 254 - b[6]
            recs[(t, c)] = mk(b)
        recs[(t, "distractor")] = mk(base + np.concatenate([rng.integers(-60, 61, 6), [0]]))
        b = base + np.concatenate([rng.integers(-80, 81, 6), [0]])
        if rng.random() < 0.6:
            b[6] = 254 - b[6]
        recs[(t, "empty")] = mk(b)
    return recs


def summarize(recs):
    rows = []
    tasks = sorted({t for t, _ in recs})
    for t in tasks:
        if (t, "en") not in recs:
            continue
        for c in CONDS:
            if (t, c) in recs:
                rows.append(dict(task_id=t, pair=c, **pair_metrics(recs[(t, "en")], recs[(t, c)])))
        if (t, "tr") in recs and (t, "tr2") in recs:
            rows.append(dict(task_id=t, pair="tr_vs_tr2", **pair_metrics(recs[(t, "tr")], recs[(t, "tr2")])))
    return rows


def table(rows, metrics=("l2_n6", "cos_n6", "l2_a7", "tok_mismatch", "max_bin", "within5", "grip_agree")):
    pairs = [p for p in CONDS + ["tr_vs_tr2"] if any(r["pair"] == p for r in rows)]
    head = "| pair (ref=en) | n | " + " | ".join(metrics) + " |\n|---|---|" + "---|" * len(metrics) + "\n"
    body = ""
    for p in pairs:
        sub = [r for r in rows if r["pair"] == p]
        cells = []
        for m in metrics:
            v = np.array([r[m] for r in sub], dtype=float)
            if m == "grip_agree":
                cells.append(f"{int(v.sum())}/{len(v)}")
            elif m in ("tok_mismatch", "max_bin"):
                cells.append(f"{v.mean():.1f}")
            else:
                cells.append(f"{np.nanmean(v):.3f} ± {np.nanstd(v):.3f}")
        body += f"| {'tr→tr2' if p == 'tr_vs_tr2' else 'en→' + p} | {len(sub)} | " + " | ".join(cells) + " |\n"
    return head + body


def tests(rows, metric="l2_n6"):
    """tr vs para, tr vs empty (paired by task). With n=5 the smallest p of the exact two-sided Wilcoxon is 0.0625: significance is IMPOSSIBLE,
    only the effect direction/size is reported."""
    out = []
    by = {(r["task_id"], r["pair"]): r[metric] for r in rows}
    tasks = sorted({t for t, _ in by})
    for a, b in (("tr", "para"), ("tr", "empty"), ("tr", "distractor"), ("tr2", "tr")):
        x = np.array([by[(t, a)] for t in tasks if (t, a) in by and (t, b) in by])
        y = np.array([by[(t, b)] for t in tasks if (t, a) in by and (t, b) in by])
        if len(x) < 2:
            continue
        d = x - y
        line = f"{metric}: {a} − {b}: mean difference {d.mean():+.3f} (n={len(d)}, {a} > {b} in {int((d > 0).sum())}/{len(d)} tasks)"
        if wilcoxon is not None and np.any(d != 0):
            try:
                p = wilcoxon(d, method="exact").pvalue
                line += f", Wilcoxon exact two-sided p = {p:.4f} (min 0.0625 for n={len(d)})"
            except Exception as e:  # noqa
                line += f", Wilcoxon could not be computed ({e})"
        out.append(line)
    return out


def plot(rows, out_png, title_note):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = [p for p in CONDS + ["tr_vs_tr2"] if any(r["pair"] == p for r in rows)]
    labels = {"para": "EN paraphrase", "tr": "TR", "tr2": "TR-2", "distractor": "distractor (other bowl)", "empty": "empty", "tr_vs_tr2": "TR→TR-2"}
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    panels = [("l2_n6", "L2, 6 dims, normalized [-1,1]"), ("cos_n6", "cosine, 6 dims, normalized"),
              ("tok_mismatch", "number of differing tokens (0–7)"), ("grip_agree", "gripper agreement (number of tasks)")]
    for ax, (m, lab) in zip(axes, panels):
        xs = np.arange(len(pairs))
        for i, p in enumerate(pairs):
            v = np.array([r[m] for r in rows if r["pair"] == p], dtype=float)
            if m == "grip_agree":
                ax.bar(i, v.sum(), color=COLORS[p], edgecolor="#333")
                ax.set_ylim(0, max(5, len(v)))
            else:
                ax.bar(i, np.nanmean(v), color=COLORS[p], edgecolor="#333")
                ax.scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, color="#333", s=18, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels([labels[p] for p in pairs], rotation=30, ha="right", fontsize=8)
        ax.set_title(lab, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Lecture 15 — metrics per condition (reference: EN; dot = task)\n{title_note}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print("figure:", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", default="results/actions.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--demo", action="store_true", help="test the pipeline with SYNTHETIC data (not real model output)")
    ap.add_argument("--png", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.demo:
        recs = make_demo()
        note = "ILLUSTRATIVE (SYNTHETIC) DATA — only a test of the analysis pipeline; the real numbers will be produced on the Mac"
    else:
        recs = read_actions(args.actions)
        note = f"real run: {args.actions}"
    rows = summarize(recs)
    with open(os.path.join(args.out, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    md = table(rows)
    tst = tests(rows)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(("**" + note + "**\n\n") + md + "\n" + "\n".join("- " + t for t in tst) + "\n")
    print(note + "\n")
    print(md)
    print("\n".join(tst))
    plot(rows, args.png or os.path.join(args.out, "metrics.png"), note)


if __name__ == "__main__":
    main()
