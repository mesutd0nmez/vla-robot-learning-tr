"""Ders 15 — actions.csv'den Tatar-tarzı metrikleri hesaplar: L2, cosine (7-D / 6-D, normalize / gerçek birim),
token uyumsuzluğu, bin mesafesi, gripper uyumu; koşul başına özet tablo + Wilcoxon (n=5) + görsel.

    python analyze_day15.py --actions results/actions.csv --out results/        # gerçek koşu
    python analyze_day15.py --demo --out demo/                                 # SENTETİK veriyle hattı test et

Referans: her görevde `en` koşulu. Karşılaştırılan koşullar: para, tr, tr2, distractor, empty; ek çift: tr→tr2 (TR-içi parafraz).
Metrik tanımları (n = [-1,1] bin merkezi vektörü, a = LIBERO birimli action, a[6] ∈ [0,1]):
  l2_n7, cos_n7   : 7 boyut, normalize uzay (gripper ±0.996 → tek flip L2'ye ~2.0 katar!)
  l2_n6, cos_n6   : ilk 6 boyut, normalize uzay (gripper ayrı raporlanır)  ← önerilen ana metrik
  l2_a7, cos_a7   : 7 boyut, gerçek birim (Tatar vd. ile karşılaştırılabilir; OpenVLA gripper çıktısı ±1 → flip ≈ 2.0)
  tok_mismatch    : 7 token'dan kaçı farklı (0..7)
  max_bin         : en büyük |Δbin| (255 bin üzerinden)
  within5         : |Δbin| ≤ 5 olan boyut oranı (Shin 2026'nın "5-bin accuracy"inin çift versiyonu)
  grip_agree      : LIBERO'ya giden gripper işareti aynı mı (1/0)
"""
import argparse
import csv
import os

import numpy as np

try:
    from scipy.stats import wilcoxon
except Exception:  # scipy yoksa p-değeri atlanır
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
    """SENTETİK veri: hattı test etmek için. Gerçek model çıktısı DEĞİLDİR. Bin'ler rastgele; koşullar
    'para küçük sapma, tr/tr2 orta + bazen gripper flip, distractor farklı hedef, empty uzak' varsayımıyla üretilir."""
    rng = np.random.default_rng(seed)
    recs = {}
    for t in range(5):
        base = np.concatenate([rng.integers(60, 200, 6), [rng.choice([0, 254])]])
        def mk(b):
            b = np.clip(b, 0, 254)
            n, a = bins_to_action(b)
            return dict(bins=b, n=n, a=a, tok=VOCAB - b - 1, g=-np.sign(2 * a[6] - 1), instruction="(sentetik)", prompt_tokens=0)
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
    head = "| çift (ref=en) | n | " + " | ".join(metrics) + " |\n|---|---|" + "---|" * len(metrics) + "\n"
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
    """tr vs para, tr vs empty (görev eşli). n=5'te kesin iki yönlü Wilcoxon'un en küçük p'si 0.0625: anlamlılık İMKÂNSIZ,
    yalnızca etki yönü/büyüklüğü raporlanır."""
    out = []
    by = {(r["task_id"], r["pair"]): r[metric] for r in rows}
    tasks = sorted({t for t, _ in by})
    for a, b in (("tr", "para"), ("tr", "empty"), ("tr", "distractor"), ("tr2", "tr")):
        x = np.array([by[(t, a)] for t in tasks if (t, a) in by and (t, b) in by])
        y = np.array([by[(t, b)] for t in tasks if (t, a) in by and (t, b) in by])
        if len(x) < 2:
            continue
        d = x - y
        line = f"{metric}: {a} − {b}: ortalama fark {d.mean():+.3f} (n={len(d)}, {int((d > 0).sum())}/{len(d)} görevde {a} > {b})"
        if wilcoxon is not None and np.any(d != 0):
            try:
                p = wilcoxon(d, method="exact").pvalue
                line += f", Wilcoxon kesin iki yönlü p = {p:.4f} (n={len(d)} için min 0.0625)"
            except Exception as e:  # noqa
                line += f", Wilcoxon hesaplanamadı ({e})"
        out.append(line)
    return out


def plot(rows, out_png, title_note):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = [p for p in CONDS + ["tr_vs_tr2"] if any(r["pair"] == p for r in rows)]
    labels = {"para": "EN parafraz", "tr": "TR", "tr2": "TR-2", "distractor": "çeldirici (öteki kâse)", "empty": "boş", "tr_vs_tr2": "TR→TR-2"}
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    panels = [("l2_n6", "L2, 6 boyut, normalize [-1,1]"), ("cos_n6", "cosine, 6 boyut, normalize"),
              ("tok_mismatch", "farklı token sayısı (0–7)"), ("grip_agree", "gripper uyumu (görev sayısı)")]
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
    fig.suptitle(f"Ders 15 — koşul başına metrikler (referans: EN; nokta = görev)\n{title_note}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print("görsel:", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", default="results/actions.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--demo", action="store_true", help="SENTETİK veriyle hattı test et (gerçek model çıktısı değildir)")
    ap.add_argument("--png", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.demo:
        recs = make_demo()
        note = "TEMSİLİ (SENTETİK) VERİ — yalnızca analiz hattının testi; gerçek sayılar Mac'te üretilecek"
    else:
        recs = read_actions(args.actions)
        note = f"gerçek koşu: {args.actions}"
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
