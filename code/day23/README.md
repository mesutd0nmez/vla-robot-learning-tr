# day23

Ders 23 — Cohn hattındaki yedi makale (abstract düzeyi). Okuma günü; iki küçük script: biri Blackwell, Barry, Cohn'un (arXiv:2410.03492) **tahmin aralığını** LIBERO koşularına uyarlar, öteki dersin görsellerini makalelerin **gerçek tablo değerlerinden** çizer.

| Dosya | Ne | Durum |
|---|---|---|
| `prediction_interval.py` | PI = x̄ ± t(α/2, n−1)·s·√(1/n + 1/n') (makalenin ekindeki formülle aynı); `--scores` ile tekrar koşularının success rate'lerinden aralık; `--runs-csv` ile `results/runs.csv`'deki aynı hücrenin (model × suite × task × condition) tekrarlarından aralık; `--simulate` ile yalnızca binom gürültüsüyle genişlik-tekrar eğrisi (TEMSİLİ) | `--selftest` geçti; `--simulate` çalıştırıldı |
| `plot_day23.py` | Görsel 02 (StepGame Tablo 4 + DecompSR Tablo 6), 03 (DecompSR Tablo 1–2 + kardinal yön 2024→2025), 04 (`datasets/04` ilişki aileleri × Cohn benchmark kapsamı), 05 (PI genişliği simülasyonu) | çalıştırıldı |

Mac'te (`vla` env; model yok, ~10 sn):
```bash
cd code/day23
python prediction_interval.py --selftest
python prediction_interval.py --scores 0.7 0.8 0.6            # 3 tekrar koşusu → PI95
python prediction_interval.py --simulate --p 0.9 --episodes 10 50 --max-repeats 10
# koşuların bitince:
python prediction_interval.py --runs-csv ../../projects/05-spatial0-4kosul/results/runs.csv --group model,suite,task_id,condition
python plot_day23.py
```

## prediction_interval.py

```python
"""Ders 23 — Blackwell, Barry, Cohn (arXiv:2410.03492) tahmin aralığı (prediction interval), LIBERO koşularına uyarlanmış.

Makaledeki formül (bölüm "Prediction interval", ekteki Python ile aynı):
    PI = x̄ ± t(α/2, n−1) · s · sqrt(1/n + 1/n')      α = 0.05, n' = n  →  t_crit · s · sqrt(2/n)
x̄, s : aynı benchmark'ın n bağımsız TEKRAR KOŞUSUNDAN gelen skorların ortalaması / örneklem std'si (ddof=1).
Yorum: "bir sonraki tek koşunun skoru %95 olasılıkla bu aralığa düşer" — bu, Wilson aralığından (Ders 18) FARKLI bir şeydir:
Wilson tek koşu İÇİNDEKİ episode örneklemesini, PI koşudan koşuya (seed, flow-matching gürültüsü, API) değişkenliği ölçer.
"""
import argparse, csv, math, sys
from collections import defaultdict
try:
    from scipy import stats
except ImportError:
    stats = None

_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
         11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
         20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
         29: 2.045, 30: 2.042}

def t_crit(df, alpha=0.05):
    if stats is not None:
        return float(stats.t.ppf(1 - alpha / 2, df))
    return _T975[min(df, 30)]

def prediction_interval(scores, alpha=0.05, n_future=None):
    """Makaledeki PI. Döndürür: (ortalama, yarı genişlik, alt, üst). n<2 ise yarı genişlik inf."""
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
    return 2 * prediction_interval(scores, alpha)[1]

def simulate_widths(p, episodes, max_repeats, trials, seed=0):
    """TEMSİLİ: her 'koşu' = `episodes` bağımsız Bernoulli(p) episode'unun success rate'i (binom gürültüsü)."""
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
    m, h, lo, hi = prediction_interval([0.6, 0.7, 0.8])
    assert abs(m - 0.7) < 1e-12 and abs(h - 4.303 * 0.1 * math.sqrt(2 / 3)) < 2e-3, (m, h)
    assert width([0.9, 0.9, 0.9, 0.9]) == 0.0
    h_n = prediction_interval([0.6, 0.7, 0.8])[1]
    h_inf = prediction_interval([0.6, 0.7, 0.8], n_future=10**9)[1]
    assert h_inf < h_n and h_inf > 0
    assert prediction_interval([0.7])[1] == float("inf")
    assert width([0.590, 0.592, 0.591]) <= 0.01
    print("selftest OK")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--scores", nargs="+", type=float)
    ap.add_argument("--runs-csv")
    ap.add_argument("--group", default="model,suite,task_id,condition")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--p", type=float, default=0.7)
    ap.add_argument("--episodes", nargs="+", type=int, default=[10, 50])
    ap.add_argument("--max-repeats", type=int, default=10)
    ap.add_argument("--trials", type=int, default=2000)
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if a.scores:
        sc = [x / 100 if x > 1 else x for x in a.scores]
        m, h, lo, hi = prediction_interval(sc)
        print(f"n={len(sc)}  x̄={m:.3f}  PI95 = {m:.3f} ± {h:.3f}  [{lo:.3f}, {hi:.3f}]  genişlik={2*h:.3f}")
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
        for key, sc in sorted(cells.items()):
            m, h, lo, hi = prediction_interval(sc)
            print(" / ".join(key), "|", len(sc), "|", f"{m:.3f}", "|", f"{h:.3f}" if h != float("inf") else "inf (tek koşu)")
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

Repodaki dosya bunun yorumlu/tam hâli (`--runs-csv` çıktısında genişlik sütunu da var).

## plot_day23.py

```python
"""Ders 23 görselleri — makalelerin GERÇEK tablo değerleri + datasets/04 + bir simülasyon.
  python plot_day23.py --pairs ../../datasets/04-tr-spatial-commands/data/pairs.csv --out ../../lectures/images/lecture-23
02 derinlik çöküşü (StepGame Tablo 4; DecompSR Tablo 6) · 03 yüzey etkisi (DecompSR Tablo 1–2; kardinal yön) ·
04 datasets/04 ilişki aileleri × Cohn kapsamı · 05 PI genişliği simülasyonu (TEMSİLİ)
"""
import argparse, csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from prediction_interval import simulate_widths

# Li, Hogg, Cohn (AAAI-24) Tablo 4 — düzeltilmiş StepGame, GPT-4 (%)
K10 = list(range(1, 11))
GPT4_BASE = [100, 70, 55, 45, 40, 25, 40, 35, 35, 25]
GPT4_COT  = [None, 80, 75, 95, 85, 85, 90, 80, 60, 65]
GPT4_TOT  = [None, None, 85, 85, 90, 90, 85, 90, 100, 95]
# McPheat vd. (DecompSR) Tablo 6 — 5-shot
KD = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]
DECOMP = {"GPT-4o":  [0.99, 0.60, 0.35, 0.29, 0.29, 0.27, 0.20, 0.22, 0.19, 0.21, 0.15, 0.18, 0.15],
          "o4-mini": [0.99, 0.83, 0.75, 0.70, 0.65, 0.62, 0.59, 0.58, 0.52, 0.52, 0.36, 0.17, 0.07]}
# Tablo 2 (diller) — k = 1, 2, 5, 10
LANG = {"GPT-4o": {"en": [0.99, 0.60, 0.29, 0.21], "hi": [0.96, 0.47, 0.22, 0.18], "sv": [0.82, 0.43, 0.26, 0.20]},
        "o4-mini": {"en": [0.99, 0.83, 0.65, 0.52], "hi": [0.95, 0.73, 0.63, 0.48], "sv": [0.30, 0.23, 0.09, 0.07]}}

def family(rel):
    r = rel.lower()
    if r.startswith("yön"): return "yön (CDC)"
    if r.startswith(("ec", "ntpp", "tpp")): return "topoloji (RCC)"
    if r.startswith("mesafe"): return "mesafe"
    return "bölge / 2 referans"

def fig02(out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for ys, lab in ((GPT4_BASE, "GPT-4 düz"), (GPT4_COT, "GPT-4 + CoT"), (GPT4_TOT, "GPT-4 + ToT")):
        pts = [(k, y) for k, y in zip(K10, ys) if y is not None]
        a1.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=lab)
    a1.set_xlabel("k (hop)"); a1.set_ylabel("%"); a1.legend()
    for name, ys in DECOMP.items():
        a2.plot(KD, ys, marker="o", label=name)
    a2.set_xscale("log"); a2.axhline(0.125, ls=":"); a2.legend()
    fig.savefig(out, dpi=160)

# fig03 (diller + nonce + kardinal), fig04 (pairs.csv → aile ızgarası), fig05 (simulate_widths) repodaki dosyada.

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="../../datasets/04-tr-spatial-commands/data/pairs.csv")
    ap.add_argument("--out", default="../../lectures/images/lecture-23")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    fig02(os.path.join(a.out, "02-depth-collapse.png"))

if __name__ == "__main__":
    main()
```

Repodaki `plot_day23.py` dört görselin tamamını üretir; bu blok veri sabitlerini ve iskeleti gösterir.
