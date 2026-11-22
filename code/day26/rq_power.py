"""Ders 26 — RQ'ların ölçütünü sayıya bağlamak: eşli tasarımda (aynı init state, iki koşul)
(a) kaç init state ile hangi büyüklükte fark yakalanır (McNemar exact, kesin güç),
(b) "fark YOK" demek için (TOST / eşdeğerlik) kaç init state gerekir.

  python rq_power.py                 # tabloyu yazdırır
  python rq_power.py --png <klasör>  # 02-mde-tost.png üretir

Model (Ders 18 ile aynı mantık, biraz daha gerçekçi):
  Aynı init state'te iki koşul koşulur. Uyumsuz çift olasılıkları
    P(clean başarır, x başaramaz) = Δ + r        (Δ = gerçek fark, r = "ters" gürültü oranı)
    P(clean başaramaz, x başarır) = r
  b, c bu iki olasılıkla multinom dağılır; McNemar exact p-değeri b+c uyumsuz çift içinde
  iki yönlü binom(0,5) testidir. Güç = p < 0,05 çıkan (b, c) çiftlerinin toplam olasılığı — kesin hesap.
  MDE = %80 güç veren en küçük Δ.
  Eşdeğerlik: TOST %90 güven aralığının yarı genişliği (eşli oranlar farkı için Wald SE =
  sqrt(b + c − (b − c)²/n)/n; planlamada b = c = q·n/2, q = toplam uyumsuzluk oranı) → yarı genişlik
  ≈ 1,645·sqrt(q/n). Eşdeğerlik sınırı δ için gereken n ≈ 1,645²·q/δ².
Kaynak: McNemar exact (klasik); TOST: Lakens 2017 (SPPS 8:355) — yöntem; sayılar bu script'in hesabı.
"""
import argparse, math, os
import numpy as np
from scipy.stats import binom, multinomial

ALPHA = 0.05
Z90 = 1.6449  # TOST için iki tek yönlü %5 test = %90 CI


def mcnemar_p(b, c):
    """İki yönlü exact McNemar p-değeri (uyumsuz çiftler içinde binom(0,5))."""
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * binom.cdf(k, m, 0.5))


def power_paired(n, delta, r):
    """Kesin güç: (b, c) multinom(n; Δ+r, r, 1−Δ−2r) üzerinden toplam."""
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
    print("(a) McNemar exact — %80 güç için en küçük yakalanabilir eşli fark (MDE, puan)")
    print("     n   r=0 (iyimser)   r=0,05   r=0,10")
    for n in ns:
        row = [mde(n, r) for r in (0.0, 0.05, 0.10)]
        print(f"  {n:4d}   " + "   ".join(f"{100*x:6.0f}" if x == x else "   nan" for x in row))
    print("\n(a') Güç: Δ=15 ve Δ=30 puan, r=0,05")
    for n in ns:
        print(f"  n={n:3d}: Δ=15 → {100*power_paired(n,0.15,0.05):5.1f}%   Δ=30 → {100*power_paired(n,0.30,0.05):5.1f}%")
    print("\n(b) TOST — gerçek fark 0 iken %90 CI yarı genişliği (puan); q = uyumsuz çift oranı")
    print("     n    q=0,10   q=0,20   q=0,40")
    for n in ns + [200]:
        print(f"  {n:4d}   " + "   ".join(f"{100*tost_halfwidth(n,q):6.1f}" for q in (0.1, 0.2, 0.4)))
    print("\n(b') Eşdeğerlik sınırı δ için gereken n (q=0,20): " +
          ", ".join(f"δ={int(100*d)} → n≈{n_for_equivalence(d,0.2)}" for d in (0.05, 0.10, 0.15, 0.20, 0.25)))


def png(out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    YD, BD, GD, RD = "#c79a00", "#3b6ea8", "#2e8b57", "#c0392b"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ns = np.array([10, 15, 20, 30, 40, 50, 75, 100])
    ax = axes[0]
    for r, col, lab in [(0.0, GD, "r = 0 (iyimser: koşul yalnız kaybettirir)"),
                        (0.05, YD, "r = 0,05"), (0.10, RD, "r = 0,10 (gürültülü politika)")]:
        ys = [100 * mde(int(n), r) for n in ns]
        ax.plot(ns, ys, "-o", color=col, lw=2, ms=5, label=lab)
    ax.axhline(15, color="#555", ls="--", lw=0.9); ax.text(11, 12.5, "Ders 19 --near eşiği (15 puan)", ha="left", fontsize=8.5, color="#555")
    ax.axhline(25, color="#999", ls=":", lw=0.9); ax.text(11, 22.5, "--far eşiği (25)", ha="left", fontsize=8.5, color="#777")
    ax.set_xlabel("eşli init state sayısı n"); ax.set_ylabel("%80 güçle yakalanan en küçük fark (puan)")
    ax.set_title("(a) RQ-a 'fark var' — McNemar exact, kesin güç", fontsize=11)
    ax.set_ylim(0, 100); ax.grid(alpha=0.25); ax.legend(fontsize=8.5, loc="upper right")
    ax = axes[1]
    ns2 = np.array([10, 20, 30, 50, 75, 100, 150, 200])
    for q, col, lab in [(0.1, GD, "q = 0,10 uyumsuz çift"), (0.2, YD, "q = 0,20"), (0.4, RD, "q = 0,40")]:
        ax.plot(ns2, [100 * tost_halfwidth(int(n), q) for n in ns2], "-o", color=col, lw=2, ms=5, label=lab)
    for d, txt in [(15, "δ = 15: 'tr ≈ clean' kabul sınırı (öneri)"), (10, "δ = 10: 'tr ≈ empty' sınırı (Ders 19)")]:
        ax.axhline(d, color="#555", ls="--", lw=0.9); ax.text(201, d + 1, txt, ha="right", fontsize=8.5, color="#555")
    ax.set_xlabel("eşli init state sayısı n"); ax.set_ylabel("%90 CI yarı genişliği (puan), gerçek fark 0")
    ax.set_title("(b) RQ-a 'fark yok' — TOST eşdeğerlik (Wald yaklaşımı)", fontsize=11)
    ax.set_ylim(0, 40); ax.grid(alpha=0.25); ax.legend(fontsize=8.5)
    fig.suptitle("Ders 26 — 'Fark var' ile 'fark yok' farklı n ister: r=0,05'te n=50 ancak Δ≥23'ü yakalar (Δ=15 için n≈100); "
                 "eşdeğerlik (q=0,2) δ=15 → n≈25, δ=10 → n≈55", fontsize=10.5)
    fig.tight_layout(); os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "02-mde-tost.png"), dpi=160); plt.close(fig)
    print("yazıldı:", os.path.join(out, "02-mde-tost.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--png", default=None)
    a = ap.parse_args()
    table()
    if a.png:
        png(a.png)
