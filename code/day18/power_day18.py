"""Ders 18 — görsel 03 ve 04: 10 episode ne gösterir, ne göstermez? (saf matematik; model/GPU gerekmez)

  python power_day18.py <çıktı_klasörü/>

03-wilson-n10.png : k/n başarıda Wilson %95 aralığı, n = 10 / 20 / 50 (tüm k değerleri). Gerçek hesap.
04-power.png      : clean koşulunun gerçek başarı olasılığı p_clean = 0.95 iken, ikinci koşulun gerçek olasılığı p_x
                    düşükse iki test bu farkı α = 0.05'te yakalar mı?
                    (a) eşsiz: Fisher exact (bağımsız iki binom; tüm (k1,k2) çiftleri üzerinden kesin güç),
                    (b) eşli: McNemar exact — aynı init state'te iki koşul; modelde "clean başarır, x başaramaz" çifti
                        olasılığı p_clean − p_x, tersi 0 varsayımıyla (en iyimser eşli durum: koşul yalnızca kaybettirir).
                    Güç = p < 0.05 çıkan sonuçların toplam olasılığı. Kesin hesap (simülasyon değil).
Kaynak yok (hesap); eğitsel varsayımlar başlıkta yazılı.
"""
import math, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom, fisher_exact

out = (sys.argv[1] if len(sys.argv) > 1 else ".").rstrip("/") + "/"; os.makedirs(out, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
Y, B, G, R = "#ffd166", "#cfe8ff", "#d4f5d4", "#ffd6d6"
YD, BD, GD, RD = "#c79a00", "#3b6ea8", "#2e8b57", "#c0392b"


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


# ---------------- 03: Wilson ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
for ax, n, col in zip(axes, [10, 20, 50], [RD, YD, GD]):
    ks = np.arange(0, n + 1)
    lo = np.array([wilson(k, n)[0] for k in ks]); hi = np.array([wilson(k, n)[1] for k in ks])
    x = ks / n * 100
    ax.fill_between(x, lo * 100, hi * 100, color=col, alpha=0.18, lw=0)
    ax.plot(x, lo * 100, color=col, lw=1.6); ax.plot(x, hi * 100, color=col, lw=1.6)
    ax.plot(x, x, color="#555", lw=0.8, ls="--")
    ax.set_title(f"n = {n} episode", fontsize=11)
    ax.set_xlabel("gözlenen success rate (%)")
    ax.grid(alpha=0.25)
    for k in ([5, 9] if n == 10 else [10, 18] if n == 20 else [25, 45]):
        l, h = wilson(k, n)
        high = k / n > 0.8
        ax.annotate(f"{k}/{n}: {100*l:.0f}–{100*h:.0f}", (k / n * 100, (l * 100 - 7) if high else (h * 100 + 2)),
                    ha="right" if high else "center", fontsize=9, color=col)
    # 10 episode: 9/10 ile 5/10 aralıkları kesişiyor mu?
    if n == 10:
        l9, h9 = wilson(9, 10); l5, h5 = wilson(5, 10)
        ax.axhspan(l9 * 100, h5 * 100, color=R, alpha=0.5, lw=0)
        ax.text(2, (l9 + h5) / 2 * 100, f"9/10 ile 5/10'un\naralıkları çakışıyor\n({100*l9:.0f}–{100*h5:.0f})", fontsize=8.5, color=RD, va="center")
axes[0].set_ylabel("Wilson %95 aralığı (%)")
fig.suptitle("Ders 18 — k/n için Wilson %95 güven aralığı: 10 episode 'yön' verir, 'sayı' vermez (kesin hesap)", fontsize=12)
fig.tight_layout(); fig.savefig(out + "03-wilson-n10.png", dpi=160); plt.close(fig)


# ---------------- 04: güç ----------------
def power_fisher(p1, p2, n, alpha=0.05):
    pw = 0.0
    for k1 in range(n + 1):
        q1 = binom.pmf(k1, n, p1)
        if q1 < 1e-12:
            continue
        for k2 in range(n + 1):
            q2 = binom.pmf(k2, n, p2)
            if q2 < 1e-12:
                continue
            if fisher_exact([[k1, n - k1], [k2, n - k2]])[1] < alpha:
                pw += q1 * q2
    return pw


def mcnemar_p(b, c):
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)


def power_mcnemar(p1, p2, n, alpha=0.05):
    # çift türleri: (✓,✓) p2 ; (✓,✗) p1-p2 ; (✗,✗) 1-p1 ; (✗,✓) 0  — iyimser eşli model
    pb = p1 - p2
    pw = 0.0
    for b in range(n + 1):  # b uyumsuz çift (clean✓, x✗); c = 0 → p = 2 * 0.5**b
        pr = binom.pmf(b, n, pb)
        if mcnemar_p(b, 0) < alpha:
            pw += pr
    return pw


P1 = 0.95
px = np.round(np.arange(0.05, 0.951, 0.05), 2)
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
for ax, (fn, title) in zip(axes, [(power_fisher, "(a) eşsiz: Fisher exact (iki bağımsız grup)"),
                                  (power_mcnemar, "(b) eşli: McNemar exact (aynı init state; iyimser model: tek yönlü uyumsuzluk)")]):
    for n, col, ls in [(10, RD, "-"), (20, YD, "-"), (50, GD, "-")]:
        pw = [fn(P1, p, n) for p in px]
        ax.plot((P1 - px) * 100, np.array(pw) * 100, color=col, lw=2, ls=ls, label=f"n = {n}")
        # bu n için gücün %80'i geçtiği en küçük fark
        ok = [(P1 - p) * 100 for p, w in zip(px, pw) if w >= 0.8]
        if ok:
            ax.annotate(f"≥%80 güç: Δ ≥ {min(ok):.0f} puan", (min(ok), 80), xytext=(min(ok) + 3, 80 - (0 if n == 10 else 9 if n == 20 else 18)),
                        fontsize=9, color=col, arrowprops=dict(arrowstyle="-", color=col, lw=0.8))
    ax.axhline(80, color="#777", lw=0.8, ls=":")
    ax.set_xlabel("gerçek fark: p_clean − p_x (puan); p_clean = 0.95")
    ax.set_title(title, fontsize=10.5); ax.grid(alpha=0.25); ax.legend(frameon=False, loc="lower right")
    for ref, yy, lab in [(0.95 - 0.66, 30, "π0.5 Spatial boş komut\nΔ≈29 (97.5→66, Chen vd.)"), (0.95 - 0.80, 8, "OpenVLA-OFT Spatial\nboş Δ≈14 (93.5→80)")]:
        ax.axvline(ref * 100, color="#999", lw=0.8, ls="--"); ax.text(ref * 100 + 0.8, yy, lab, fontsize=7.5, color="#555")
axes[0].set_ylabel("güç: farkı α=0.05'te yakalama olasılığı (%)")
fig.suptitle("Ders 18 — 10 / 20 / 50 episode hangi farkı güvenilir yakalar? (kesin güç hesabı; p_clean = 0.95 varsayımı)", fontsize=12)
fig.tight_layout(); fig.savefig(out + "04-power.png", dpi=160); plt.close(fig)

# metin özeti (derste kullanılan sayılar)
for n in (10, 20, 50):
    f29 = power_fisher(P1, 0.66, n); m29 = power_mcnemar(P1, 0.66, n)
    f15 = power_fisher(P1, 0.80, n); m15 = power_mcnemar(P1, 0.80, n)
    f55 = power_fisher(P1, 0.40, n); m55 = power_mcnemar(P1, 0.40, n)
    print(f"n={n:2d}  Δ=15: Fisher {100*f15:4.0f}%  McNemar {100*m15:4.0f}% | Δ=29: Fisher {100*f29:4.0f}%  McNemar {100*m29:4.0f}% | Δ=55: Fisher {100*f55:4.0f}%  McNemar {100*m55:4.0f}%")
print("Wilson 9/10: %.3f–%.3f  5/10: %.3f–%.3f  10/10: %.3f–%.3f  0/10: %.3f–%.3f" % (*wilson(9, 10), *wilson(5, 10), *wilson(10, 10), *wilson(0, 10)))
print("McNemar en küçük p, n=10, b uyumsuz: " + ", ".join(f"b={b}: {mcnemar_p(b,0):.4f}" for b in (3, 4, 5, 6, 10)))
