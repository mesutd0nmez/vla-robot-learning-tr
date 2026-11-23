"""Ders 29 — görseller: 02 Gantt (T4 saatleri Ders 27 budget_lean.csv / budget.csv'den), 03 risk matrisi.

  python plot_day29.py [çıktı klasörü]
Gantt'taki hafta aralıkları ÖNERİ (thesis/proposal-v0.1.md §6); saatler Ders 27 bütçe modelinin orta senaryosu.
Risk matrisindeki olasılık/etki puanları bu dersin DEĞERLENDİRMESİ (ölçüm değil).
"""
import csv, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = sys.argv[1] if len(sys.argv) > 1 else "../../lectures/images/lecture-29"
D27 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "day27", "results")
Y, B, G, R = "#ffd166", "#cfe8ff", "#d4f5d4", "#ffd6d6"
YD, BD, GD, RD = "#c79a00", "#3b6ea8", "#2e8b57", "#c0392b"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def hours(fn):
    rows = list(csv.DictReader(open(os.path.join(D27, fn), encoding="utf-8")))
    h = lambda cells: sum(float(r["h_orta"]) for r in rows if r["cell"] in cells)
    return {"İP1": h({"P1", "P2", "K1", "K2"}), "İP2": h({"A1", "A2", "A3", "A4"}),
            "İP3": h({"B1", "B2", "B3", "B4", "B5"}), "İP4": h({"C1", "C2"})}


lean, full = hours("budget_lean.csv"), hours("budget.csv")
print("T4 saat (orta) yalın:", {k: round(v, 1) for k, v in lean.items()}, "toplam", round(sum(lean.values()), 1))
print("T4 saat (orta) tam  :", {k: round(v, 1) for k, v in full.items()}, "toplam", round(sum(full.values()), 1))

# --- 02 Gantt
wp = [("İP1 pilot + kalibrasyon (P1, P2, K1, K2; K3 Mac)", 1, 2, B, BD, "İP1"),
      ("İP2 RQ-a çekirdek (A1–A4)", 2, 5, Y, YD, "İP2"),
      ("İP3 RQ-b (B1–B5)", 5, 7, Y, YD, "İP3"),
      ("İP4 RQ-c (C1, C2)", 7, 8, Y, YD, "İP4"),
      ("İP5 analiz (McNemar / TOST / κ)", 8, 10, G, GD, None),
      ("İP6 tez yazımı", 9, 16, G, GD, None)]
fig, ax = plt.subplots(figsize=(13, 4.8))
for i, (lab, a, b, fc, ec, key) in enumerate(wp):
    yy = len(wp) - 1 - i
    ax.barh(yy, b - a + 1, left=a - 0.5, color=fc, edgecolor=ec, height=0.6)
    txt = lab if key is None else f"{lab} — {lean[key]:.0f} sa yalın / {full[key]:.0f} sa tam"
    ax.text(a - 0.4, yy + 0.38, txt, fontsize=8.5, va="bottom")
for wk, lab in [(5, "ara sınav: RQ-a tablosu"), (16, "son sınav: tez taslağı")]:
    ax.axvline(wk + 0.5, color=RD, ls="--", lw=1); ax.text(wk + 0.55, len(wp) - 0.3, lab, color=RD, fontsize=8.5)
ax.set_yticks([]); ax.set_xticks(range(1, 17)); ax.set_xlim(0.4, 17.6); ax.set_ylim(-0.6, len(wp) + 0.2)
ax.set_xlabel("hafta (hafta 1 = 30 Kasım 2026 — öneri)")
tl, tf = sum(lean.values()), sum(full.values())
ax.set_title(f"Ders 29 — Önerinin takvimi: deney İP'leri yalın matriste ≈ {tl:.0f} T4 sa (tam çekirdek ≈ {tf:.0f}); "
             f"Kaggle ~30 sa/hafta ile ≈ {tl/30:.1f} hafta (tam: {tf/30:.1f}) — saatler Ders 27 bütçe modeli (orta)", fontsize=9.5)
ax.grid(axis="x", alpha=0.25)
fig.tight_layout(); os.makedirs(OUT, exist_ok=True); fig.savefig(os.path.join(OUT, "02-gantt.png"), dpi=160); plt.close(fig)

# --- 03 risk matrisi (olasılık × etki, 1–5, değerlendirme)
risks = [("R1 çağrı süresi yavaş\n(kötümser 497 sa)", 3, 4), ("R2 4-bit S-OV\n84,7'yi tutmaz", 3, 3),
         ("R3 Spatial'da empty\ntavana yakın", 4, 3), ("R4 bddl_cf\nyüklenmez", 2, 3),
         ("R5 LoRA onayı\nyok (S9)", 3, 2), ("R6 VLANeXt T4'te\nkoşmaz", 2, 5), ("R7 MT çıktısı\nüretilemez (HF)", 2, 2)]
fig, ax = plt.subplots(figsize=(9.5, 7))
for x in range(1, 6):
    for y in range(1, 6):
        s = x * y
        c = G if s <= 4 else (Y if s <= 9 else R)
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color=c, alpha=0.55, lw=0))
off = {}
for lab, p, imp in risks:
    k = (p, imp); n = off.get(k, 0); off[k] = n + 1
    ax.plot(p, imp + 0.18 - 0.36 * n, "o", color="#333", ms=5)
    ax.text(p + 0.06, imp + 0.18 - 0.36 * n, lab, fontsize=7.8, va="center")
ax.set_xlim(0.5, 5.5); ax.set_ylim(0.5, 5.5); ax.set_xticks(range(1, 6)); ax.set_yticks(range(1, 6))
ax.set_xlabel("olasılık (1 düşük – 5 yüksek)"); ax.set_ylabel("etki (1 küçük – 5 tezi durdurur)")
ax.legend(handles=[Patch(color=G, alpha=0.55, label="izle"), Patch(color=Y, alpha=0.55, label="B planı hazır olsun"),
                   Patch(color=R, alpha=0.55, label="önce bunu çöz")], loc="lower right", fontsize=8.5)
ax.set_title("Ders 29 — Risk matrisi (puanlar bu dersin değerlendirmesi, ölçüm değil);\nher riskin B planı thesis/proposal-v0.1.md §7 ve dersin §5'inde", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "03-risk-matrix.png"), dpi=160); plt.close(fig)
print("yazıldı: 02-gantt.png, 03-risk-matrix.png →", OUT)
