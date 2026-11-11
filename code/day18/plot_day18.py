"""Ders 18 — görsel 02 ve 06: literatürdeki 'boş komut' ve 'yanlış komut' sayıları (elle aktarıldı; model/GPU gerekmez).

  python plot_day18.py <çıktı_klasörü/>

02-empty-literature.png  LIBERO-Spatial'da doğru İngilizce komut vs boş komut (Ø), model başına.
   Kaynaklar: Chen vd. 2026 (arXiv:2606.15714) Tablo 6 — Spatial sütunları en / Ø (OpenVLA-OFT satırı makalede ABot-M0 ile
   aynı; makalenin kendi Tablo 1'iyle çelişiyor → 'şüpheli' işareti); Zhan vd. 2026 (arXiv:2601.04052) Tablo 8/9 — π0 ve π0.5
   correct / blank; Fang vd. 2026 (arXiv:2602.17659) Tablo I — π0.5 vision-only (dil girdisi kaldırılmış) Spatial 68.2.
06-wrong-literature.png  LIBERO-CF CF-Spatial (Fang vd. Tablo III, CAG'siz): yanlış (karşı-olgusal) komutta model hangi görevi yapıyor?
   Grounding/Success × Faithful (komuta sadık) / Biased (eğitimdeki görevi yapıyor).
"""
import os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out = (sys.argv[1] if len(sys.argv) > 1 else ".").rstrip("/") + "/"; os.makedirs(out, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
Y, B, G, R = "#ffd166", "#cfe8ff", "#d4f5d4", "#ffd6d6"
YD, BD, GD, RD = "#c79a00", "#3b6ea8", "#2e8b57", "#c0392b"

# ---------- 02 ----------
# (etiket, en, empty, kaynak kısaltması, not)
ROWS = [
    ("OpenVLA-OFT", 93.5, 80.0, "Chen T6", "satır şüpheli"),
    ("π0.5", 97.5, 66.0, "Chen T6", ""),
    ("π0.5", 95.4, 65.0, "Zhan T8/9", ""),
    ("π0.5 (vision-only)", 97.2, 68.2, "Fang T1", "dil girdisi yok"),
    ("π0", 96.8, 29.8, "Zhan T8/9", ""),
    ("Cosmos Policy", 96.5, 55.5, "Chen T6", ""),
    ("Qwen3-VL-OFT", 92.5, 34.0, "Chen T6", ""),
    ("Qwen3-VL-FAST", 89.0, 12.5, "Chen T6", ""),
    ("Qwen3-VL-GR00T", 94.0, 64.5, "Chen T6", ""),
    ("Qwen3-VL-π", 93.0, 41.0, "Chen T6", ""),
    ("Qwen2.5-VL-OFT", 91.5, 66.5, "Chen T6", ""),
    ("Qwen2.5-VL-FAST", 86.5, 2.0, "Chen T6", ""),
    ("Qwen2.5-VL-GR00T", 95.5, 62.5, "Chen T6", ""),
    ("Qwen2.5-VL-π", 88.0, 52.5, "Chen T6", ""),
]
fig, ax = plt.subplots(figsize=(15, 6.4))
y = np.arange(len(ROWS))[::-1]
for yi, (lab, en, em, src, note) in zip(y, ROWS):
    ax.plot([em, en], [yi, yi], color="#bbb", lw=3, solid_capstyle="round", zorder=1)
    ax.scatter(en, yi, s=80, color=GD, zorder=3, edgecolor="white", lw=1)
    ax.scatter(em, yi, s=80, color=RD, zorder=3, edgecolor="white", lw=1)
    ax.text(en + 1.2, yi, f"{en:.1f}", va="center", fontsize=9, color=GD)
    ax.text(em - 1.2, yi, f"{em:.1f}", va="center", ha="right", fontsize=9, color=RD)
    ax.text(103, yi, f"Δ −{en - em:.0f}  ·  {src}" + (f"  ({note})" if note else ""), va="center", fontsize=8.5, color="#555")
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in ROWS])
ax.set_xlim(-2, 135); ax.set_xlabel("LIBERO-Spatial success rate (%)")
ax.axvspan(29.8, 68.2, color=R, alpha=0.25, lw=0)
ax.set_ylim(-1.6, len(ROWS) - 0.4)
ax.text(49, -1.1, "boş komutla Spatial'da 'görüntüden çözülebilen' pay: çoğu modelde %30–68", ha="center", fontsize=9, color=RD)
ax.scatter([], [], s=80, color=GD, label="doğru İngilizce komut"); ax.scatter([], [], s=80, color=RD, label="boş komut (Ø) / dil girdisi yok")
ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.grid(axis="x", alpha=0.25)
ax.set_title("Ders 18 — literatürde 'boş komut' etkisi, LIBERO-Spatial (gerçek makale değerleri; Chen vd. 2606.15714 T6, Zhan vd. 2601.04052 T8–9, Fang vd. 2602.17659 T1)",
             fontsize=10.5)
fig.tight_layout(); fig.savefig(out + "02-empty-literature.png", dpi=160); plt.close(fig)

# ---------- 06 ----------
CF = {  # CF-Spatial, CAG'siz: (grounding faithful, grounding biased, success faithful, success biased)
    "OpenVLA-OFT": (6.8, 84.7, 1.1, 80.7),
    "π0": (70.9, 37.1, 34.4, 19.1),
    "π0.5": (39.3, 61.3, 24.4, 56.9),
}
fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
for ax, (title, i_f, i_b) in zip(axes, [("Grounding: gripper hangi nesneye dokundu?", 0, 1), ("Success: hangi görev tamamlandı?", 2, 3)]):
    x = np.arange(len(CF)); w = 0.36
    f = [v[i_f] for v in CF.values()]; b = [v[i_b] for v in CF.values()]
    ax.bar(x - w / 2, f, w, color=B, edgecolor=BD, label="Faithful — yanlış komutun hedefi (komutu okudu)")
    ax.bar(x + w / 2, b, w, color=R, edgecolor=RD, label="Biased — eğitimdeki orijinal görev (görüntüden gitti)")
    for xi, (fv, bv) in enumerate(zip(f, b)):
        ax.text(xi - w / 2, fv + 1.5, f"{fv:.1f}", ha="center", fontsize=9, color=BD)
        ax.text(xi + w / 2, bv + 1.5, f"{bv:.1f}", ha="center", fontsize=9, color=RD)
    ax.set_xticks(x); ax.set_xticklabels(list(CF)); ax.set_ylim(0, 100); ax.set_title(title, fontsize=11); ax.grid(axis="y", alpha=0.25)
axes[0].set_ylabel("%"); axes[0].legend(frameon=False, fontsize=9, loc="upper right")
fig.suptitle("Ders 18 — 'yanlış komut' verilince ne olur? LIBERO-CF, CF-Spatial (Fang vd. 2026, arXiv:2602.17659, Tablo III, CAG'siz)\n"
             "Bizim 'wrong' koşulunda success = Biased sütunu; Faithful'ı ancak videodan (hangi kâse?) sayabilirsin", fontsize=10.5)
fig.tight_layout(); fig.savefig(out + "06-wrong-literature.png", dpi=160); plt.close(fig)
print("02-empty-literature.png, 06-wrong-literature.png →", out)

# ---------- 07: olası örüntüler (TEMSİLİ — gerçek veri değil; Ders 19'un okuma anahtarı) ----------
PAT = [
    ("A) Dil görmezden geliniyor\n(saf shortcut)", [90, 85, 85, 85], "clean ≈ empty ≈ wrong ≈ tr\n→ Spatial görev 0 sahneden çözülüyor"),
    ("B) Dil okunuyor, Türkçe okunmuyor", [90, 40, 40, 40], "tr ≈ empty: Türkçe'den hiçbir şey çıkmıyor\n(Chen vd.: OFT/π0.5 örüntüsü)"),
    ("C) Yanlış komut 'biased'", [90, 40, 80, 40], "wrong ≈ clean ama empty düşük:\nkomut 'bir kâse al' diyor, sahne hangisini seçiyor"),
    ("D) Çok dilli omurga işe yarıyor", [90, 40, 40, 75], "tr > empty: Qwen3-VL Türkçe'den\nen azından nesne/eylemi çıkarıyor (hipotez)"),
]
fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), sharey=True)
cols = [GD, RD, RD, YD]
for ax, (title, vals, note) in zip(axes, PAT):
    ax.bar(["clean", "empty", "wrong", "tr"], vals, color=[G, R, R, Y], edgecolor=cols, lw=1.2)
    ax.set_ylim(0, 128); ax.set_yticks([0, 20, 40, 60, 80, 100]); ax.set_title(title, fontsize=10.5); ax.grid(axis="y", alpha=0.25)
    ax.text(1.5, 124, note, ha="center", va="top", fontsize=8.5, color="#444")
axes[0].set_ylabel("success rate (%) — TEMSİLİ")
fig.suptitle("Ders 18 — 4 satırlık tablonun olası dört örüntüsü (TEMSİLİ değerler; gerçek sayılar senin koşundan gelecek, yorum Ders 19)", fontsize=11)
fig.tight_layout(); fig.savefig(out + "07-expected-patterns.png", dpi=160); plt.close(fig)
print("07-expected-patterns.png →", out)
