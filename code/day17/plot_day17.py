"""Ders 17 — görsel 09: LIBERO-plus'ta 7 bozulma × model (VLANeXt makalesi Tablo III; sayılar elle aktarıldı).

Kullanım:  python plot_day17.py <çıktı_klasörü/>
Kaynak: Wu vd. 2026, VLANeXt, arXiv:2602.18532v2 Tablo III — LIBERO'da eğitilip LIBERO-plus'ta doğrudan test
(bozulma görülmemiş). Temiz LIBERO sayıları aynı makalenin LIBERO tablosundan (Ders 12 görsel 04'teki değerler).
"""
import os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out = (sys.argv[1] if len(sys.argv) > 1 else ".").rstrip("/") + "/"; os.makedirs(out, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
CATS = ["Camera", "Robot", "Language", "Light", "Background", "Noise", "Layout"]
T3 = {  # model, suite -> 7 kategori
    ("OpenVLA-OFT", "Spatial"): [88.3, 40.0, 80.5, 98.3, 97.3, 96.3, 93.9],
    ("OpenVLA-OFT", "Object"): [38.9, 25.4, 99.0, 73.7, 97.6, 72.3, 71.8],
    ("OpenVLA-OFT", "Goal"): [62.0, 25.2, 53.2, 93.9, 92.5, 75.2, 59.1],
    ("OpenVLA-OFT", "Long"): [38.7, 38.2, 87.0, 89.4, 86.8, 63.5, 76.9],
    ("VLANeXt-B", "Spatial"): [95.7, 78.6, 86.9, 99.7, 98.8, 98.0, 96.6],
    ("VLANeXt-B", "Object"): [99.5, 48.5, 98.6, 99.3, 84.7, 99.8, 78.2],
    ("VLANeXt-B", "Goal"): [96.6, 63.6, 51.5, 97.5, 70.8, 96.8, 63.9],
    ("VLANeXt-B", "Long"): [69.7, 72.0, 90.3, 86.9, 75.8, 81.7, 84.3],
}
CLEAN = {("OpenVLA-OFT", "Spatial"): 97.6, ("OpenVLA-OFT", "Goal"): 97.9, ("VLANeXt-B", "Spatial"): 99.0, ("VLANeXt-B", "Goal"): 96.6}

fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
for ax, suite in zip(axes, ["Spatial", "Goal"]):
    x = np.arange(len(CATS)); w = 0.38
    for i, (model, col) in enumerate([("OpenVLA-OFT", "#3a9a5c"), ("VLANeXt-B", "#e09a00")]):
        vals = T3[(model, suite)]
        bars = ax.bar(x + (i - 0.5) * w, vals, w * 0.94, color=col, label=f"{model} (temiz LIBERO {suite}: {CLEAN[(model, suite)]})", zorder=3)
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=8)
        ax.axhline(CLEAN[(model, suite)], color=col, lw=1, ls="--", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(CATS); ax.set_ylim(0, 108); ax.yaxis.grid(True, color="#e5e5e5", zorder=0)
    ax.set_title(f"LIBERO-plus {suite}: bozulma başına success rate (%)"); ax.legend(fontsize=8, loc="lower left")
    # Language sütununu vurgula
    ax.axvspan(1.5, 2.5, color="#ffd166", alpha=0.18, zorder=0)
axes[0].set_ylabel("Success rate (%)")
fig.suptitle("Hangi bozulma neyi kırıyor? OpenVLA-OFT vs VLANeXt-B, LIBERO'da eğitilip LIBERO-plus'ta test — kaynak: VLANeXt Tablo III (arXiv:2602.18532v2)", fontsize=10.5)
fig.text(0.5, 0.005, "Kesikli çizgi: aynı modelin temiz LIBERO skoru. Spatial'da Robot (40→79) ve Camera en çok kıran; Goal'da Language (53 / 52) iki modelde de en kötü — komut okunmak zorunda olan tek suite.",
         ha="center", fontsize=8.5, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 0.95)); fig.savefig(out + "09-vlanext-per-perturbation.png", dpi=170); plt.close(fig)
print("yazıldı:", out + "09-vlanext-per-perturbation.png")
