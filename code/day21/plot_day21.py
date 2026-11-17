"""Ders 21 — görsel 05: tez ortamı karar matrisi (LIBERO+MuJoCo vs Webots).

Puanlar ÖZNEL (Claude'un bu dersteki değerlendirmesi, 0–3); kaynak değil, tartışma başlangıcıdır.
Danışmanla konuştuktan sonra puanları değiştirip yeniden çiz:  python plot_day21.py --out 05-decision.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CRITERIA = [
    ("Hazır benchmark + liderlik tablosu (Spatial/Goal, Plus, CF, X)", 3, 0),
    ("VLA eval kodu hazır (OpenVLA, VLANeXt, OFT…)",                   3, 0),
    ("Eğitim verisi (50 demo/görev, HDF5, RLDS)",                      3, 0),
    ("Mac M2'de kurulum / çalışma kolaylığı",                          1, 3),
    ("Oda ölçeği, mobil taban (RoomSpace tarzı sahne)",                 0, 3),
    ("Sahne/PROTO üretimi, kamera yerleşimi esnekliği",                 1, 3),
    ("ROS 2 köprüsü (gerçek robota geçiş)",                             0, 3),
    ("Türkçe komut deneyine süre: 'bugün' başlanabilir mi",            3, 1),
    ("Sonuçların literatürle karşılaştırılabilirliği",                  3, 0),
    ("Aynı robot (Panda) iki ortamda da var",                           3, 2),
]
LABELS = ["LIBERO + robosuite + MuJoCo", "Webots R2025a"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="05-decision.png")
    a = ap.parse_args()
    M = np.array([[c[1], c[2]] for c in CRITERIA], float)
    fig, ax = plt.subplots(figsize=(11, 6.8))
    cmap = matplotlib.colors.ListedColormap(["#ffd6d6", "#fde9c4", "#d4f5d4", "#9be29b"])
    ax.imshow(M, cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto")
    for i, (name, s1, s2) in enumerate(CRITERIA):
        for j, s in enumerate((s1, s2)):
            ax.text(j, i, ["yok", "zayıf", "orta", "güçlü"][s], ha="center", va="center", fontsize=11, weight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(LABELS, fontsize=11)
    ax.set_yticks(range(len(CRITERIA))); ax.set_yticklabels([c[0] for c in CRITERIA], fontsize=9.5)
    ax.xaxis.tick_top()
    tot = M.sum(axis=0)
    ax.set_xlabel(f"toplam (eşit ağırlık): {tot[0]:.0f} / {tot[1]:.0f}   —   tez sorusu benchmark ister, Webots bir benchmark değil",
                  fontsize=10, labelpad=10)
    ax.set_title("Ders 21 — Tez ortamı karar matrisi (ÖZNEL puanlama 0–3; danışmanla güncellenecek)", fontsize=12, pad=38)
    for i in range(len(CRITERIA) + 1):
        ax.axhline(i - 0.5, color="white", lw=2)
    ax.axvline(0.5, color="white", lw=2)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("kaydedildi:", a.out)


if __name__ == "__main__":
    main()
