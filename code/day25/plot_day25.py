"""Ders 25 — görseller 02, 03, 05, 06. Sayılar: Faz 3 derslerinin kendi çıktıları (başlıkta ders numarası),
makale tabloları (başlıkta arXiv) ve bu klasördeki results/smoke.csv (gerçek koşu). Temsili veri yok.

python plot_day25.py            # lectures/images/lecture-25/ altına yazar
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as _beta, norm

HERE = Path(__file__).resolve().parent
OUT = HERE / ".." / ".." / "lectures" / "images" / "lecture-25"
OUT.mkdir(parents=True, exist_ok=True)
Y, B, G, R = "#ffd166", "#cfe8ff", "#d4f5d4", "#ffd6d6"       # sarı model/anahtar, mavi girdi, yeşil çıktı, kırmızı sorun
YE, BE, GE, RE = "#b8860b", "#3b6fa0", "#2e8b57", "#c0392b"
GRAY = "#9a9a9a"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


# ---------------------------------------------------------------- 02: ölçüm durumu matrisi (Faz 3 bilançosu)
# Durumlar: 2 = bu ortamda gerçek çıktıyla ölçüldü, 1 = script hazır, koşu Mesut'ta, 0 = hipotez / ölçülmedi
ROWS = [
    # (satır, ders, gerçek, Mesut'ta, hipotez)  — Faz 3 günlük girişlerinden
    ("LIBERO-Plus bozulma sayımları (376 kamera, 500 Panda, 300 _add_)", 17, 2),
    ("Language 50 yeniden yazım: R1 26 / R2 22 / R0 2, 'between' 48/50", 17, 2),
    ("Türkçe _language_trK BDDL üretimi (bddl 1.0.1, 10/10)", 17, 2),
    ("Dört koşul × 10 init state success tablosu (T4)", 18, 1),
    ("Wilson / kesin güç hesapları (n=10, 20, 50)", 18, 2),
    ("Komutsuz üst sınır (BDDL): 9/10 · 10/10 · 1/10 · 9/10", 19, 2),
    ("50 init state geometrisi (.pruned_init): yayılım ≤ 3 cm", 19, 2),
    ("Örüntü A/B/C/D okuması — gerçek tabloyla", 19, 1),
    ("datasets/04: 20 çift × 5 yüzey, 90 referans hücresi, 5 CF BDDL", 20, 2),
    ("Anadil kontrolü: 'soldaki' hangi kâse, tr_a/tr_b doğallık", 20, 1),
    ("Eğitim görüntülerinin de 180° döndürülmüş olduğu", 20, 0),
    ("Webots: kurulum dokümanı, FK/IK, kuru koşu controller", 21, 2),
    ("Webots gerçek render (launcher M2'de çalışıyor mu)", 21, 1),
    ("Tokenizasyon: 100 komut × 2 tokenizer (şişme, ek, action aralığı)", 22, 2),
    ("Qwen3-VL tokenizer = npm Qwen3 tokenizer.json (bayt aynılığı)", 22, 1),
    ("PI formülü ve genişlik simülasyonu", 23, 2),
    ("VLANeXt'in koşudan koşuya değişkenliği (seed/flow-matching)", 23, 0),
    ("Parser: set içi 79/80, held-out 5/12", 24, 2),
    ("MT round-trip (Opus-MT): birebir eşleşme, gecikme", 24, 1),
    ("QLoRA T4 bellek 7,7 GB / 65–98 saat", 24, 0),
    ("Türkçe'nin Qwen3-VL'de yeri (Çince tarafı mı, Arapça tarafı mı)", 24, 0),
]


def fig02():
    fig, ax = plt.subplots(figsize=(13, 8.2))
    cols = ["Bu ortamda gerçek çıktı", "Script hazır, koşu Mesut'ta", "Hipotez / ölçülmedi"]
    colmap = {2: 0, 1: 1, 0: 2}
    colors = {0: G, 1: Y, 2: R}
    for i, (label, day, st) in enumerate(ROWS):
        y = len(ROWS) - 1 - i
        for j in range(3):
            ax.add_patch(plt.Rectangle((j, y), 1, 1, facecolor=colors[j] if colmap[st] == j else "#f7f7f7",
                                       edgecolor="white", linewidth=2))
        if colmap[st] == 0:
            ax.text(0.5, y + 0.5, "✓", ha="center", va="center", fontsize=12, color=GE, fontweight="bold")
        elif colmap[st] == 1:
            ax.text(1.5, y + 0.5, "→", ha="center", va="center", fontsize=12, color=YE, fontweight="bold")
        else:
            ax.text(2.5, y + 0.5, "?", ha="center", va="center", fontsize=12, color=RE, fontweight="bold")
        ax.text(-0.08, y + 0.5, f"D{day}  {label}", ha="right", va="center", fontsize=8.6)
    ax.set_xlim(0, 3); ax.set_ylim(0, len(ROWS))
    ax.set_xticks([0.5, 1.5, 2.5]); ax.set_xticklabels(cols, fontsize=9.5)
    ax.xaxis.tick_top(); ax.set_yticks([])
    for s in ("left", "bottom"): ax.spines[s].set_visible(False)
    n2 = sum(1 for r in ROWS if r[2] == 2); n1 = sum(1 for r in ROWS if r[2] == 1); n0 = sum(1 for r in ROWS if r[2] == 0)
    ax.set_title(f"Faz 3 bilançosu: {len(ROWS)} ölçüm — {n2} gerçek, {n1} Mesut'un koşusunda, {n0} hipotez\n"
                 "Kaynak: plan/daily-log.md Ders 17–24 girişleri; 'gerçek' = bu oturumda phase3_smoke.py ile yeniden üretildi (28/30 script)",
                 fontsize=10.5, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "02-measured-vs-pending.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- 03: Faz 3'ün sayıları (dört küçük panel)
def fig03():
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    # (a) komutsuz üst sınır vs literatür boş-komut (Ders 19 §2–3)
    a = axes[0, 0]
    suites = ["Spatial", "Object", "Goal", "Long"]
    bound = [90, 100, 10, 90]
    a.bar(range(4), bound, color=B, edgecolor=BE, width=0.55, label="BDDL üst sınırı (suite_structure.py)")
    lit = {  # Ders 19 görsel 02: Chen Ø (T6), Zhan blank (T9), Fang vision-only (T1)
        "OpenVLA-OFT": ([80.0, 91.5, 10.0, 76.5], [83.2, 98.2, 10.0, 85.0]),
        "π0.5": ([66.0, 64.5, 9.0, 70.0], [65.0, 51.8, 11.4, 72.0], [68.2, 56.6, 10.2, 71.6]),
        "Qwen3-VL + OFT": ([34.0, 42.5, 8.5, 45.0],),
    }
    mk = {"OpenVLA-OFT": ("o", YE), "π0.5": ("s", GE), "Qwen3-VL + OFT": ("^", RE)}
    for name, series in lit.items():
        arr = np.array(series)
        lo, hi, md = arr.min(0), arr.max(0), arr.mean(0)
        x = np.arange(4) + {"OpenVLA-OFT": -0.12, "π0.5": 0.0, "Qwen3-VL + OFT": 0.12}[name]
        a.errorbar(x, md, yerr=[md - lo, hi - md], fmt=mk[name][0], color=mk[name][1], ms=6, capsize=3, lw=1.2, label=f"{name} boş komut")
    a.set_xticks(range(4)); a.set_xticklabels(suites); a.set_ylim(0, 105); a.set_ylabel("success rate (%)")
    a.legend(fontsize=7.5, loc="center", bbox_to_anchor=(0.66, 0.50), framealpha=0.95)
    a.set_title("(a) Komutsuz tavan sabit, kullanımı modele göre — Ders 19\nChen 2606.15714 T6 · Zhan 2601.04052 T9 · Fang 2602.17659 T1", fontsize=9.5)
    # (b) tokenizasyon (Ders 22, gerçek)
    b = axes[0, 1]
    surf = ["en", "tr_a", "tr_ascii", "tr_bare"]
    L = [13.6, 22.4, 17.6, 14.9]; Q = [12.3, 16.9, 16.0, 12.2]
    x = np.arange(4); w = 0.36
    b.bar(x - w / 2, L, w, color=Y, edgecolor=YE, label="LLaMA-2 (OpenVLA)")
    b.bar(x + w / 2, Q, w, color=B, edgecolor=BE, label="Qwen3 (VLANeXt)")
    for i in range(4):
        b.text(x[i] - w / 2, L[i] + 0.3, f"{L[i]:.1f}", ha="center", fontsize=8)
        b.text(x[i] + w / 2, Q[i] + 0.3, f"{Q[i]:.1f}", ha="center", fontsize=8)
    b.set_xticks(x); b.set_xticklabels(surf); b.set_ylabel("token / komut"); b.set_ylim(0, 26)
    b.legend(fontsize=8, loc="upper right")
    b.set_title("(b) Aynı 100 komut, iki tokenizer — Ders 22 (tokens.csv, bu oturumda birebir yeniden üretildi)\n"
                "tr_a/en şişme 1,63× vs 1,41×; Türkçe harf bedeli (tr_a − tr_ascii) 4,9 vs 0,9 token", fontsize=9.5)
    # (c) Chen Tablo 1: LLaMA-2 OFT vs Qwen3-VL OFT vs Qwen3-VL GR00T
    c = axes[1, 0]
    langs = ["zh", "fr", "ru", "ar", "∅"]
    rowsC = {"OpenVLA-OFT (LLaMA-2)": ([-28.6, -28.6, -27.8, -29.8, -27.5], Y, YE),
             "Qwen3-VL + OFT başı": ([-7.3, -39.1, -48.8, -47.1, -63.1], R, RE),
             "Qwen3-VL + GR00T başı": ([-6.1, -33.0, -38.8, -36.0, -48.0], B, BE)}
    x = np.arange(5); w = 0.26
    for k, (name, (v, col, ed)) in enumerate(rowsC.items()):
        c.bar(x + (k - 1) * w, v, w, color=col, edgecolor=ed, label=name)
    c.axhline(0, color="#333", lw=0.8); c.set_xticks(x); c.set_xticklabels(langs)
    c.set_ylabel("EN'e göre fark (puan)"); c.legend(fontsize=8, loc="lower left")
    c.set_title("(c) 'Çok dilli omurga' yalnız Çince'de iyi — Ders 24\nChen vd. 2606.15714 Tablo 1, LIBERO ort.; Türkçe ölçülmedi (S1 bunu ölçer)", fontsize=9.5)
    # (d) Wilson aralıkları n=10 vs n=50 (Ders 18)
    d = axes[1, 1]
    cases = [("9/10", 9, 10), ("5/10", 5, 10), ("45/50", 45, 50), ("25/50", 25, 50)]
    for i, (lab, k, n) in enumerate(cases):
        lo, hi = wilson(k, n)
        col = Y if n == 10 else G; ed = YE if n == 10 else GE
        d.barh(i, (hi - lo) * 100, left=lo * 100, color=col, edgecolor=ed, height=0.5)
        d.plot([k / n * 100], [i], "o", color="#333", ms=5)
        d.text(hi * 100 + 1, i, f"{lo*100:.0f}–{hi*100:.0f}", va="center", fontsize=8.5)
    d.set_yticks(range(4)); d.set_yticklabels([c[0] for c in cases]); d.set_xlim(0, 115)
    d.set_xlabel("success rate (%) — Wilson %95")
    d.set_title("(d) 10 episode yön söyler, 50 episode fark söyler — Ders 18 §5\n9/10 ve 5/10 aralıkları çakışır; 45/50 ve 25/50 ayrışır", fontsize=9.5)
    fig.suptitle("Faz 3'ün sayıları — dördü de gerçek (ders numarası ve kaynak panel başlığında)", fontsize=11.5)
    fig.tight_layout(); fig.savefig(OUT / "03-phase3-numbers.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- 05: kuru koşu sonuçları (results/smoke.csv, gerçek)
def fig05():
    rows = list(csv.DictReader(open(HERE / "results" / "smoke.csv", encoding="utf-8")))
    days = sorted({int(r["ders"]) for r in rows})
    ok = [sum(1 for r in rows if int(r["ders"]) == d and r["sonuc"] == "OK") for d in days]
    er = [sum(1 for r in rows if int(r["ders"]) == d and r["sonuc"] == "HATA") for d in days]
    sk = [sum(1 for r in rows if int(r["ders"]) == d and r["sonuc"] == "ATLANDI") for d in days]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(days))
    ax.bar(x, ok, color=G, edgecolor=GE, label="OK (çalıştı, çıkış 0)")
    ax.bar(x, er, bottom=ok, color=R, edgecolor=RE, label="HATA")
    ax.bar(x, sk, bottom=np.array(ok) + np.array(er), color="#eeeeee", edgecolor=GRAY, label="ATLANDI (ikili/model yok)")
    for i, d in enumerate(days):
        names = [r["script"].split()[0] for r in rows if int(r["ders"]) == d and r["sonuc"] == "ATLANDI"]
        if names:
            ax.text(x[i], ok[i] + er[i] + sk[i] + 0.15, "\n".join(names), ha="center", va="bottom", fontsize=7.5, color="#555")
        ax.text(x[i], 0.2, f"{ok[i]}/{ok[i]+er[i]+sk[i]}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"Ders {d}" for d in days]); ax.set_ylabel("script / kontrol sayısı")
    ax.set_ylim(0, max(np.array(ok) + np.array(er) + np.array(sk)) + 2.2)
    ax.legend(fontsize=8.5, loc="upper right")
    t_ok, t_all = sum(ok), len(rows)
    ax.set_title(f"Faz 3 araç envanteri kuru koşuda: {t_ok}/{t_all} OK — phase3_smoke.py, bu oturum (model yok, GPU yok)\n"
                 "Girdiler: LIBERO + LIBERO-plus klonları (--depth 1), Meta tokenizer.model (sha256 OK), npm Qwen3 tokenizer.json, bddl 1.0.1",
                 fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "05-smoke.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- 06: LangGap (Hou & Zhao 2026, arXiv:2603.00592) — π0.5, aynı sahne, yalnızca komut değişir
def fig06():
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8))
    suites = ["Spatial", "Goal", "Object"]
    orig = [97.0, 97.0, 100.0]; ext = [5.9, 30.0, 37.7]           # Tablo VII
    x = np.arange(3); w = 0.36
    a.bar(x - w / 2, orig, w, color=G, edgecolor=GE, label="Orijinal 40 görev")
    a.bar(x + w / 2, ext, w, color=R, edgecolor=RE, label="Genişletilmiş 59 görev (aynı sahne, komut değişti)")
    for i in range(3):
        a.text(x[i] - w / 2, orig[i] + 1, f"{orig[i]:.1f}", ha="center", fontsize=8.5)
        a.text(x[i] + w / 2, ext[i] + 1, f"{ext[i]:.1f}", ha="center", fontsize=8.5)
    a.set_xticks(x); a.set_xticklabels(suites); a.set_ylim(0, 125); a.set_ylabel("success rate (%) — π0.5")
    a.legend(fontsize=8, loc="upper center", ncol=2, framealpha=0.95)
    a.set_title("Tablo VII — suite başına: Spatial 97,0 → 5,9", fontsize=10)
    dims = ["Nesne\ndeğişti", "Hedef\ndeğişti", "Mekânsal\ntarif değişti", "Çekmece\neylemi"]
    vals = [29.3, 0.0, 11.0, 31.7]; ns = ["38 görev / 760 ep", "13 / 260", "5 / 100", "3 / 60"]
    cols = [B, R, Y, B]; eds = [BE, RE, YE, BE]
    b.bar(range(4), vals, color=cols, edgecolor=eds, width=0.6)
    for i in range(4):
        b.text(i, vals[i] + 1, f"{vals[i]:.1f}", ha="center", fontsize=9)
        b.text(i, -5.5, ns[i], ha="center", fontsize=7.5, color="#555")
    b.set_xticks(range(4)); b.set_xticklabels(dims, fontsize=8.5); b.set_ylim(-8, 45); b.axhline(0, color="#333", lw=0.8)
    b.set_ylabel("success rate (%) — π0.5, genişletilmiş görevler")
    b.set_title("Tablo VIII — hangi bileşen değişince düşüyor; 'Mekânsal tarif' 5 görev × 20 episode", fontsize=10)
    fig.suptitle("LangGap (Hou & Zhao, Şub 2026, arXiv:2603.00592): aynı sahnede komutun bir bileşeni değişince π0.5 93,8 → 21,4\n"
                 "Yalnızca İngilizce, Türkçe yok; 'Spatial Description' hücresi datasets/04 B bloğunun İngilizce karşılığı", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "06-langgap.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    fig02(); fig03(); fig05(); fig06()
    print("yazıldı:", OUT.resolve())
