"""Ders 23 görselleri — Cohn hattındaki yedi makaleden GERÇEK tablo değerleri + datasets/04'ün ilişki etiketleri + bir simülasyon.

  python plot_day23.py --pairs ../../datasets/04-tr-spatial-commands/data/pairs.csv --out ../../lectures/images/lecture-23

02-depth-collapse.png  hop/derinlik arttıkça çöküş: StepGame düzeltilmiş (Li, Hogg, Cohn AAAI-24, Tablo 4) ve DecompSR (McPheat vd. 2025, Tablo 6)
03-surfaces.png        aynı soru, farklı yüzey: DecompSR dil (Tablo 2) + nonce (Tablo 1); kardinal yön 2024→2025 (Cohn & Blackwell) + T4 şablonu
04-relation-map.png    datasets/04'teki 20 çiftin ilişki ailesi × Cohn benchmark'larının kapsadığı aileler (pairs.csv `relation` sütunundan)
05-pi-width.png        TEMSİLİ: Blackwell tahmin aralığının genişliği, tekrar sayısı × episode sayısı (binom simülasyonu; prediction_interval.py)
Renk dili: sarı #ffd166 model/anahtar, mavi #cfe8ff girdi, yeşil #d4f5d4 çıktı, kırmızı #ffd6d6 sorun.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from prediction_interval import simulate_widths

YEL, BLU, GRN, RED = "#ffd166", "#cfe8ff", "#d4f5d4", "#ffd6d6"
YEL_D, BLU_D, GRN_D, RED_D, GREY = "#c99a1e", "#4a7fb5", "#3f9d5a", "#c0504d", "#777777"

# ---- Li, Hogg, Cohn (AAAI-24) arXiv:2401.03991, Tablo 4 — düzeltilmiş StepGame, GPT-4 (%; "/" = raporlanmadı) ----
K10 = list(range(1, 11))
GPT4_BASE = [100, 70, 55, 45, 40, 25, 40, 35, 35, 25]
GPT4_COT = [None, 80, 75, 95, 85, 85, 90, 80, 60, 65]
GPT4_TOT = [None, None, 85, 85, 90, 90, 85, 90, 100, 95]
TURBO_BASE = [62, 43, 30, 35, 29, 25, 29, 31, 16, 20]

# ---- McPheat vd. 2025 arXiv:2511.02627 (DecompSR), Tablo 6 — 5-shot, doğruluk 0–1 ----
KD = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]
DECOMP = {
    "GPT-3.5":          [0.73, 0.20, 0.16, 0.12, 0.12, 0.15, 0.14, 0.12, 0.11, 0.12, 0.15, 0.12, 0.12],
    "GPT-4o":           [0.99, 0.60, 0.35, 0.29, 0.29, 0.27, 0.20, 0.22, 0.19, 0.21, 0.15, 0.18, 0.15],
    "Claude Sonnet 3.7":[0.98, 0.77, 0.53, 0.41, 0.37, 0.26, 0.22, 0.22, 0.20, 0.20, 0.12, 0.10, 0.09],
    "DeepSeek-R1":      [0.99, 0.67, 0.67, 0.65, 0.62, 0.61, 0.54, 0.48, 0.37, 0.34, 0.14, 0.17, 0.14],
    "o4-mini":          [0.99, 0.83, 0.75, 0.70, 0.65, 0.62, 0.59, 0.58, 0.52, 0.52, 0.36, 0.17, 0.07],
}
# Tablo 2 (diller, 5-shot) ve Tablo 1 (nonce) — k = 1, 2, 5, 10
KL = [1, 2, 5, 10]
LANG = {
    "GPT-4o":  {"İngilizce": [0.99, 0.60, 0.29, 0.21], "Hintçe": [0.96, 0.47, 0.22, 0.18], "İsveççe": [0.82, 0.43, 0.26, 0.20]},
    "o4-mini": {"İngilizce": [0.99, 0.83, 0.65, 0.52], "Hintçe": [0.95, 0.73, 0.63, 0.48], "İsveççe": [0.30, 0.23, 0.09, 0.07]},
}
NONCE = {"GPT-4o": ([0.99, 0.60, 0.29, 0.21], [0.37, 0.14, 0.14, 0.10]),
         "o4-mini": ([0.99, 0.83, 0.65, 0.52], [0.51, 0.38, 0.34, 0.26])}

# ---- Cohn & Blackwell: COSIT-24 arXiv:2406.16528 ve "Revisited" arXiv:2507.12059 — `large` (5.760 soru), metinde verilen değerler ----
CARD = [("rastgele\n8 yön", 0.125, GREY), ("OLMo 2\n'25 en kötü", 0.16, RED_D), ("gemini-1.0\n'24 en kötü", 0.25, RED_D),
        ("Mistral 24B\n'25 yerel", 0.41, BLU_D), ("gpt-3.5-t.\n'24 en iyi", 0.595, YEL_D), ("o1\n'25 en iyi", 0.92, YEL_D)]
T4 = [("gpt-3.5\n'24", 0.41), ("o1\n'25", 0.55), ("GPT-4.5\n'25", 0.0)]

# ---- datasets/04 `relation` → aile; Cohn benchmark'larının kapsadığı aileler ----
FAMILIES = ["yön (CDC)", "topoloji (RCC)", "mesafe", "bölge / 2 referans"]
BENCH_COVER = {  # kaynak: ilgili makalenin ilişki tanımı (ders §2 tablosu)
    "StepGame / DecompSR\n(8 yön, grid)":          ["yön (CDC)"],
    "RoomSpace\n(9 yön, TPP/NTPP, yakın/uzak)":    ["yön (CDC)", "topoloji (RCC)", "mesafe", "bölge / 2 referans"],
    "Kardinal yön\n(8 yön, metin)":                ["yön (CDC)"],
    "QSTRBench\n(CDC, RCC-5/8/22, 9IM, STAR)":     ["yön (CDC)", "topoloji (RCC)"],
}


def family(rel):
    r = rel.lower()
    if r.startswith("yön"):
        return "yön (CDC)"
    if r.startswith(("ec", "ntpp", "tpp")):
        return "topoloji (RCC)"
    if r.startswith("mesafe"):
        return "mesafe"
    return "bölge / 2 referans"


def fig02(out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    def clean(ys):
        return [(k, y) for k, y in zip(K10, ys) if y is not None]
    for ys, col, lab, ls in ((GPT4_BASE, RED_D, "GPT-4, düz prompt", "-"), (GPT4_COT, BLU_D, "GPT-4 + CoT", "--"),
                             (GPT4_TOT, GRN_D, "GPT-4 + ToT(CoT)", "-."), (TURBO_BASE, GREY, "GPT-3.5 Turbo, düz", ":")):
        pts = clean(ys)
        a1.plot([p[0] for p in pts], [p[1] for p in pts], ls, color=col, marker="o", lw=2, label=lab)
    a1.set_xticks(K10); a1.set_ylim(0, 105); a1.set_xlabel("k (hop sayısı)"); a1.set_ylabel("doğruluk (%)")
    a1.set_title("Düzeltilmiş StepGame — Li, Hogg, Cohn (AAAI-24), Tablo 4", fontsize=11)
    a1.grid(alpha=0.3); a1.legend(fontsize=9, loc="lower left")
    a1.annotate("100 → 25:\nmodel okuyor,\nzinciri taşıyamıyor", xy=(10, 25), xytext=(6.2, 8), fontsize=9,
                arrowprops=dict(arrowstyle="->", color=RED_D), color=RED_D)
    cols = {"GPT-3.5": GREY, "GPT-4o": RED_D, "Claude Sonnet 3.7": YEL_D, "DeepSeek-R1": BLU_D, "o4-mini": GRN_D}
    for name, ys in DECOMP.items():
        a2.plot(KD, ys, marker="o", lw=2, color=cols[name], label=name)
    a2.axhline(0.125, color=GREY, ls=":", lw=1); a2.text(1.05, 0.14, "1/8 = tahmin (çıkarım)", fontsize=8, color=GREY)
    a2.set_xscale("log"); a2.set_xticks(KD); a2.set_xticklabels([str(k) for k in KD], fontsize=8)
    a2.set_ylim(0, 1.05); a2.set_xlabel("k (derinlik, log)"); a2.set_ylabel("doğruluk (0–1)")
    a2.set_title("DecompSR — McPheat vd. (2025), Tablo 6 (5-shot)", fontsize=11)
    a2.grid(alpha=0.3, which="both"); a2.legend(fontsize=9, loc="upper right")
    a2.annotate("k=100: herkes\ntahmin düzeyinde", xy=(100, 0.12), xytext=(22, 0.45), fontsize=9,
                arrowprops=dict(arrowstyle="->", color=RED_D), color=RED_D)
    fig.suptitle("Derinlik arttıkça çöküş: 2024'te de 2025'te de aynı eğri (gerçek makale değerleri)", fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def fig03(out):
    fig = plt.figure(figsize=(14, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.9, 1.15])
    a1, a2, a3 = (fig.add_subplot(gs[0, i]) for i in range(3))
    # (a) diller
    w = 0.13; xs = list(range(len(KL)))
    lang_cols = {"İngilizce": YEL_D, "Hintçe": BLU_D, "İsveççe": RED_D}
    for mi, (model, d) in enumerate(LANG.items()):
        for li, (lang, ys) in enumerate(d.items()):
            off = (mi * 3 + li - 2.5) * w
            a1.bar([x + off for x in xs], ys, w, color=lang_cols[lang], alpha=1.0 if mi == 0 else 0.55,
                   edgecolor="black" if mi == 1 else "none", lw=0.6)
    a1.set_xticks(xs); a1.set_xticklabels([f"k={k}" for k in KL]); a1.set_ylim(0, 1.05)
    a1.set_title("DecompSR Tablo 2 — aynı sorular, üç dil\n(dolu: GPT-4o · çerçeveli: o4-mini)", fontsize=10)
    a1.legend(handles=[Patch(color=c, label=l) for l, c in lang_cols.items()], fontsize=9, loc="upper right")
    a1.set_ylabel("doğruluk"); a1.grid(axis="y", alpha=0.3)
    a1.annotate("o4-mini İsveççe k=1: 0,30\n(İngilizce 0,99)", xy=(0 + (3 + 2 - 2.5) * w, 0.30), xytext=(0.9, 0.78), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color=RED_D), color=RED_D)
    # (b) nonce
    for mi, (model, (en, nn)) in enumerate(NONCE.items()):
        c = YEL_D if mi == 0 else GRN_D
        a2.plot(KL, en, "-o", color=c, lw=2, label=f"{model} İngilizce")
        a2.plot(KL, nn, "--s", color=c, lw=2, label=f"{model} nonce yön adları")
    a2.set_xticks(KL); a2.set_ylim(0, 1.05); a2.set_xlabel("k"); a2.grid(alpha=0.3)
    a2.set_title("DecompSR Tablo 1 — yön sözcükleri\nuydurma kelimelerle değiştirilince", fontsize=10)
    a2.legend(fontsize=8, loc="upper right")
    # (c) kardinal
    xs3 = list(range(len(CARD)))
    a3.bar(xs3, [c[1] for c in CARD], color=[c[2] for c in CARD], edgecolor="black", lw=0.5)
    for x, (lab, v, _) in zip(xs3, CARD):
        a3.text(x, v + 0.02, f"{v:.2f}" if v != 0.125 else "0,125", ha="center", fontsize=9)
    a3.set_xticks(xs3); a3.set_xticklabels([c[0] for c in CARD], fontsize=7.5)
    a3.set_ylim(0, 1.08); a3.set_title("Kardinal yön benchmark'ı (`large`, 5.760 soru)\nCohn & Blackwell 2024 → 2025", fontsize=10)
    a3.grid(axis="y", alpha=0.3)
    ins = a3.inset_axes([0.05, 0.50, 0.40, 0.42])
    ins.bar(range(3), [t[1] for t in T4], color=[BLU_D, GRN_D, RED_D], edgecolor="black", lw=0.5)
    ins.set_xticks(range(3)); ins.set_xticklabels([t[0] for t in T4], fontsize=6.5); ins.set_ylim(0, 0.7)
    ins.set_title("T4 'yol' şablonu", fontsize=8); ins.tick_params(axis="y", labelsize=7)
    for i, t in enumerate(T4):
        ins.text(i, t[1] + 0.02, f"{t[1]:.2f}", ha="center", fontsize=7)
    fig.suptitle("Aynı soru, farklı yüzey: dil, sözcük, şablon değişince doğruluk değişiyor (gerçek makale değerleri)", fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def fig04(pairs_csv, out):
    with open(pairs_csv, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    fams = [family(r["relation"]) for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.6, 1]})
    fam_col = {"yön (CDC)": YEL, "topoloji (RCC)": BLU, "mesafe": GRN, "bölge / 2 referans": RED}
    # sol: 20 çift × aile ızgarası
    for i, (r, fm) in enumerate(zip(rows, fams)):
        j = FAMILIES.index(fm)
        a1.add_patch(plt.Rectangle((j, i), 1, 1, color=fam_col[fm], ec="white"))
        a1.text(j + 0.5, i + 0.5, r["rel_span"], ha="center", va="center", fontsize=8)
        a1.text(-0.1, i + 0.5, f"{r['pair_id']} ({r['block']})", ha="right", va="center", fontsize=8,
                color=RED_D if r["frame_dependent"].strip() == "evet" else "black")
    a1.set_xlim(-2.2, len(FAMILIES)); a1.set_ylim(len(rows), 0)
    a1.set_xticks([j + 0.5 for j in range(len(FAMILIES))]); a1.set_xticklabels(FAMILIES, fontsize=9)
    a1.set_yticks([]); a1.xaxis.tick_top()
    for sp in a1.spines.values():
        sp.set_visible(False)
    a1.set_title("datasets/04 — 20 çiftin ilişki ailesi (`relation` sütunu; kırmızı id = çerçeveye bağlı)", fontsize=10, pad=28)
    counts = {fm: fams.count(fm) for fm in FAMILIES}
    a1.text(len(FAMILIES) / 2, len(rows) + 0.9, "   ".join(f"{k}: {v}" for k, v in counts.items()), ha="center", fontsize=9)
    # sağ: benchmark × aile kapsama
    for bi, (b, cov) in enumerate(BENCH_COVER.items()):
        for j, fm in enumerate(FAMILIES):
            on = fm in cov
            a2.add_patch(plt.Rectangle((j, bi), 1, 1, color=fam_col[fm] if on else "#f2f2f2", ec="white"))
            a2.text(j + 0.5, bi + 0.5, "✓" if on else "–", ha="center", va="center", fontsize=12, color="black" if on else "#aaaaaa")
        a2.text(-0.1, bi + 0.5, b, ha="right", va="center", fontsize=8.5)
    a2.set_xlim(-3.4, len(FAMILIES)); a2.set_ylim(len(BENCH_COVER), 0)
    a2.set_xticks([j + 0.5 for j in range(len(FAMILIES))]); a2.set_xticklabels([f.replace(" (", "\n(").replace(" / ", " /\n") for f in FAMILIES], fontsize=8.5)
    a2.set_yticks([]); a2.xaxis.tick_top()
    for sp in a2.spines.values():
        sp.set_visible(False)
    a2.set_title("Cohn benchmark'ları hangi aileyi test ediyor?", fontsize=10, pad=28)
    a2.text(len(FAMILIES) / 2, len(BENCH_COVER) + 0.5, "LIBERO'da altın etiket simülatörden (BDDL predicate) gelir;\n"
            "RoomSpace'te ASP çözücüden, DecompSR'de yapım kuralından.", ha="center", fontsize=8.5, color=GREY)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def fig05(out):
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    cfg = [(0.7, 10, RED_D, "-"), (0.9, 10, RED_D, "--"), (0.7, 50, BLU_D, "-"), (0.9, 50, BLU_D, "--")]
    for p, ep, col, ls in cfg:
        w = simulate_widths(p, ep, 10, 1500, seed=int(p * 100) + ep)
        ns = sorted(w)
        ax.plot(ns, [w[n] for n in ns], ls, marker="o", color=col, lw=2, label=f"p={p}, {ep} episode/koşu")
    ax.axhline(0.01, color=GRN_D, ls=":", lw=2); ax.text(2.1, 0.012, "Blackwell eşiği 0,01 (metin benchmark'ı, 5.760 soru)", fontsize=8.5, color=GRN_D)
    ax.axhline(0.38, color=GREY, ls="-.", lw=1); ax.text(6.2, 0.40, "karş.: Ders 18 Wilson 9/10 → [0,60, 0,98] genişlik 0,38", fontsize=8.5, color=GREY)
    ax.set_yscale("log"); ax.set_ylim(0.007, 4); ax.set_xticks(range(2, 11))
    ax.set_xlabel("tekrar koşusu sayısı n"); ax.set_ylabel("PI95 genişliği (log)")
    ax.set_title("TEMSİLİ — tahmin aralığı genişliği, yalnızca binom gürültüsüyle (prediction_interval.py --simulate)\n"
                 "gerçek koşuda seed / flow-matching / env değişkenliği eklenir → gerçek genişlik bundan dar olamaz", fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="../../datasets/04-tr-spatial-commands/data/pairs.csv")
    ap.add_argument("--out", default="../../lectures/images/lecture-23")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig02(os.path.join(a.out, "02-depth-collapse.png"))
    fig03(os.path.join(a.out, "03-surfaces.png"))
    fig04(a.pairs, os.path.join(a.out, "04-relation-map.png"))
    fig05(os.path.join(a.out, "05-pi-width.png"))
    print("yazıldı:", a.out)


if __name__ == "__main__":
    main()
