"""Ders 28 — görseller 02–04 (gerçek çıktılardan: results/ablation_surface.csv, results/budget_v01_v02.csv)."""
import csv, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "lectures", "images", "lecture-28")
os.makedirs(out, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
Y, B, G, R = "#ffd166", "#cfe8ff", "#d4f5d4", "#ffd6d6"
YD, BD, GD, RD = "#c79a00", "#3b6ea8", "#2e8b57", "#c0392b"
rows = list(csv.DictReader(open(os.path.join(HERE, "results", "ablation_surface.csv"), encoding="utf-8")))
V = ["tr_a", "tr_ascii", "tr_bare", "tr_bare_ascii"]
LAB = {"tr_a": "tr_a\nTR harf + ek", "tr_ascii": "tr_ascii\nASCII + ek", "tr_bare": "tr_bare\nTR harf, eksiz", "tr_bare_ascii": "tr_bare_ascii\nASCII, eksiz\n(YENİ)"}

# 02 — 2×2 token tablosu
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
for ax, tk, name in zip(axes, ("llama2", "qwen3"), ("LLaMA-2 tokenizer (S-OV)", "Qwen3 tokenizer (S-VN)")):
    m = [np.mean([int(r[f"{tk}_{v}"]) for r in rows]) for v in V]
    en = np.mean([int(r[f"{tk}_en"]) for r in rows])
    cols = [Y, B, G, "#e8e8e8"]; edges = [YD, BD, GD, "#666"]
    bars = ax.bar(range(4), m, color=cols, edgecolor=edges)
    for i, y in enumerate(m):
        ax.text(i, y + 0.3, f"{y:.1f}", ha="center", fontsize=9)
    ax.axhline(en, color="#555", ls="--", lw=1); ax.text(-0.45, en + 0.3, f"en = {en:.1f}", ha="left", fontsize=8.5, color="#555")
    letter = ((m[0] - m[1]) + (m[2] - m[3])) / 2; suffix = ((m[0] - m[2]) + (m[1] - m[3])) / 2; inter = (m[0] - m[1]) - (m[2] - m[3])
    ax.set_title(f"{name}\nharf etkisi {letter:+.1f} · ek etkisi {suffix:+.1f} · etkileşim {inter:+.1f} token", fontsize=10)
    ax.set_xticks(range(4)); ax.set_xticklabels([LAB[v] for v in V], fontsize=8.5)
    ax.grid(axis="y", alpha=0.25)
axes[0].set_ylabel("çift başına ortalama token (20 çift)")
fig.suptitle("Ders 28 — Yüzey ablation'ı bir 2×2: harf × ek. Eksik hücre tr_bare_ascii üretildi; token sayıları GERÇEK "
             "(Meta tokenizer.model + Qwen3 tokenizer.json; mevcut 4 yüzey datasets/05 ile 160/160 birebir)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(out, "02-surface-2x2-tokens.png"), dpi=160); plt.close(fig)

# 03 — ek adımı Türkçe harfleri de götürüyor
ids = [r["pair_id"] for r in rows]
keep = np.array([int(r["trletters_bare"]) for r in rows]); tot = np.array([int(r["trletters_tr_a"]) for r in rows])
lost = tot - keep
fig, ax = plt.subplots(figsize=(13, 4.8))
ax.bar(ids, keep, color=B, edgecolor=BD, label="kökte kalan Türkçe harf (tr_bare'de de var)")
ax.bar(ids, lost, bottom=keep, color=R, edgecolor=RD, label="ek adımıyla giden Türkçe harf (ekteki ı/ü/ğ, yumuşamış kök sonu)")
for i, (k, l) in enumerate(zip(keep, lost)):
    ax.text(i, k + l + 0.1, str(k + l), ha="center", fontsize=8)
ax.set_ylabel("Türkçe harf sayısı (tr_a)"); ax.legend(fontsize=8.5, loc="upper right"); ax.grid(axis="y", alpha=0.25)
ax.set_title(f"Ders 28 — 'Ek' adımı saf değil: tr_a → tr_bare {tot.sum()} Türkçe harfin {lost.sum()}'ini de siler "
             f"(%{100*lost.sum()/tot.sum():.0f}) → ek etkisini ASCII düzeyinde ölç (tr_ascii − tr_bare_ascii)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(out, "03-suffix-step-confound.png"), dpi=160); plt.close(fig)

# 04 — bütçe v0.1 → v0.2
bud = list(csv.DictReader(open(os.path.join(HERE, "results", "budget_v01_v02.csv"), encoding="utf-8")))
fig, ax = plt.subplots(figsize=(11, 4.8))
lab = [f"{b['matrix']}\nn_init {b['n_init_core']}" for b in bud]
mid = np.array([float(b["pilot_core_orta"]) for b in bud]); lo = np.array([float(b["pilot_core_iyimser"]) for b in bud]); hi = np.array([float(b["pilot_core_kötümser"]) for b in bud])
col = [B if b["matrix"] == "v0.1" else Y for b in bud]; ed = [BD if b["matrix"] == "v0.1" else YD for b in bud]
ax.bar(range(len(bud)), mid, color=col, edgecolor=ed)
ax.errorbar(range(len(bud)), mid, yerr=[mid - lo, hi - mid], fmt="none", ecolor="#444", capsize=4)
for i, b in enumerate(bud):
    ax.text(i, hi[i] + 8, f"{mid[i]:.0f} sa\n≈ {b['kaggle_weeks_orta']} hafta", ha="center", fontsize=8.5)
ax.axhline(30, color="#555", ls="--", lw=1); ax.text(3.4, 33, "Kaggle ~30 sa/hafta", ha="right", fontsize=8.5)
ax.set_xticks(range(len(bud))); ax.set_xticklabels(lab); ax.set_ylabel("pilot + çekirdek T4 saati (orta; çizgi iyimser–kötümser)")
ax.set_ylim(0, hi.max() * 1.2); ax.grid(axis="y", alpha=0.25)
ax.set_title("Ders 28 — Ablation'ların fiyatı (Ders 27 modeli, model tahmini): çekirdeğe giren tek yeni hücre AB1 (+46 sa orta, yalında +18)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(out, "04-budget-v01-v02.png"), dpi=160); plt.close(fig)
print("yazıldı: 02, 03, 04 →", out)
