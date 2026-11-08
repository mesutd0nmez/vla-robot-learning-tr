"""Ders 15 görselleri 02, 03, 05 (matplotlib).
  02-metric-geometry.png : libero_spatial q01/q99 (GERÇEK, dataset_statistics.json) → bir bin ve bir gripper flip'inin
                           L2'ye katkısı, normalize uzay vs gerçek birim
  03-token-counts.png    : commands.csv'nin LLaMA-2 token sayıları (GERÇEK, count_tokens.mjs çıktısı token_counts.json)
  05-literature-gap.png  : Chen vd. 2026 (arXiv:2606.15714) Tablo 1 LIBERO göreli düşüşler + Dong vd. 2026 (arXiv:2606.11906)
                           Tablo 1 LIBERO-Spatial mutlak başarı (GERÇEK makale değerleri)
    python plot_day15.py --tokens token_counts.json --out ../../lectures/images/lecture-15
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

Y, B, G, R = "#ffd166", "#cfe8ff", "#d4f5d4", "#ffd6d6"
Q01 = np.array([-0.7454732114076613, -0.6616071462631226, -0.9375, -0.1071428582072258, -0.20678570866584778, -0.1842857152223587])
Q99 = np.array([0.9375, 0.8758928775787354, 0.9321428537368774, 0.1039285734295845, 0.17678570747375488, 0.14571428298950195])
DIMS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


def fig02(out):
    one_bin_norm = 2 / 255
    one_bin_real = (Q99 - Q01) / 255
    full_norm = np.full(6, 2.0)                 # bin 0 → 254 : -0.996 → +0.996
    full_real = (Q99 - Q01) * (254 / 255)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    x = np.arange(7)
    ax = axes[0]
    ax.bar(x - 0.2, np.r_[full_norm, 1.992], 0.4, color=Y, edgecolor="#333", label="normalize uzay [-1,1] (bin merkezi)")
    ax.bar(x + 0.2, np.r_[full_real, 1.992], 0.4, color=B, edgecolor="#333", label="gerçek birim (q01/q99 ile açılmış)")
    for i in range(7):
        ax.text(i + 0.2, (np.r_[full_real, 1.992])[i] + 0.03, f"{(np.r_[full_real, 1.992])[i]:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(DIMS)
    ax.set_ylabel("tek boyutun L2'ye EN BÜYÜK katkısı")
    ax.set_title("Bir boyut uçtan uca değişirse L2'ye ne katar?\n(libero_spatial q01/q99 — gerçek değerler)", fontsize=10)
    ax.set_ylim(0, 2.6)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    ax.bar(x[:6] - 0.2, np.full(6, one_bin_norm), 0.4, color=Y, edgecolor="#333", label="1 bin, normalize (2/255 = 0.0078)")
    ax.bar(x[:6] + 0.2, one_bin_real, 0.4, color=B, edgecolor="#333", label="1 bin, gerçek birim ((q99−q01)/255)")
    for i in range(6):
        ax.text(i + 0.2, one_bin_real[i] + 0.0002, f"{one_bin_real[i]:.4f}", ha="center", fontsize=8)
    ax.set_xticks(x[:6])
    ax.set_xticklabels(DIMS[:6])
    ax.set_title("Bir bin'lik fark ne kadar?\n(dönüş eksenleri gerçek birimde 5–8× daha ince)", fontsize=10)
    ax.set_ylim(0, 0.0105)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Metrik geometrisi: gripper tek başına L2 ≈ 2.0 katar; 'hangi uzayda ölçtün' yazılmazsa sayılar karşılaştırılamaz",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "02-metric-geometry.png"), dpi=150)


def fig03(tokens_json, out):
    d = json.load(open(tokens_json, encoding="utf-8"))
    rows = d["rows"]
    conds = ["en", "para", "tr", "tr2", "distractor", "empty"]
    labels = {"en": "EN", "para": "EN parafraz", "tr": "TR", "tr2": "TR-2", "distractor": "çeldirici", "empty": "boş"}
    colors = {"en": B, "para": B, "tr": Y, "tr2": Y, "distractor": R, "empty": R}
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    w = 0.13
    for j, c in enumerate(conds):
        v = [next(r["prompt_tokens_with_bos"] for r in rows if r["task_id"] == t and r["condition"] == c) for t in range(5)]
        ax.bar(np.arange(5) + (j - 2.5) * w, v, w, color=colors[c], edgecolor="#333", label=labels[c])
        for t, val in enumerate(v):
            ax.text(t + (j - 2.5) * w, val + 0.4, str(val), ha="center", fontsize=6.5)
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"görev {t}" for t in range(5)])
    ax.set_ylabel("prompt token sayısı (BOS dahil)")
    ax.set_title("Prompt uzunluğu: 'In: What action … to {komut}?\\nOut:' (LLaMA-2 tokenizer)", fontsize=10)
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 62)
    ax = axes[1]
    ph = [p for p in d["phrases"] if p["kind"] in ("en", "tr", "tr2")]
    en = [p for p in ph if p["kind"] == "en"]
    tr = [p for p in ph if p["kind"] == "tr"]
    tr2 = [p for p in ph if p["kind"] == "tr2"]
    xs = np.arange(5)
    ax.bar(xs - 0.27, [p["n"] for p in en], 0.27, color=B, edgecolor="#333", label="EN ifade")
    ax.bar(xs, [p["n"] for p in tr], 0.27, color=Y, edgecolor="#333", label="TR (-daki)")
    ax.bar(xs + 0.27, [p["n"] for p in tr2], 0.27, color="#f3c86a", edgecolor="#333", label="TR-2 (duran)")
    for i in range(5):
        ax.text(xs[i] - 0.27, en[i]["n"] + 0.2, str(en[i]["n"]), ha="center", fontsize=8)
        ax.text(xs[i], tr[i]["n"] + 0.2, str(tr[i]["n"]), ha="center", fontsize=8)
        ax.text(xs[i] + 0.27, tr2[i]["n"] + 0.2, str(tr2[i]["n"]), ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(["between…\narasındaki", "next to…\nyanındaki", "table center\nortasındaki", "on the box\nüstündeki",
                        "in the drawer\nçekmecesindeki"], fontsize=8)
    ax.set_ylabel("yalnızca mekânsal ifadenin token sayısı")
    ax.set_title("Mekânsal ifade tek başına: TR 1.6–3×\n(ör. 'çekmecesindeki' = ç|ek|m|ec|es|inde|ki)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Gerçek token sayıları — commands.csv × llama-tokenizer-js (Mac'te HF tokenizer ile doğrula)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "03-token-counts.png"), dpi=150)


def fig05(out):
    # Chen vd. 2026, arXiv:2606.15714, Tablo 1 — LIBERO ortalaması, İngilizce'ye göre puan düşüşü
    langs = ["zh", "fr", "ru", "ar", "komut yok"]
    oft = [-28.6, -28.6, -27.8, -29.8, -27.5]
    pi05 = [-39.6, -38.0, -41.8, -41.4, -43.5]
    # Dong vd. 2026, arXiv:2606.11906, Tablo 1 — LIBERO-Spatial mutlak başarı (%)
    langs2 = ["EN", "zh", "ja", "ko", "fr", "es", "ar", "th", "pt", "vi"]
    oft2 = [97.6, 55.8, 56.8, 49.2, 64.0, 63.8, 52.8, 53.0, 62.6, 59.2]
    pi2 = [97.8, 67.8, 70.0, 69.8, 72.8, 73.6, 68.0, 68.6, 75.4, 69.6]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axes[0]
    x = np.arange(len(langs))
    ax.bar(x - 0.2, oft, 0.4, color=[B] * 4 + [R], edgecolor="#333", label="OpenVLA-OFT")
    ax.bar(x + 0.2, pi05, 0.4, color=[Y] * 4 + [R], edgecolor="#333", hatch="//", label="π0.5")
    for i in range(5):
        ax.text(i - 0.2, oft[i] - 1.5, f"{oft[i]}", ha="center", fontsize=8)
        ax.text(i + 0.2, pi05[i] - 1.5, f"{pi05[i]}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(langs)
    ax.set_ylabel("başarı farkı (puan, EN'ye göre)")
    ax.set_ylim(-50, 0)
    ax.set_title("Chen vd. 2026 (arXiv:2606.15714) Tablo 1, LIBERO ort.\nOpenVLA-OFT: yabancı dil ≈ 'komut yok' tabanı", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    x = np.arange(len(langs2))
    ax.bar(x - 0.2, oft2, 0.4, color=B, edgecolor="#333", label="OpenVLA-OFT")
    ax.bar(x + 0.2, pi2, 0.4, color=Y, edgecolor="#333", label="π0.5")
    for i in range(len(langs2)):
        ax.text(i - 0.2, oft2[i] + 1, f"{oft2[i]:.0f}", ha="center", fontsize=7)
        ax.text(i + 0.2, pi2[i] + 1, f"{pi2[i]:.0f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(langs2)
    ax.set_ylim(0, 110)
    ax.set_ylabel("LIBERO-Spatial başarı (%)")
    ax.set_title("Dong vd. 2026 (arXiv:2606.11906) Tablo 1, LIBERO-Spatial\n9 dilde 22–48 puan düşüş; Türkçe yok", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Haziran 2026: çok dilli VLA boşluğu iki bağımsız çalışmada ölçüldü — ikisinde de metrik success rate, Türkçe yok",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "05-literature-gap.png"), dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="token_counts.json")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig02(args.out)
    if os.path.exists(args.tokens):
        fig03(args.tokens, args.out)
    else:
        print("token_counts.json yok; önce count_tokens.mjs çalıştır")
    fig05(args.out)
    print("yazıldı:", args.out)


if __name__ == "__main__":
    main()
