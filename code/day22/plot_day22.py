"""Ders 22 görselleri — tokenize_set.py çıktısından (gerçek sayılar).

  python plot_day22.py --data ../../datasets/05-tokenization-tr-commands/data --out ../../lectures/images/lecture-22

02-token-counts.png   yüzey × tokenizer ortalama token (min–max çubuğu) + çift başına tr_a/en oranı (LLaMA-2 vs Qwen3)
03-segmentation.png   seçilmiş ilişki sözcüklerinin parça şeritleri; kesik çizgi = biçimbirim sınırı (tr_morph ayrımı)
04-alignment.png      biçimbirim↔token sınır örtüşmesi, ilişki ekinin token biçimleri, tr_a − tr_ascii farkı
Renk dili: sarı #ffd166 model/anahtar, mavi #cfe8ff girdi, yeşil #d4f5d4 çıktı, kırmızı #ffd6d6 sorun.
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

YEL, BLU, GRN, RED = "#ffd166", "#cfe8ff", "#d4f5d4", "#ffd6d6"
YEL_D, BLU_D, GRN_D, RED_D = "#c99a1e", "#4a7fb5", "#3f9d5a", "#c0504d"
L_COL, Q_COL = YEL_D, BLU_D   # LLaMA-2 = sarı (OpenVLA), Qwen3 = mavi (VLANeXt)
VARIANTS = ["en", "tr_a", "tr_b", "tr_ascii", "tr_bare"]


def read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def fig02(tokens, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})
    w = 0.38
    for k, (name, col, lab) in enumerate((("llama2", L_COL, "LLaMA-2 (OpenVLA)"), ("qwen3", Q_COL, "Qwen3 (VLANeXt)"))):
        xs, ms, lo, hi = [], [], [], []
        for i, v in enumerate(VARIANTS):
            n = [int(r["n_tokens"]) for r in tokens if r["variant"] == v and r["tokenizer"] == name]
            xs.append(i + (k - 0.5) * w)
            ms.append(mean(n)); lo.append(mean(n) - min(n)); hi.append(max(n) - mean(n))
        ax1.bar(xs, ms, w, color=col, alpha=0.9, label=lab, yerr=[lo, hi], capsize=3, ecolor="#555", error_kw={"lw": 1})
        for x, m in zip(xs, ms):
            ax1.text(x, m + 0.6, f"{m:.1f}", ha="center", va="bottom", fontsize=9)
    ax1.set_xticks(range(len(VARIANTS))); ax1.set_xticklabels(VARIANTS)
    ax1.set_ylabel("komut başına token (20 çift ortalaması; çubuk min–max)")
    ax1.set_title("Aynı 100 komut, iki tokenizer — yalnızca komut metni (şablon yok)")
    ax1.legend(loc="upper right"); ax1.grid(axis="y", alpha=0.3); ax1.set_ylim(0, 40)
    # oran
    byid = {(r["pair_id"], r["variant"], r["tokenizer"]): int(r["n_tokens"]) for r in tokens}
    pids = sorted({r["pair_id"] for r in tokens})
    blocks = {"A": GRN_D, "B": RED_D, "C": "#7a5cc2", "D": "#777"}
    for p in pids:
        rl = byid[(p, "tr_a", "llama2")] / byid[(p, "en", "llama2")]
        rq = byid[(p, "tr_a", "qwen3")] / byid[(p, "en", "qwen3")]
        ax2.scatter(rl, rq, s=60, color=blocks[p[0]], edgecolor="k", zorder=3)
        ax2.annotate(p, (rl, rq), textcoords="offset points", xytext=(4, 3), fontsize=7)
    ax2.plot([0.8, 2.1], [0.8, 2.1], "--", color="#999", lw=1)
    ax2.axhline(1, color="#bbb", lw=0.8); ax2.axvline(1, color="#bbb", lw=0.8)
    ax2.set_xlabel("LLaMA-2: token(tr_a) / token(en)"); ax2.set_ylabel("Qwen3: token(tr_a) / token(en)")
    ax2.set_title("Çift başına Türkçe/İngilizce oranı (köşegen = eşit şişme)")
    for b, c in blocks.items():
        ax2.scatter([], [], color=c, edgecolor="k", label=f"blok {b}")
    ax2.legend(loc="lower right", fontsize=8); ax2.grid(alpha=0.3)
    ax2.set_xlim(0.8, 2.1); ax2.set_ylim(0.8, 2.1)
    fig.suptitle("Kaynak: tokenize_set.py — Meta LLaMA-2 tokenizer.model (sha256 doğrulandı) ve Qwen3 tokenizer.json; datasets/04 v0.1", fontsize=9, y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "02-token-counts.png"), dpi=170); plt.close(fig)


def fig03(rel, words, out):
    """Parça şeritleri. rel_span.csv'den (cümle içi konumuyla) + words.csv'den birkaç sözcük."""
    pick = [("A01", "arasındaki"), ("A05", "üstündeki"), ("B07", "soldaki"), ("B08", "sağdaki"), ("B09", "öndeki"),
            ("C13", "üstüne"), ("C16", "içine"), ("D19", "sağına"), ("D20", "soluna")]
    rel_by = defaultdict(dict)
    for r in rel:
        rel_by[r["pair_id"]][r["tokenizer"]] = r
    extra = [("kaseyi", "kase-yi"), ("kâseyi", "kâse-yi"), ("tabağın", "tabağ-ın"), ("çekmecesindeki", "çekmece-si-nde-ki")]
    wby = defaultdict(dict)
    for r in words:
        wby[r["word"]][r["tokenizer"]] = r
    rows = []
    for pid, w in pick:
        L, Q = rel_by[pid]["llama2"], rel_by[pid]["qwen3"]
        rows.append((f"{pid} {w}", L["rel_seg"], L["pieces"].split("|"), Q["pieces"].split("|")))
    for w, seg in extra:
        rows.append((w, seg, wby[w]["llama2"]["pieces_mid"].split("|"), wby[w]["qwen3"]["pieces_mid"].split("|")))

    fig, ax = plt.subplots(figsize=(11.5, 0.62 * len(rows) + 1.6))
    scale = 0.55  # birim: karakter
    y = 0
    for label, seg, lp, qp in rows:
        surface = seg.replace("-", "")
        cuts, pos = [], 0
        for part in seg.split("-")[:-1]:
            pos += len(part); cuts.append(pos)
        off = scale if any(pp and pp[0].startswith("▁") for pp in (lp, qp)) else 0
        for k, (pieces, col, tag) in enumerate(((lp, YEL, "L"), (qp, BLU, "Q"))):
            yy = y + (0.0 if k == 0 else 0.52)   # L üstte, Q altta
            x = 0.0 if (pieces and pieces[0].startswith("▁")) else off   # cümle başı (▁ yok) satırı hizala
            for pc in pieces:
                txt = pc
                n = max(len(pc.replace("▁", "")), 1) if pc != "▁" else 1
                width = n * scale
                ax.add_patch(Rectangle((x, yy), width, 0.46, facecolor=col, edgecolor="k", lw=0.8))
                ax.text(x + width / 2, yy + 0.23, txt, ha="center", va="center", fontsize=9, family="DejaVu Sans Mono")
                x += width
            ax.text(-0.25, yy + 0.23, tag, ha="right", va="center", fontsize=9, color=L_COL if k == 0 else Q_COL, weight="bold")
        # biçimbirim sınırları (▁ bir karakter yer kaplar)
        for c in cuts:
            ax.plot([off + c * scale] * 2, [y - 0.08, y + 1.06], "--", color=RED_D, lw=1.2)
        ax.text(-1.0, y + 0.5, label + "\n" + seg, ha="right", va="center", fontsize=9)
        y += 1.35
    ax.set_xlim(-6.5, 9.5); ax.set_ylim(y, -0.3)   # ilk satır üstte
    ax.axis("off")
    ax.set_title("İlişki sözcükleri parça parça — sarı: LLaMA-2 (OpenVLA), mavi: Qwen3 (VLANeXt); kırmızı kesik çizgi: biçimbirim sınırı (tr_morph ayrımı)\n"
                 "▁ = sözcük başı boşluğu; kutu genişliği ≈ karakter sayısı. Kaynak: rel_span.csv / words.csv (gerçek tokenizer çıktısı; cümle içindeki konum)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "03-segmentation.png"), dpi=170); plt.close(fig)


def fig04(seg, sf, tokens, out):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.6), gridspec_kw={"width_ratios": [0.9, 1.1, 1.2]})
    # (a) sınır örtüşmesi
    stats = {}
    for name in ("llama2", "qwen3"):
        S = [r for r in seg if r["tokenizer"] == name]
        tm, th, tt = sum(int(r["morph_cuts"]) for r in S), sum(int(r["cuts_hit"]) for r in S), sum(int(r["token_cuts"]) for r in S)
        stats[name] = (th / tm, th / tt, tt)
    x = [0, 1]
    for k, (name, col, lab) in enumerate((("llama2", L_COL, "LLaMA-2"), ("qwen3", Q_COL, "Qwen3"))):
        rec, prec, tt = stats[name]
        ax1.bar([xx + (k - 0.5) * 0.38 for xx in x], [rec, prec], 0.38, color=col, label=f"{lab} ({tt} sözcük-içi kesim)")
        for xx, v in zip(x, (rec, prec)):
            ax1.text(xx + (k - 0.5) * 0.38, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax1.set_xticks(x); ax1.set_xticklabels(["recall\n(biçimbirim sınırı\ntoken sınırı mı?)", "precision\n(token sınırı\nbiçimbirim sınırı mı?)"], fontsize=8.5)
    ax1.set_ylim(0, 1); ax1.set_title("Biçimbirim ↔ token sınırı\n(20 tr_a komutu, 111 biçimbirim sınırı)", fontsize=10)
    ax1.legend(fontsize=7.5, loc="upper right"); ax1.grid(axis="y", alpha=0.3)
    # (b) ek biçimleri
    grp = defaultdict(list)
    for r in sf:
        grp[(r["tokenizer"], r["suffix"])].append((r["token_form"], int(r["count"])))
    labels, vals, cols = [], [], []
    for key in (("llama2", "ki"), ("qwen3", "ki"), ("llama2", "ne/na/a (DAT)"), ("qwen3", "ne/na/a (DAT)")):
        forms = grp[key]
        labels.append(f"{'L' if key[0]=='llama2' else 'Q'}  -{key[1].split()[0]}\n" + " ".join(f"{f}" for f, _ in forms))
        vals.append(len(forms)); cols.append(L_COL if key[0] == "llama2" else Q_COL)
    ax2.barh(range(len(vals)), vals, color=cols)
    ax2.set_yticks(range(len(vals))); ax2.set_yticklabels(labels, fontsize=7.5, family="DejaVu Sans Mono")
    ax2.invert_yaxis(); ax2.set_xlabel("aynı eki taşıyan farklı token biçimi sayısı")
    ax2.set_title("İlişki eki tek bir token değil\n(-ki: 9 sözcük, DAT: 9 sözcük)", fontsize=10); ax2.grid(axis="x", alpha=0.3)
    # (c) tr_a − tr_ascii
    byid = {(r["pair_id"], r["variant"], r["tokenizer"]): int(r["n_tokens"]) for r in tokens}
    pids = sorted({r["pair_id"] for r in tokens})
    dl = [byid[(p, "tr_a", "llama2")] - byid[(p, "tr_ascii", "llama2")] for p in pids]
    dq = [byid[(p, "tr_a", "qwen3")] - byid[(p, "tr_ascii", "qwen3")] for p in pids]
    xs = range(len(pids))
    ax3.bar([i - 0.2 for i in xs], dl, 0.4, color=L_COL, label=f"LLaMA-2 (ort. {mean(dl):.1f})")
    ax3.bar([i + 0.2 for i in xs], dq, 0.4, color=Q_COL, label=f"Qwen3 (ort. {mean(dq):.1f})")
    ax3.axhline(0, color="k", lw=0.8)
    ax3.set_xticks(list(xs)); ax3.set_xticklabels(pids, rotation=90, fontsize=7)
    ax3.set_ylabel("token(tr_a) − token(tr_ascii)")
    ax3.set_title("Türkçe harflerin token maliyeti\n(ç ğ ı ö ş ü → c g i o s u)", fontsize=10)
    ax3.legend(fontsize=8); ax3.grid(axis="y", alpha=0.3)
    fig.suptitle("Kaynak: tokenize_set.py çıktıları (seg_align.csv, suffix_forms.csv, tokens.csv) — gerçek tokenizer çıktısı, model yok", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "04-alignment.png"), dpi=170); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--data", default=os.path.join(here, "..", "..", "datasets", "05-tokenization-tr-commands", "data"))
    ap.add_argument("--out", default=os.path.join(here, "..", "..", "lectures", "images", "lecture-22"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tokens = read(os.path.join(a.data, "tokens.csv"))
    rel = read(os.path.join(a.data, "rel_span.csv"))
    words = read(os.path.join(a.data, "words.csv"))
    seg = read(os.path.join(a.data, "seg_align.csv"))
    sf = read(os.path.join(a.data, "suffix_forms.csv"))
    fig02(tokens, a.out); fig03(rel, words, a.out); fig04(seg, sf, tokens, a.out)
    print("yazıldı:", a.out)
