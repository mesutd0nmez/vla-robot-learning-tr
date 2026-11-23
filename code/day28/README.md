# day28

Ders 28 — baseline + ablation tasarımı. Model yok; yüzey analizi ve tokenizer sayımı gerçek, bütçe Ders 27 modeli. Çıktılar: `thesis/ablation-plan.md`, `thesis/experiment-matrix.csv` v0.2.

| Dosya | Ne | Durum |
|---|---|---|
| `ablation_check.py` | `datasets/04` yüzeylerinin tek faktör değiştirip değiştirmediği (harf / ek), eksik 2×2 hücresi `tr_bare_ascii`, iki tokenizer ile token sayımı (datasets/05 ile çapraz kontrol), 2×2 etkileri | çalıştırıldı → `results/ablation_surface.csv`, `results/summary.md` |
| `price_ablations.py` | Ablation hücrelerini Ders 27 `budget.py` ile fiyatlar; v0.1 ↔ v0.2 | çalıştırıldı → `results/budget_v01_v02.csv`, `results/ablation_cells.csv` |
| `plot_day28.py` | Görsel 02–04 | çalıştırıldı |

```bash
cd code/day28
python ablation_check.py                                   # tokenizer'sız (datasets/05 sayıları)
python ablation_check.py --llama tokenizer.model --qwen package/models/tokenizer.json   # code/day25/README.md'deki indirme
python price_ablations.py
python plot_day28.py
```

## ablation_check.py

```python
"""Ders 28 — Yüzey ablation'ı gerçekten tek faktör mü değiştiriyor?

RQ-b'nin yüzey ekseni dört Türkçe biçimle tanımlı (datasets/04): tr_a (Türkçe harf + ek), tr_ascii
(harfler ASCII'ye katlanmış), tr_bare (ekler soyulmuş, harfler duruyor). Ablation dilinde bu bir 2×2:
  harf {TR, ASCII} × ek {var, yok}  →  tr_a | tr_ascii | tr_bare | tr_bare_ascii (EKSİK HÜCRE — burada üretilir)
Bir ablation adımı yalnızca bir faktörü değiştirmeli. Bu script her çift için bunu ölçer:
  1. fold(tr_a) == tr_ascii ?           (harf adımı yalnızca harf mi değiştiriyor — sözcük, ek, sıra aynı mı)
  2. tr_a_seg'in kökleri == tr_bare ?    (ek adımı yalnızca ek mi soyuyor — ünsüz yumuşaması geri alınarak)
  3. Türkçe harf sayısı, ek sayısı, sözcük sayısı her yüzeyde
  4. Token sayısı: LLaMA-2 (OpenVLA) ve Qwen3 (VLANeXt) — mevcut yüzeyler datasets/05 tokens.csv ile çapraz
     kontrol edilir (birebir olmalı); yeni tr_bare_ascii aynı tokenizer'larla sayılır.
  5. 2×2 token etkisi: harf etkisi, ek etkisi, etkileşim (tokenizer başına ortalama ± min/max).

  python ablation_check.py --llama tokenizer.model --qwen package/models/tokenizer.json
  # tokenizer'lar: code/day25/README.md (Meta tokenizer.model sha256 9e556afd…; npm @lenml/tokenizer-qwen3)
  python ablation_check.py            # tokenizer'sız: yalnız 1–3 ve datasets/05'ten okunan sayılar
"""
import argparse, csv, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
D04 = os.path.join(HERE, "..", "..", "datasets", "04-tr-spatial-commands", "data", "pairs.csv")
D05 = os.path.join(HERE, "..", "..", "datasets", "05-tokenization-tr-commands", "data", "tokens.csv")
TR = "çğıöşüâîûÇĞİÖŞÜÂÎÛ"
FOLD = str.maketrans("çğıöşüâîûÇĞİÖŞÜÂÎÛ", "cgiosuaiuCGIOSUAIU")
HARDEN = {"ğ": "k", "b": "p", "c": "ç", "d": "t"}   # tabağ- → tabak, dolab- → dolap (ünsüz yumuşamasının tersi)


def stem_ok(word_seg, bare_word):
    """Kök = tr_a_seg'in ilk parçası; ekli sözcükte kök sonu yumuşamış olabilir (tabağ-ın → tabak),
    ama her ğ/b/c/d yumuşama değildir (sağ-da-ki → sağ). İkisinden biri tutarsa kabul."""
    s = word_seg.split("-")[0]
    cands = {s}
    if "-" in word_seg and s and s[-1] in HARDEN:
        cands.add(s[:-1] + HARDEN[s[-1]])
    return bare_word in cands


def ntr(s):
    return sum(ch in TR for ch in s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama"); ap.add_argument("--qwen")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    pairs = list(csv.DictReader(open(D04, encoding="utf-8")))
    tok05 = {(r["pair_id"], r["variant"], r["tokenizer"]): int(r["n_tokens"]) for r in csv.DictReader(open(D05, encoding="utf-8"))}
    toks = {}
    if a.llama and a.qwen:
        import sentencepiece as spm
        from tokenizers import Tokenizer
        sp = spm.SentencePieceProcessor(model_file=a.llama); qw = Tokenizer.from_file(a.qwen)
        toks = {"llama2": lambda s: len(sp.encode(s)), "qwen3": lambda s: len(qw.encode(s, add_special_tokens=False).ids)}
    rows, mism = [], 0
    for p in pairs:
        tr_a, seg, asc, bare = p["tr_a"], p["tr_a_seg"], p["tr_ascii"], p["tr_bare"]
        bare_ascii = bare.translate(FOLD)
        seg_join = seg.replace("-", "") == tr_a
        letter_only = tr_a.translate(FOLD) == asc
        sw, bw = seg.split(), bare.split()
        suffix_only = len(sw) == len(bw) and all(stem_ok(x, y) for x, y in zip(sw, bw))
        stems = " ".join(x.split("-")[0] for x in sw)
        surf = {"en": p["en"], "tr_a": tr_a, "tr_ascii": asc, "tr_bare": bare, "tr_bare_ascii": bare_ascii}
        r = dict(pair_id=p["pair_id"], block=p["block"], seg_join_ok=seg_join, letter_step_only_letters=letter_only,
                 suffix_step_only_suffixes=suffix_only, stems_if_diff="" if suffix_only else stems,
                 n_suffix_tr_a=seg.count("-"), n_words_tr_a=len(tr_a.split()), n_words_bare=len(bare.split()),
                 trletters_tr_a=ntr(tr_a), trletters_ascii=ntr(asc), trletters_bare=ntr(bare), trletters_bare_ascii=ntr(bare_ascii),
                 tr_bare_ascii=bare_ascii)
        for tk in ("llama2", "qwen3"):
            for v, s in surf.items():
                n = toks[tk](s) if toks else tok05.get((p["pair_id"], v, tk))
                r[f"{tk}_{v}"] = n
                if toks and v != "tr_bare_ascii" and (p["pair_id"], v, tk) in tok05 and tok05[(p["pair_id"], v, tk)] != n:
                    mism += 1
        rows.append(r)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "ablation_surface.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    L = []
    L.append(f"# Ders 28 — yüzey ablation kontrolü ({len(rows)} çift)\n")
    L.append(f"- tr_a_seg tirelerden arındırılınca tr_a: {sum(r['seg_join_ok'] for r in rows)}/{len(rows)}")
    L.append(f"- harf adımı (tr_a → tr_ascii) yalnız harf değiştiriyor: {sum(r['letter_step_only_letters'] for r in rows)}/{len(rows)}")
    L.append(f"- ek adımı (tr_a → tr_bare) yalnız ek soyuyor (kök, gerekirse ünsüz sertleşmesi geri alınarak): {sum(r['suffix_step_only_suffixes'] for r in rows)}/{len(rows)}")
    bad = [r for r in rows if not r["suffix_step_only_suffixes"]]
    for r in bad:
        L.append(f"  - {r['pair_id']}: kökler `{r['stems_if_diff']}` ≠ tr_bare")
    L.append(f"- sözcük sayısı tr_a = tr_bare: {sum(r['n_words_tr_a']==r['n_words_bare'] for r in rows)}/{len(rows)}")
    L.append(f"- ek sayısı tr_a: toplam {sum(r['n_suffix_tr_a'] for r in rows)}, çift başına {st.mean(r['n_suffix_tr_a'] for r in rows):.1f} (tr_bare'de 0)")
    L.append(f"- Türkçe harf: tr_a {sum(r['trletters_tr_a'] for r in rows)}, tr_bare {sum(r['trletters_bare'] for r in rows)}, "
             f"tr_ascii {sum(r['trletters_ascii'] for r in rows)}, tr_bare_ascii {sum(r['trletters_bare_ascii'] for r in rows)}")
    if toks:
        L.append(f"- datasets/05 tokens.csv çapraz kontrol (4 yüzey × 20 × 2 tokenizer): uyuşmayan {mism}")
    L.append("\n## 2×2 token tablosu (çift başına ortalama token; parantez min–max)\n")
    L.append("| tokenizer | en | tr_a (TR, ek) | tr_ascii (ASCII, ek) | tr_bare (TR, eksiz) | tr_bare_ascii (ASCII, eksiz) | harf etkisi | ek etkisi | etkileşim |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    eff = {}
    for tk in ("llama2", "qwen3"):
        m = {v: st.mean(r[f"{tk}_{v}"] for r in rows) for v in ("en", "tr_a", "tr_ascii", "tr_bare", "tr_bare_ascii")}
        rng = {v: (min(r[f"{tk}_{v}"] for r in rows), max(r[f"{tk}_{v}"] for r in rows)) for v in m}
        letter = ((m["tr_a"] - m["tr_ascii"]) + (m["tr_bare"] - m["tr_bare_ascii"])) / 2
        suffix = ((m["tr_a"] - m["tr_bare"]) + (m["tr_ascii"] - m["tr_bare_ascii"])) / 2
        inter = (m["tr_a"] - m["tr_ascii"]) - (m["tr_bare"] - m["tr_bare_ascii"])
        eff[tk] = (letter, suffix, inter)
        L.append(f"| {tk} | " + " | ".join(f"{m[v]:.1f} ({rng[v][0]}–{rng[v][1]})" for v in m) + f" | {letter:+.1f} | {suffix:+.1f} | {inter:+.1f} |")
    L.append("\nharf etkisi = TR harfli − ASCII (iki ek düzeyinin ortalaması); ek etkisi = ekli − eksiz; etkileşim = harf etkisinin ekli ve eksizdeki farkı.")
    txt = "\n".join(L)
    open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
```

## price_ablations.py

```python
"""Ders 28 — Ablation hücrelerini Ders 27'nin bütçe modeliyle fiyatla (v0.1 → v0.2 farkı).

  python price_ablations.py      # ../../thesis/experiment-matrix.csv (v0.2) ve results/experiment-matrix_v0.1.csv

Ders 27'nin `budget.py`'si içe aktarılır (aynı episode modeli, aynı senaryolar); onun results/ klasörüne
DOKUNULMAZ — çıktılar code/day28/results/ altına yazılır.
Yeniden okuma (yeni episode gerektirmeyen) ablation'lar matriste satır değil; ablation-plan.md'de listelenir.
"""
import csv, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "day27"))
import budget as B  # noqa: E402

B.P_SUCC.update({"tr_bare_ascii": 0.4, "tr_cs": 0.6})   # yalnız maliyet karışımı (hipotez), sonuç tahmini değil


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def totals(res, lean=None):
    t = {}
    for tier in ("pilot", "core", "ext"):
        t[tier] = {k: sum(r["h_" + k] for r in res if r["tier"] == tier and r["where"] == "T4") for k in B.SCEN}
    return t


def main():
    v1 = load(os.path.join(HERE, "results", "experiment-matrix_v0.1.csv"))
    v2 = load(os.path.join(HERE, "..", "..", "thesis", "experiment-matrix.csv"))
    out = []
    for name, rows in (("v0.1", v1), ("v0.2", v2)):
        for lean in (None, 20):
            res = B.budget([dict(r) for r in rows], n_init_core=lean)
            t = totals(res)
            pc = {k: t["pilot"][k] + t["core"][k] for k in B.SCEN}
            out.append(dict(matrix=name, n_init_core=lean or 50, **{f"pilot_core_{k}": round(v, 1) for k, v in pc.items()},
                            ext_orta=round(t["ext"]["orta"], 1), kaggle_weeks_orta=round(pc["orta"] / B.KAGGLE_WEEK_H, 1)))
            if name == "v0.2" and lean is None:
                cells = [r for r in res if r["cell"].startswith("AB") or r["cell"] == "B6"]
    with open(os.path.join(HERE, "results", "budget_v01_v02.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    with open(os.path.join(HERE, "results", "ablation_cells.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cells[0])); w.writeheader(); w.writerows(cells)
    print("Ablation hücreleri (v0.2, n_init 50):")
    for r in cells:
        print(f"  {r['cell']:4s} {r['tier']:5s} episode {r['episodes']:5d}  T4 sa iyimser/orta/kötümser {r['h_iyimser']:5.1f} / {r['h_orta']:5.1f} / {r['h_kötümser']:5.1f}  {r['status']}")
    print("\nToplam (pilot + çekirdek, T4):")
    for o in out:
        print(f"  {o['matrix']}  n_init {o['n_init_core']:2d}: iyimser {o['pilot_core_iyimser']:6.1f}  orta {o['pilot_core_orta']:6.1f}  "
              f"kötümser {o['pilot_core_kötümser']:6.1f}  | genişletme (orta) {o['ext_orta']:6.1f} | Kaggle haftası {o['kaggle_weeks_orta']}")


if __name__ == "__main__":
    main()
```

## plot_day28.py

```python
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
```
