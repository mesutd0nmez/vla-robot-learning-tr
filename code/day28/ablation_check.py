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
