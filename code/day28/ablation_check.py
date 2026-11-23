"""Lecture 28 — Does the surface ablation really change only one factor?

The surface axis of RQ-b is defined by four Turkish forms (datasets/04): tr_a (Turkish letters + suffixes), tr_ascii
(letters folded to ASCII), tr_bare (suffixes stripped, letters kept). In ablation terms this is a 2×2:
  letter {TR, ASCII} × suffix {present, absent}  →  tr_a | tr_ascii | tr_bare | tr_bare_ascii (MISSING CELL — produced here)
An ablation step must change only one factor. This script measures that for every pair:
  1. fold(tr_a) == tr_ascii ?           (does the letter step change only letters — same words, suffixes, order?)
  2. stems of tr_a_seg == tr_bare ?     (does the suffix step strip only suffixes — with consonant softening undone)
  3. Number of Turkish letters, suffixes and words on every surface
  4. Token counts: LLaMA-2 (OpenVLA) and Qwen3 (VLANeXt) — the existing surfaces are cross-checked against
     datasets/05 tokens.csv (must match exactly); the new tr_bare_ascii is counted with the same tokenizers.
  5. 2×2 token effect: letter effect, suffix effect, interaction (mean per tokenizer ± min/max).

  python ablation_check.py --llama tokenizer.model --qwen package/models/tokenizer.json
  # tokenizers: code/day25/README.md (Meta tokenizer.model sha256 9e556afd…; npm @lenml/tokenizer-qwen3)
  python ablation_check.py            # without tokenizers: only 1–3 and the counts read from datasets/05
"""
import argparse, csv, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
D04 = os.path.join(HERE, "..", "..", "datasets", "04-tr-spatial-commands", "data", "pairs.csv")
D05 = os.path.join(HERE, "..", "..", "datasets", "05-tokenization-tr-commands", "data", "tokens.csv")
TR = "çğıöşüâîûÇĞİÖŞÜÂÎÛ"
FOLD = str.maketrans("çğıöşüâîûÇĞİÖŞÜÂÎÛ", "cgiosuaiuCGIOSUAIU")
HARDEN = {"ğ": "k", "b": "p", "c": "ç", "d": "t"}   # tabağ- → tabak, dolab- → dolap (the reverse of consonant softening)


def stem_ok(word_seg, bare_word):
    """Stem = the first piece of tr_a_seg; in a suffixed word the stem end may have softened (tabağ-ın → tabak),
    but not every ğ/b/c/d is softening (sağ-da-ki → sağ). Accept if either one matches."""
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
    L.append(f"# Lecture 28 — surface ablation check ({len(rows)} pairs)\n")
    L.append(f"- tr_a_seg with hyphens removed equals tr_a: {sum(r['seg_join_ok'] for r in rows)}/{len(rows)}")
    L.append(f"- letter step (tr_a → tr_ascii) changes only letters: {sum(r['letter_step_only_letters'] for r in rows)}/{len(rows)}")
    L.append(f"- suffix step (tr_a → tr_bare) strips only suffixes (stem, with consonant hardening undone where needed): {sum(r['suffix_step_only_suffixes'] for r in rows)}/{len(rows)}")
    bad = [r for r in rows if not r["suffix_step_only_suffixes"]]
    for r in bad:
        L.append(f"  - {r['pair_id']}: stems `{r['stems_if_diff']}` ≠ tr_bare")
    L.append(f"- word count tr_a = tr_bare: {sum(r['n_words_tr_a']==r['n_words_bare'] for r in rows)}/{len(rows)}")
    L.append(f"- suffix count tr_a: total {sum(r['n_suffix_tr_a'] for r in rows)}, per pair {st.mean(r['n_suffix_tr_a'] for r in rows):.1f} (0 in tr_bare)")
    L.append(f"- Turkish letters: tr_a {sum(r['trletters_tr_a'] for r in rows)}, tr_bare {sum(r['trletters_bare'] for r in rows)}, "
             f"tr_ascii {sum(r['trletters_ascii'] for r in rows)}, tr_bare_ascii {sum(r['trletters_bare_ascii'] for r in rows)}")
    if toks:
        L.append(f"- datasets/05 tokens.csv cross-check (4 surfaces × 20 × 2 tokenizers): mismatches {mism}")
    L.append("\n## 2×2 token table (mean tokens per pair; min–max in parentheses)\n")
    L.append("| tokenizer | en | tr_a (TR, suffix) | tr_ascii (ASCII, suffix) | tr_bare (TR, no suffix) | tr_bare_ascii (ASCII, no suffix) | letter effect | suffix effect | interaction |")
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
    L.append("\nletter effect = TR letters − ASCII (mean over the two suffix levels); suffix effect = suffixed − bare; interaction = difference of the letter effect between suffixed and bare.")
    txt = "\n".join(L)
    open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
