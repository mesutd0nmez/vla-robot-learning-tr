"""Lecture 22 — the same 100 commands, two tokenizers: LLaMA-2 (OpenVLA) and Qwen3 (Qwen3-VL / VLANeXt).

NO model needed. Two sources:
  LLaMA-2 : Meta's `tokenizer.model` (SentencePiece BPE, 32,000 pieces; sha256 9e556afd…) -> the `sentencepiece` package
            or a tokenizer.json / HF id (openvla/openvla-7b) via HF `tokenizers`
  Qwen3   : `tokenizer.json` (byte-level BPE, 151,643 + 26 special) -> HF `tokenizers` (Rust; the one transformers uses)
Input: datasets/04-tr-spatial-commands/data/{commands,pairs}.csv (Lecture 20).

Outputs (--out directory):
  tokens.csv       100 rows × 2 tokenizers: token count, word count, fertility (tokens/word), bytes/token,
                   standalone whitespace token, byte fragment (a piece that is not valid UTF-8 on its own),
                   total tokens inside the OpenVLA prompt (incl. BOS) / text tokens inside the Qwen template, the pieces
  rel_span.csv     20 pairs × 2 tokenizers: the pieces of the relation word (üstündeki, soldaki, sağına…) inside the sentence;
                   overlap of morpheme boundaries with token boundaries; whether the final suffix (-ki / -ne / -na) is a separate piece
  seg_align.csv    20 tr_a commands × 2 tokenizers: overlap of ALL morpheme boundaries in tr_a_seg with token boundaries
  words.csv        every distinct word in the set × 2 tokenizers: in-sentence pieces (sağına/soluna, kase/kâse, üstüne/ustune)
  suffix_forms.csv token forms that carry the relation suffix (-ki, -ne/-na): in how many different tokens does the same suffix appear?
  action_collision.csv  tokens that fall into the last 256 ids (31744–31999) of the LLaMA-2 vocab, which OpenVLA reserves for action bins
  crosscheck.md    (--jsids) are the llama-tokenizer-js ids identical to the SentencePiece ids? (the open hypothesis of Lecture 15)
  summary.md       summary tables (the numbers in the lecture text come from here)

Usage (Mac, `vla` env):
  pip install sentencepiece tokenizers
  curl -L -o tokenizer.model https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.model   # Meta LLaMA-2 tokenizer
  shasum -a 256 tokenizer.model      # 9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347 (Meta's checklist)
  python tokenize_set.py --llama tokenizer.model --qwen Qwen/Qwen3-VL-2B-Instruct --out ../../datasets/05-tokenization-tr-commands/data
  # offline: --qwen /path/tokenizer.json ;  with OpenVLA's HF tokenizer: --llama openvla/openvla-7b
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "datasets", "04-tr-spatial-commands", "data")
OPENVLA_PROMPT = "In: What action should the robot take to {ins}?\nOut:"
# VLANeXt: the command is wrapped in the Qwen chat template (Lecture 13); image tokens are counted separately (256 px -> 64 tokens/image).
QWEN_TEMPLATE = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{ins}<|im_end|>\n<|im_start|>assistant\n"
LLAMA_ACTION_IDS = range(32000 - 256, 32000)  # OpenVLA: "the 256 least used tokens … the last 256 tokens"
LLAMA_SHA256 = "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347"


# ----------------------------------------------------------------------------------------------------------------
# Common interface for the two tokenizers: encode(text) -> ids, pieces (▁ = word-initial space), offsets (characters)
# ----------------------------------------------------------------------------------------------------------------
class SPTokenizer:
    """Meta tokenizer.model (SentencePiece). add_dummy_prefix=True: a ▁ is prepended to every text (HF legacy behavior)."""
    name = "llama2"

    def __init__(self, path):
        import hashlib
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor(model_file=path)
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()
        self.src = f"{path}  sha256={'OK (Meta LLaMA/LLaMA-2)' if h == LLAMA_SHA256 else h[:16] + '… (DIFFERENT from the Meta hash)'}"
        self.vocab_size = self.sp.get_piece_size()

    def encode(self, text, add_bos=False):
        try:                                   # sentencepiece >= 0.2.1
            proto = self.sp.encode(text, return_type="proto")
        except (TypeError, ValueError):        # older versions
            proto = self.sp.encode(text, out_type="immutable_proto")
        ids = [p.id for p in proto.pieces]
        pieces = [p.piece for p in proto.pieces]
        # byte -> character position
        b2c, b = {}, 0
        for ci, ch in enumerate(text):
            for _ in ch.encode("utf-8"):
                b2c[b] = ci
                b += 1
        b2c[b] = len(text)
        offsets = [(b2c.get(p.begin, 0), b2c.get(p.end, len(text))) for p in proto.pieces]
        if add_bos:
            ids = [self.sp.bos_id()] + ids
            pieces = ["<s>"] + pieces
            offsets = [(0, 0)] + offsets
        return ids, pieces, offsets

    def id_to_piece(self, i):
        return self.sp.id_to_piece(i)


class HFTokenizer:
    """HF `tokenizers` (Rust). For Qwen the byte-level pieces are decoded to make them readable; the word-initial space becomes ▁."""

    def __init__(self, spec, name):
        from tokenizers import Tokenizer
        self.name = name
        self.tok = Tokenizer.from_file(spec) if os.path.exists(spec) else Tokenizer.from_pretrained(spec)
        self.src = spec
        self.vocab_size = self.tok.get_vocab_size(with_added_tokens=False)

    def encode(self, text, add_bos=False):
        enc = self.tok.encode(text, add_special_tokens=add_bos)
        pieces = []
        for i, raw in zip(enc.ids, enc.tokens):
            d = self.tok.decode([i])
            if "�" in d:            # a byte fragment that is not valid UTF-8 on its own
                pieces.append(f"<{raw}>")
            else:
                pieces.append(d.replace(" ", "▁") if d.strip() != "" or d == "" else "▁" * len(d))
        return list(enc.ids), pieces, list(enc.offsets)

    def id_to_piece(self, i):
        return self.tok.decode([i])


def load_llama(spec):
    if spec.endswith(".model"):
        return SPTokenizer(spec)
    return HFTokenizer(spec, "llama2")


# ----------------------------------------------------------------------------------------------------------------
def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"written: {path} ({len(rows)} rows)")


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def morph_boundaries(seg):
    """'üst-ü-nde-ki' -> surface 'üstündeki', internal boundaries {3, 4, 7} (word start/end not included)."""
    parts = seg.split("-")
    surface = "".join(parts)
    cuts, pos = set(), 0
    for p in parts[:-1]:
        pos += len(p)
        cuts.add(pos)
    return surface, cuts


def word_in_context(tok, word, initial):
    """The pieces of the word inside a sentence. LLaMA: the dummy prefix always adds ▁ (position does not matter).
    Qwen: no space at the sentence start (in the VLANeXt template the command comes right after <|vision_end|>), in the middle ' ' + word."""
    if tok.name == "llama2" or initial:
        ids, pieces, offs = tok.encode(word)
        shift = 0
    else:
        ids, pieces, offs = tok.encode(" " + word)
        shift = 1
    cuts = set()
    for (s, e) in offs:
        s, e = s - shift, e - shift
        if 0 < s < len(word):
            cuts.add(s)
        if 0 < e < len(word):
            cuts.add(e)
    return ids, pieces, cuts


def align(cuts_morph, cuts_tok):
    hit = len(cuts_morph & cuts_tok)
    return hit, (hit / len(cuts_morph) if cuts_morph else None), (hit / len(cuts_tok) if cuts_tok else None)


def fmt(x):
    return "" if x is None else f"{x:.2f}"


# ----------------------------------------------------------------------------------------------------------------
def analyze(args):
    toks = [load_llama(args.llama), HFTokenizer(args.qwen, "qwen3")]
    for t in toks:
        print(f"[{t.name}] {t.src}  vocab={t.vocab_size}")
    commands = read_csv(args.commands)
    pairs = {r["pair_id"]: r for r in read_csv(args.pairs)}
    os.makedirs(args.out, exist_ok=True)

    # ---- 1) tokens.csv --------------------------------------------------------------------------------------------
    rows = []
    for r in commands:
        ins = r["instruction"]
        words = ins.split()
        for t in toks:
            ids, pieces, _ = t.encode(ins)
            n_space_alone = sum(1 for p in pieces if set(p) == {"▁"})
            n_frag = sum(1 for p in pieces if p.startswith("<") and p.endswith(">") and len(p) > 2)
            if t.name == "llama2":
                prompt_n = len(t.encode(OPENVLA_PROMPT.format(ins=ins.lower()), add_bos=True)[0])  # incl. BOS
                n_action = sum(1 for i in ids if i in LLAMA_ACTION_IDS)
            else:
                prompt_n = len(t.encode(QWEN_TEMPLATE.format(ins=ins))[0])  # incl. 1 image placeholder
                n_action = 0
            rows.append({
                "id": r["id"], "pair_id": r["pair_id"], "block": r["pair_id"][0], "variant": r["variant"], "lang": r["lang"],
                "tokenizer": t.name, "n_tokens": len(ids), "n_words": len(words), "n_chars": len(ins),
                "n_bytes": len(ins.encode("utf-8")), "fertility": round(len(ids) / len(words), 3),
                "bytes_per_token": round(len(ins.encode("utf-8")) / len(ids), 3),
                "n_space_alone": n_space_alone, "n_byte_fragments": n_frag, "n_in_action_range": n_action,
                "prompt_tokens": prompt_n, "pieces": "|".join(p.replace("|", "¦") for p in pieces),
                "ids": " ".join(map(str, ids)),
            })
    write_csv(os.path.join(args.out, "tokens.csv"), rows)

    # ---- 2) rel_span.csv + seg_align.csv + suffix_forms.csv ----------------------------------------------------------
    rel_rows, seg_rows, suffix_forms = [], [], defaultdict(Counter)
    for pid, p in pairs.items():
        surface, mcuts = morph_boundaries(p["rel_seg"])
        assert surface == p["rel_span"], (pid, surface, p["rel_span"])
        last_suffix = p["rel_seg"].split("-")[-1] if "-" in p["rel_seg"] else ""
        rel_words = surface.split()           # 'üst çekmecesindeki' is two words
        for t in toks:
            # multi-word rel_span: word by word, according to its position in the sentence
            pieces_all, tcuts_all, ids_all, base = [], set(), [], 0
            for w in rel_words:
                initial = p["tr_a"].startswith(w) and p["tr_a"].index(w) == 0 and base == 0
                ids, pieces, cuts = word_in_context(t, w, initial)
                pieces_all += pieces
                ids_all += ids
                tcuts_all |= {c + base for c in cuts}
                base += len(w) + 1
            hit, rec, prec = align(mcuts, tcuts_all)
            cut_before_suffix = len(surface) - len(last_suffix)
            suffix_sep = bool(last_suffix) and cut_before_suffix in tcuts_all and \
                not any(cut_before_suffix < c < len(surface) for c in tcuts_all)
            # the token form carrying the last suffix (if the suffix is not separate, the piece it is contained in)
            if last_suffix:
                last_piece = pieces_all[-1]
                suffix_forms[(t.name, last_suffix if last_suffix in ("ki",) else "ne/na/a (DAT)")][last_piece] += 1
            rel_rows.append({
                "pair_id": pid, "block": pid[0], "rel_span": surface, "rel_seg": p["rel_seg"], "n_morphemes": len(mcuts) + 1,
                "tokenizer": t.name, "n_pieces": len(ids_all), "pieces": "|".join(pieces_all), "ids": " ".join(map(str, ids_all)),
                "morph_cuts": len(mcuts), "token_cuts": len(tcuts_all), "cuts_hit": hit, "recall": fmt(rec), "precision": fmt(prec),
                "last_suffix": last_suffix, "last_suffix_separate": "evet" if suffix_sep else ("hayır" if last_suffix else "—"),
            })
        seg_words = p["tr_a_seg"].split()
        for t in toks:
            tm = th = tt = 0
            for wi, sw in enumerate(seg_words):
                surf, mc = morph_boundaries(sw)
                _, _, tc = word_in_context(t, surf, wi == 0)
                h, _, _ = align(mc, tc)
                tm += len(mc)
                th += h
                tt += len(tc)
            seg_rows.append({"pair_id": pid, "block": pid[0], "tokenizer": t.name, "n_words": len(seg_words),
                             "morph_cuts": tm, "token_cuts": tt, "cuts_hit": th,
                             "recall": fmt(th / tm if tm else None), "precision": fmt(th / tt if tt else None)})
    write_csv(os.path.join(args.out, "rel_span.csv"), rel_rows)
    write_csv(os.path.join(args.out, "seg_align.csv"), seg_rows)
    sf_rows = [{"tokenizer": k[0], "suffix": k[1], "token_form": form, "count": n}
               for k, c in sorted(suffix_forms.items()) for form, n in c.most_common()]
    write_csv(os.path.join(args.out, "suffix_forms.csv"), sf_rows)

    # ---- 3) words.csv --------------------------------------------------------------------------------------------------
    vocab = Counter()
    initial_words = set()
    for r in commands:
        ws = r["instruction"].split()
        initial_words.add((r["lang"], ws[0]))
        for w in ws:
            vocab[(r["lang"], w)] += 1
    wrows = []
    for (lang, w), n in sorted(vocab.items()):
        for t in toks:
            ids_c, pieces_c, _ = word_in_context(t, w, False)[0], word_in_context(t, w, False)[1], None
            ids_i, pieces_i, _ = word_in_context(t, w, True)[0], word_in_context(t, w, True)[1], None
            wrows.append({"lang": lang, "word": w, "count": n, "tokenizer": t.name,
                          "n_pieces_mid": len(ids_c), "pieces_mid": "|".join(pieces_c), "ids_mid": " ".join(map(str, ids_c)),
                          "n_pieces_initial": len(ids_i), "pieces_initial": "|".join(pieces_i),
                          "occurs_initial": "evet" if (lang, w) in initial_words else "hayır"})
    write_csv(os.path.join(args.out, "words.csv"), wrows)

    # ---- 4) action_collision.csv -------------------------------------------------------------------------------------
    L = toks[0]
    coll = []
    for r in commands:
        ids, pieces, _ = L.encode(r["instruction"])
        for i, pc in zip(ids, pieces):
            if i in LLAMA_ACTION_IDS:
                coll.append({"id": r["id"], "token_id": i, "piece": pc, "bin": 32000 - i})
    sample = [{"id": "(vocab örneği)", "token_id": i, "piece": L.id_to_piece(i), "bin": 32000 - i}
              for i in list(LLAMA_ACTION_IDS)[:8] + list(LLAMA_ACTION_IDS)[-8:]]
    write_csv(os.path.join(args.out, "action_collision.csv"), coll + sample)

    # ---- 5) crosscheck -------------------------------------------------------------------------------------------------
    cross = None
    if args.jsids:
        js = json.load(open(args.jsids, encoding="utf-8"))
        same, diff = 0, []
        for r in commands:
            hf_ids = L.encode(r["instruction"])[0]
            js_ids = js.get(r["id"])
            if js_ids == hf_ids:
                same += 1
            else:
                diff.append((r["id"], js_ids, hf_ids))
        cross = (same, diff)
        with open(os.path.join(args.out, "crosscheck.md"), "w", encoding="utf-8") as f:
            f.write(f"# llama-tokenizer-js vs Meta tokenizer.model (SentencePiece)\n\n{same}/{len(commands)} commands have identical id sequences.\n")
            for d in diff:
                f.write(f"- {d[0]}: js={d[1]} sp={d[2]}\n")

    write_summary(args, rows, rel_rows, seg_rows, wrows, sf_rows, coll, cross, toks)


def write_summary(args, rows, rel_rows, seg_rows, wrows, sf_rows, coll, cross, toks):
    out = ["# Lecture 22 — tokenization summary (real numbers; `tokenize_set.py`)\n"]
    out.append("Tokenizers: " + "; ".join(f"{t.name}: {os.path.basename(t.src.split()[0])}{' ' + ' '.join(t.src.split()[1:]) if ' ' in t.src else ''} (vocab {t.vocab_size})" for t in toks) + "\n")
    out.append("## Tokens per command (surface form × tokenizer; mean over 20 pairs, min–max)\n")
    out.append("| surface | LLaMA-2 | Qwen3 | ratio L/Q | fertility L | fertility Q |\n|---|---|---|---|---|---|")
    for v in ("en", "tr_a", "tr_b", "tr_ascii", "tr_bare"):
        Lr = [r for r in rows if r["variant"] == v and r["tokenizer"] == "llama2"]
        Qr = [r for r in rows if r["variant"] == v and r["tokenizer"] == "qwen3"]
        ml, mq = mean(r["n_tokens"] for r in Lr), mean(r["n_tokens"] for r in Qr)
        out.append(f"| {v} | {ml:.1f} ({min(r['n_tokens'] for r in Lr)}–{max(r['n_tokens'] for r in Lr)}) | "
                   f"{mq:.1f} ({min(r['n_tokens'] for r in Qr)}–{max(r['n_tokens'] for r in Qr)}) | {ml/mq:.2f} | "
                   f"{mean(r['fertility'] for r in Lr):.2f} | {mean(r['fertility'] for r in Qr):.2f} |")
    out.append("\n## Turkish/English token ratio (same pair, tr_a / en)\n")
    byid = {(r["pair_id"], r["variant"], r["tokenizer"]): r["n_tokens"] for r in rows}
    pids = sorted({r["pair_id"] for r in rows})
    for name in ("llama2", "qwen3"):
        ratios = [byid[(p, "tr_a", name)] / byid[(p, "en", name)] for p in pids]
        out.append(f"- {name}: tr_a/en mean **{mean(ratios):.2f}** (min {min(ratios):.2f}, max {max(ratios):.2f})")
    out.append("\n## Mean tokens per block (tr_a)\n")
    out.append("| block | LLaMA-2 | Qwen3 | en |\n|---|---|---|---|")
    for b in "ABCD":
        bl = [byid[(p, "tr_a", "llama2")] for p in pids if p[0] == b]
        bq = [byid[(p, "tr_a", "qwen3")] for p in pids if p[0] == b]
        be = [byid[(p, "en", "llama2")] for p in pids if p[0] == b]
        out.append(f"| {b} | {mean(bl):.1f} | {mean(bq):.1f} | {mean(be):.1f} |")
    out.append("\n## Byte fragments / standalone whitespace / action range (total over TR rows)\n")
    for name in ("llama2", "qwen3"):
        TR = [r for r in rows if r["lang"] == "tr" and r["tokenizer"] == name]
        out.append(f"- {name}: byte fragments {sum(r['n_byte_fragments'] for r in TR)}, standalone whitespace (▁) {sum(r['n_space_alone'] for r in TR)}, "
                   f"ids falling in the action range {sum(r['n_in_action_range'] for r in TR)}  ({len(TR)} rows)")
    out.append("\n## Total tokens inside the template\n")
    for name, label in (("llama2", "OpenVLA prompt, incl. BOS"), ("qwen3", "text part of the Qwen chat template (+1 image placeholder; a real image is 64 tokens/frame)")):
        out.append(f"**{name}** — {label}:")
        for v in ("en", "tr_a", "tr_b", "tr_ascii", "tr_bare"):
            Pn = [r["prompt_tokens"] for r in rows if r["variant"] == v and r["tokenizer"] == name]
            out.append(f"- {v}: mean {mean(Pn):.1f} (min {min(Pn)}, max {max(Pn)})")
    out.append("\n## Pieces of the relation word (`rel_span`) inside the sentence\n")
    out.append("| pair | word | segmentation | LLaMA-2 | Qwen3 | last suffix separate (L/Q) | boundary recall (L/Q) |\n|---|---|---|---|---|---|---|")
    byp = defaultdict(dict)
    for r in rel_rows:
        byp[r["pair_id"]][r["tokenizer"]] = r
    for pid in sorted(byp):
        Lr, Qr = byp[pid]["llama2"], byp[pid]["qwen3"]
        out.append(f"| {pid} | {Lr['rel_span']} | {Lr['rel_seg']} | `{Lr['pieces']}` ({Lr['n_pieces']}) | `{Qr['pieces']}` ({Qr['n_pieces']}) | "
                   f"{Lr['last_suffix_separate']} / {Qr['last_suffix_separate']} | {Lr['recall']} / {Qr['recall']} |")
    out.append("\n## Morpheme boundary <-> token boundary (all tr_a commands, total over 20 pairs)\n")
    for name in ("llama2", "qwen3"):
        S = [r for r in seg_rows if r["tokenizer"] == name]
        tm, th, tt = sum(r["morph_cuts"] for r in S), sum(r["cuts_hit"] for r in S), sum(r["token_cuts"] for r in S)
        out.append(f"- {name}: {th} of the {tm} morpheme boundaries coincide with a token boundary (recall {th/tm:.2f}); "
                   f"{th} of the {tt} within-word token boundaries are morpheme boundaries (precision {th/tt:.2f})")
    out.append("\n## Token forms carrying the relation suffix (how many distinct forms?)\n")
    out.append("| tokenizer | suffix | token forms (count) |\n|---|---|---|")
    grp = defaultdict(list)
    for r in sf_rows:
        grp[(r["tokenizer"], r["suffix"])].append(f"`{r['token_form']}`×{r['count']}")
    for k in sorted(grp):
        out.append(f"| {k[0]} | -{k[1]} | {', '.join(grp[k])} -> **{len(grp[k])} forms** |")
    out.append("\n## Selected words (sentence-medial form)\n")
    pick = ["sağına", "soluna", "sağdaki", "soldaki", "öndeki", "üstündeki", "üstüne", "ustune", "ustundeki", "içine", "icine",
            "arasındaki", "arasindaki", "yanındaki", "ortasındaki", "kase", "kâse", "kaseyi", "kâseyi", "koy", "al", "it", "kaldır",
            "ramekinin", "tabağın", "tabagin", "ocağın", "dolabın", "çekmecesindeki", "cekmecesindeki", "rafa", "robota", "duran", "alıp"]
    out.append("| word | LLaMA-2 | Qwen3 |\n|---|---|---|")
    W = defaultdict(dict)
    for r in wrows:
        W[r["word"]][r["tokenizer"]] = r
    for w in pick:
        if w in W:
            out.append(f"| {w} | `{W[w]['llama2']['pieces_mid']}` ({W[w]['llama2']['n_pieces_mid']}) | `{W[w]['qwen3']['pieces_mid']}` ({W[w]['qwen3']['n_pieces_mid']}) |")
    out.append("\n## Qwen3: sentence-initial <-> sentence-medial difference (TR words whose piece count changes)\n")
    ch = [(r["word"], r["pieces_initial"], r["pieces_mid"]) for r in wrows if r["tokenizer"] == "qwen3" and r["lang"] == "tr" and r["n_pieces_initial"] != r["n_pieces_mid"]]
    out.append(f"- {len(ch)} words; examples: " + "; ".join(f"{w}: initial `{a}` / medial `{b}`" for w, a, b in ch[:8]))
    out.append("\n## Overlap with the action range (LLaMA-2 ids 31744–31999)\n")
    out.append(f"- tokens falling in the action range over 100 commands: **{len(coll)}**; sample contents of the range: " +
               ", ".join(L.id_to_piece(i) for L in [toks[0]] for i in list(LLAMA_ACTION_IDS)[:6]))
    if cross:
        out.append(f"\n## llama-tokenizer-js cross-check\n- {cross[0]}/100 commands have an id sequence identical to SentencePiece; differences: {len(cross[1])}")
    txt = "\n".join(out) + "\n"
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commands", default=os.path.join(DATA, "commands.csv"))
    ap.add_argument("--pairs", default=os.path.join(DATA, "pairs.csv"))
    ap.add_argument("--llama", default="openvla/openvla-7b", help="path to Meta tokenizer.model (SentencePiece) or tokenizer.json / HF id")
    ap.add_argument("--qwen", default="Qwen/Qwen3-VL-2B-Instruct", help="path to tokenizer.json or HF id")
    ap.add_argument("--jsids", help="output of count_js_ids.mjs — Lecture 15 cross-check")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "datasets", "05-tokenization-tr-commands", "data"))
    analyze(ap.parse_args())
