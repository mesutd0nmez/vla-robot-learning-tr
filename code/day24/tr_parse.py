"""Lecture 24 — the cheapest form of solution candidate 4: rule-based Turkish command -> structured frame -> English LIBERO template.

Idea (Lecture 6 / Lecture 23: "LLM parser + symbolic solver"): before the Turkish command is ever shown to the model, split it into a frame of
  (verb, target object, relation, reference(s), destination), then generate from the frame the English template LIBERO saw during training.
This script does not even use an LLM: a ~25-word lexicon + suffix stripping + 3 templates.
The goal is NOT to write a translation system; it is to give a REAL number to the question "how cheap can a translation layer be and where does it break"
on the 80 Turkish rows of datasets/04 (success = the generated English is identical to the `en` column).

Usage:
  python tr_parse.py                                   # datasets/04 commands.csv -> summary table + parse_roundtrip.csv
  python tr_parse.py --cmd "soldaki siyah kaseyi al ve tabağın üstüne koy"
  python tr_parse.py --selftest
"""
import argparse, csv, json, re, sys, unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / ".." / ".." / "datasets" / "04-tr-spatial-commands" / "data" / "commands.csv"

# ---------------------------------------------------------------- lexicon (stems folded to ASCII)
ASCII = str.maketrans("çğıöşüâîû", "cgiosuaiu")


def fold(w):
    """Fold Turkish letters to ASCII, NFC; so that the `tr_ascii` surface form and `tr_a` fall on the same key."""
    return unicodedata.normalize("NFC", w).lower().translate(ASCII)


OBJECTS = {  # stem -> English noun
    "kase": "bowl", "tabak": "plate", "ramekin": "ramekin", "masa": "table", "dolap": "cabinet", "cekmece": "drawer",
    "ocak": "stove", "raf": "rack", "sise": "bottle", "peynir": "cheese", "robot": "robot",
}
SOFT = {"tabag": "tabak", "dolab": "dolap", "ocag": "ocak"}            # undoing consonant softening
ADJ = {"siyah": "black", "ahsap": "wooden", "ust": "top", "sarap": "wine", "krem": "cream"}
RELATIONS = {  # relational noun stem -> (English selection phrase, English destination phrase)
    "ara": ("between {refs}", None), "yan": ("next to {ref}", None), "orta": ("from {ref} center", None),
    "ust": ("on {ref}", "on {ref}"), "uzer": ("on {ref}", "on {ref}"), "ic": ("in {ref}", "in {ref}"),
    "sol": ("on the left", "to the left of {ref}"), "sag": ("on the right", "to the right of {ref}"),
    "on": ("in the front", "to the front of {ref}"),
    "yakin": ("closer to {ref}", None), "uzak": ("farther from {ref}", None),
}
VERBS = {"al": "pick", "alip": "pick", "koy": "put", "yerlestir": "put", "it": "push"}
STOP = {"ve", "ile", "daha", "olan", "duran", "dogru", "taraftaki", "tarafina", "tarafindaki", "-"}
# suffix stripping: the longest matching stem + an allowed suffix sequence
SUFFIX = re.compile(r"^(l[ae]|n?[iu]n|[sy]?[iu]|[yn]?[ae]|n?d[ae]n?|t[ae]n?|ki|n)*$")  # allowed suffix sequence (ASCII)
STEMS = sorted(set(OBJECTS) | set(SOFT) | set(ADJ) | set(RELATIONS) | set(VERBS), key=len, reverse=True)


def analyze(word):
    """Reduce a word to a (kind, stem, case) triple. Case ∈ {acc, dat, loc, abl, ins, gen, ki, bare}."""
    w = fold(word)
    if w in STOP:
        return ("STOP", w, "bare")
    for stem in STEMS:  # the longest stem first (ust > us; sarap > sag...)
        if w.startswith(stem) and SUFFIX.match(w[len(stem):]):
            suf = w[len(stem):]
            stem = SOFT.get(stem, stem)
            if stem in VERBS and suf == "":
                return ("VERB", stem, "bare")
            if stem in OBJECTS:
                return ("OBJ", stem, case_of(suf))
            if stem in RELATIONS and (suf != "" or stem not in ADJ):
                return ("REL", stem, case_of(suf))
            if stem in ADJ:
                return ("ADJ", stem, "bare")
    return ("UNK", w, "bare")


def case_of(suf):
    if suf.endswith("ki"):
        return "ki"
    if re.search(r"(d|t)[ae]n$", suf):
        return "abl"
    if re.search(r"n?d[ae]$", suf) or suf.endswith("nda") or suf.endswith("nde"):
        return "loc"
    if re.search(r"(y|n)?[ae]$", suf) and suf not in ("",):
        return "dat"
    if re.search(r"n?[iıu]n$", suf):
        return "gen"
    if re.search(r"(y|n|s)?[iıu]$", suf) and suf:
        return "acc"
    if suf in ("le", "la"):
        return "ins"
    return "bare"


def noun_phrase(tokens):
    """[ADJ...] OBJ -> English 'the wooden cabinet' / 'the top drawer'."""
    adjs = [ADJ[s] for k, s, _ in tokens if k == "ADJ"]
    objs = [OBJECTS[s] for k, s, _ in tokens if k == "OBJ"]
    if not objs:
        return None
    if objs == ["cheese"] and "cream" in adjs:
        return "the cream cheese"
    if objs == ["bottle"] and "wine" in adjs:
        return "the wine bottle"
    return "the " + " ".join(adjs + [objs[-1]])


def parse(cmd):
    toks = [analyze(w) for w in re.findall(r"[^\s]+", cmd.replace("'", " "))]
    # bare "üst": an adjective if "çekmece" follows (top drawer), otherwise a relational noun (tr_bare: "tabak üst koy")
    for i, t in enumerate(toks):
        if t == ("ADJ", "ust", "bare") and not (i + 1 < len(toks) and toks[i + 1][:2] == ("OBJ", "cekmece")):
            toks[i] = ("REL", "ust", "bare")
    frame = {"verb": None, "target": None, "relation": None, "refs": [], "dest": None, "dest_rel": None, "tokens": toks}
    verbs = [s for k, s, _ in toks if k == "VERB"]
    if not verbs:
        return frame
    frame["verb"] = VERBS[verbs[0]] if verbs[0] != "koy" or len(verbs) == 1 else "pick"
    if "it" in verbs and len(verbs) == 1:
        frame["verb"] = "push"
    # --- selection relation: the first REL (the one before the destination relation) ---
    rel_idx = [i for i, (k, s, c) in enumerate(toks) if k == "REL"]
    obj_idx = [i for i, (k, s, c) in enumerate(toks) if k == "OBJ"]
    if frame["verb"] == "pick":  # Spatial template: ... siyah kase ... al ve tabak üst koy
        first_verb = next(i for i, (k, s, c) in enumerate(toks) if k == "VERB")
        before = toks[:first_verb]
        after = toks[first_verb + 1:]
        # target: the last OBJ group before "al" (kase)
        # target: the OBJ that took the accusative (-yi) before the verb; otherwise (tr_bare) the last OBJ before the verb
        acc_before = [i for i in obj_idx if i < first_verb and toks[i][2] == "acc"]
        tgt_pos = acc_before[-1] if acc_before else max((i for i in obj_idx if i < first_verb), default=None)
        if tgt_pos is None:
            return frame
        tgt_start = tgt_pos
        while tgt_start - 1 >= 0 and toks[tgt_start - 1][0] == "ADJ" and toks[tgt_start - 1][1] == "siyah":
            tgt_start -= 1
        frame["target"] = noun_phrase(toks[tgt_start:tgt_pos + 1])
        rels_sel = [i for i in rel_idx if i < first_verb]          # relational nouns before the verb (selection)
        # reference groups: [ADJ...] OBJ sequences before the verb that lie outside the target group
        groups, cur = [], []
        for i, t in enumerate(toks[:first_verb]):
            if tgt_start <= i <= tgt_pos or t[0] not in ("OBJ", "ADJ"):
                continue
            cur.append(t)
            if t[0] == "OBJ":
                groups.append(cur); cur = []
        frame["refs"] = [noun_phrase(g) for g in groups]
        if rels_sel:
            frame["relation"] = toks[rels_sel[-1]][1]
        elif groups:  # no relational noun, bare location: "dolabın üst çekmecesindeki" -> in the top drawer of the cabinet
            frame["relation"] = "ic"
        dest_objs = [i for i in obj_idx if i > first_verb]
        dest_rels = [i for i in rel_idx if i > first_verb]
        if dest_objs:
            frame["dest"] = noun_phrase([toks[dest_objs[0]]])
            frame["dest_rel"] = toks[dest_rels[0]][1] if dest_rels else "ust"
    else:  # Goal template: X-i Y-nin üst-ü-ne koy / Y-nin üst-ü-ne X-i koy
        acc = [i for i in obj_idx if toks[i][2] == "acc"]
        rel = rel_idx[0] if rel_idx else None
        if acc:
            tpos = acc[0]
        else:  # tr_bare: the first OBJ is the target
            tpos = obj_idx[0] if obj_idx else None
        if tpos is None:
            return frame
        tstart = tpos
        while tstart - 1 >= 0 and toks[tstart - 1][0] == "ADJ":
            tstart -= 1
        frame["target"] = noun_phrase(toks[tstart:tpos + 1])
        others = [i for i in obj_idx if not (tstart <= i <= tpos)]
        if others:
            frame["dest"] = noun_phrase([toks[others[0]]])
        frame["dest_rel"] = toks[rel][1] if rel is not None else None
        if frame["dest_rel"] is None and frame["dest"]:
            # "şarap şişesini rafa koy": no relational noun -> DAT = on
            frame["dest_rel"] = "ust"
    return frame


def render(frame):
    """The LIBERO English template from the frame."""
    if not frame["verb"] or not frame["target"]:
        return None
    if frame["verb"] == "pick":
        rel = frame["relation"]
        if rel is None:
            return None
        sel, _ = RELATIONS[rel]
        refs = frame["refs"]
        if rel == "ara":
            if len(refs) != 2:
                return None
            sel = sel.format(refs=f"{refs[0]} and {refs[1]}")
        elif rel == "ic":
            if len(refs) == 2:   # top drawer of the wooden cabinet
                sel = f"in {refs[1]} of {refs[0]}"
            elif len(refs) == 1:
                sel = f"in {refs[0]}"
            else:
                return None
        elif rel == "orta":
            sel = "from table center" if refs and "table" in refs[0] else None
            if sel is None:
                return None
        elif "{ref}" in sel:
            if not refs:
                return None
            sel = sel.format(ref=refs[-1])
        dest = frame["dest"] or "the plate"
        return f"pick up {frame['target']} {sel} and place it on {dest}"
    if frame["verb"] == "put":
        if not frame["dest"] or not frame["dest_rel"]:
            return None
        _, d = RELATIONS[frame["dest_rel"]]
        if d is None:
            return None
        d = d.format(ref=frame["dest"])
        if frame["dest_rel"] == "ust" and frame["dest"] == "the cabinet":
            d = "on top of the cabinet"  # that is the name of LIBERO-Goal task 4
        return f"put {frame['target']} {d}"
    if frame["verb"] == "push":
        if not frame["dest"] or not frame["dest_rel"]:
            return None
        _, d = RELATIONS[frame["dest_rel"]]
        return f"push {frame['target']} {d.format(ref=frame['dest'])}"
    return None


def run_set(csv_path, out_path):
    rows = [r for r in csv.DictReader(open(csv_path, encoding="utf-8")) if r["lang"] == "tr"]
    en = {r["pair_id"]: r["instruction"] for r in csv.DictReader(open(csv_path, encoding="utf-8")) if r["lang"] == "en"}
    out, stats = [], {}
    for r in rows:
        fr = parse(r["instruction"])
        pred = render(fr)
        gold = en[r["pair_id"]]
        ok = pred == gold
        kind = "ok" if ok else ("parse_fail" if pred is None else "wrong_en")
        stats.setdefault(r["variant"], {"ok": 0, "parse_fail": 0, "wrong_en": 0})[kind] += 1
        out.append({"id": r["id"], "pair_id": r["pair_id"], "variant": r["variant"], "tr": r["instruction"],
                    "frame": json.dumps({k: v for k, v in fr.items() if k != "tokens"}, ensure_ascii=False),
                    "en_pred": pred or "", "en_gold": gold, "result": kind})
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print(f"{'surface':<9}{'n':>4}{'exact':>9}{'wrong EN':>11}{'no parse':>11}")
    for v in ("tr_a", "tr_ascii", "tr_bare", "tr_b"):
        s = stats.get(v, {})
        n = sum(s.values())
        print(f"{v:<9}{n:>4}{s.get('ok',0):>9}{s.get('wrong_en',0):>11}{s.get('parse_fail',0):>11}")
    tot = sum(sum(s.values()) for s in stats.values()); ok = sum(s["ok"] for s in stats.values())
    print(f"{'total':<9}{tot:>4}{ok:>9}   ({100*ok/tot:.0f}%)   -> {out_path}")
    fails = [o for o in out if o["result"] != "ok"]
    if fails:
        print("\nFailures:")
        for o in fails:
            print(f"  {o['id']:<14} {o['tr']}\n      → {o['en_pred'] or '∅'}\n      ≠ {o['en_gold']}")
    return stats


def run_heldout(path):
    """12 new Turkish surface forms written AFTER the parser was frozen: what happens outside the lexicon/templates?"""
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    ok = 0
    print(f"\nHeld-out ({path.name}):")
    for r in rows:
        pred = render(parse(r["tr"]))
        hit = pred == r["en_gold"]
        ok += hit
        print(f"  [{'OK ' if hit else 'X  '}] {r['id']} {r['tr']}\n        → {pred or '∅'}" + ("" if hit else f"\n        ≠ {r['en_gold']}   ({r['note']})"))
    print(f"  held-out: {ok}/{len(rows)}")
    return ok, len(rows)


def selftest():
    cases = {
        "soldaki siyah kaseyi al ve tabağın üstüne koy": "pick up the black bowl on the left and place it on the plate",
        "kaseyi tabağın sağına koy": "put the bowl to the right of the plate",
        "tabağı ocağın önüne it": "push the plate to the front of the stove",
        "ramekinin yanindaki siyah kaseyi al ve tabagin ustune koy": "pick up the black bowl next to the ramekin and place it on the plate",
    }
    bad = 0
    for tr, en in cases.items():
        got = render(parse(tr))
        flag = "OK " if got == en else "FAIL"
        bad += got != en
        print(f"[{flag}] {tr}\n       {got}")
    print(f"selftest: {len(cases)-bad}/{len(cases)}")
    return bad == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--out", default=str(HERE / "parse_roundtrip.csv"))
    ap.add_argument("--cmd")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.cmd:
        fr = parse(a.cmd)
        print(json.dumps({k: v for k, v in fr.items()}, ensure_ascii=False, indent=1))
        print("→", render(fr))
        sys.exit(0)
    run_set(a.csv, a.out)
    if (HERE / "heldout_tr.csv").exists():
        run_heldout(HERE / "heldout_tr.csv")
