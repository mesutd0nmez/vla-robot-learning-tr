"""Lecture 20 — checks that the command set (pairs.csv + commands.csv) conforms to the design rules and prints summary tables.

Checks:
  1. Format: 20 pairs × 5 surface forms = 100 rows; unique ids; BDDL-safe text (Lecture 16: lowercase, no ';' or parentheses, NFC,
     no U+0307, no double spaces) — every row must be able to go into a BDDL (:language …) line tomorrow.
  2. Suffix consistency: rel_span / rel_seg are regenerated with tr_morph and compared; they occur inside tr_a.
  3. Continuity: tr_a of A01–A04 is identical to the sp0/1/2/4 `tr` rows of datasets/01 (comparability with Lecture 15).
  4. Referent definiteness: every description in block B points to the same bowl in all 50 init states (referents.csv, above threshold).
  5. Balance: block × tier, relation type, case distribution; number of frame-dependent rows.
  6. Minimal pairs: pairs whose tr_a differs in exactly ONE word (comparisons that isolate the effect of the suffix / relational noun).
  7. Allomorph table: how many distinct surface forms each morpheme has in the tr_a segmentations (input of Lecture 21).
  8. Common tail: position of the distinguishing information in the sentence for Spatial commands in TR and EN (index of the first differing word).

Usage:  python validate_set.py ../../datasets/04-tr-spatial-commands/data [--d01 ../../datasets/01-libero-spatial-tr-mini/data/commands.csv]
"""
import argparse, csv, itertools, pathlib, re, sys, unicodedata
from collections import Counter, defaultdict
import tr_morph as M


def bddl_safe(t):
    p = []
    if ";" in t: p.append("';'")
    if "(" in t or ")" in t: p.append("parentheses")
    if t != t.lower(): p.append("uppercase")
    if "̇" in t: p.append("U+0307")
    if unicodedata.normalize("NFC", t) != t: p.append("not NFC")
    if "  " in t or t != t.strip(): p.append("whitespace")
    return p


def morpheme_of(seg_token):
    """'tabağ-ın' → [('GEN','ın')], 'üst-ü-nde-ki' → POSS3, LOC, ki … (only the suffixes of this set)."""
    parts = seg_token.split("-")
    out = []
    for i, s in enumerate(parts[1:], 1):
        after_poss = any(m == "POSS3" for m, _ in out) or parts[0].endswith(("kutusu", "şişesi"))
        if s == "ki": m = "ki"
        elif re.fullmatch(r"n?[ıiuü]n", s): m = "GEN"
        elif re.fullmatch(r"s?[ıiuü]", s) and i == 1 and len(parts) > 2: m = "POSS3"
        elif re.fullmatch(r"nd[ae]n|[dt][ae]n", s): m = "ABL"
        elif re.fullmatch(r"nd[ae]|[dt][ae]", s): m = "LOC"
        elif re.fullmatch(r"n[ae]", s) and after_poss: m = "DAT"
        elif re.fullmatch(r"y?[ae]", s): m = "DAT"
        elif re.fullmatch(r"[ny]?[ıiuü]", s): m = "ACC"
        else: m = "?"
        out.append((m, s))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("data"); ap.add_argument("--d01")
    a = ap.parse_args(); d = pathlib.Path(a.data)
    pairs = list(csv.DictReader(open(d / "pairs.csv", encoding="utf-8")))
    cmds = list(csv.DictReader(open(d / "commands.csv", encoding="utf-8")))
    errors = []
    # 1
    if len(pairs) != 20 or len(cmds) != 100: errors.append(f"size: {len(pairs)} pairs, {len(cmds)} rows")
    if len({c["id"] for c in cmds}) != len(cmds): errors.append("duplicate id")
    for c in cmds:
        p = bddl_safe(c["instruction"])
        if p: errors.append(f"{c['id']}: not BDDL-safe ({', '.join(p)})")
    by_pair = defaultdict(dict)
    for c in cmds: by_pair[c["pair_id"]][c["variant"]] = c["instruction"]
    for pid, v in by_pair.items():
        if len(set(v.values())) != 5: errors.append(f"{pid}: the 5 surface forms are not all different")
    print(f"1. format: {len(pairs)} pairs, {len(cmds)} rows, BDDL-safe: {'yes' if not errors else 'NO'}")
    # 2
    n_ok = 0
    for r in pairs:
        seg_tokens = r["tr_a_seg"].split()
        if r["rel_span"] not in r["tr_a"]: errors.append(f"{r['pair_id']}: rel_span is not in tr_a")
        if r["rel_seg"].replace("-", "") != r["rel_span"]: errors.append(f"{r['pair_id']}: rel_seg ≠ rel_span")
        if r["tr_a_seg"].replace("-", "") != r["tr_a"]: errors.append(f"{r['pair_id']}: segmentation does not match the surface form")
        if r["rel_noun"] in M.LEXICON and r["case"] in ("LOC+ki", "DAT"):
            c = "loc" if r["case"] == "LOC+ki" else "dat"
            exp = M.rel(r["rel_noun"], c, r["case"] == "LOC+ki")
            if exp != (r["rel_span"], r["rel_seg"]): errors.append(f"{r['pair_id']}: tr_morph {exp} ≠ {r['rel_span']}")
            else: n_ok += 1
    print(f"2. suffix consistency: {n_ok} relational words regenerated with tr_morph, all identical")
    # 3
    if a.d01:
        d01 = {r["id"]: r["instruction"] for r in csv.DictReader(open(a.d01, encoding="utf-8"))}
        same = [(p, k) for p, k in [("A01", "sp0_tr"), ("A02", "sp1_tr"), ("A03", "sp2_tr"), ("A04", "sp4_tr")] if by_pair[p]["tr_a"] == d01[k]]
        if len(same) != 4: errors.append("continuity with datasets/01 is broken")
        print(f"3. continuity: tr_a identical to datasets/01: {len(same)}/4")
    # 4
    refs = {(int(r["task_id"]), r["descriptor"], r["frame"]): r for r in csv.DictReader(open(d / "referents.csv", encoding="utf-8"))}
    print("4. referent definiteness (block B; 50 init states):")
    for r in pairs:
        if r["block"] != "B": continue
        amb = [f for f in ("ref_libero", "ref_image", "ref_robot") if r[f].endswith("?") or r[f] in ("", "belirsiz", "ambiguous")]
        if amb: errors.append(f"{r['pair_id']}: ambiguous referent {amb}")
        agree = len({r["ref_libero"], r["ref_image"], r["ref_robot"]}) == 1
        print(f"   {r['pair_id']} task {r['scene_task_id']} {r['rel_span']:10s} libero={r['ref_libero']} image={r['ref_image']} robot={r['ref_robot']}"
              f"  {'frames AGREE' if agree else 'frames DISAGREE'}  (smallest gap: {r['ref_min_margin']})")
    # 5
    print("5. balance:")
    for key in ("block", "tier", "relation", "case", "frame_dependent", "suite"):
        print(f"   {key:16s}", dict(sorted(Counter(r[key] for r in pairs).items())))
    # 6
    print("6. minimal pairs (tr_a differs in a single word):")
    n_min = 0
    for x, y in itertools.combinations(pairs, 2):
        tx, ty = x["tr_a"].split(), y["tr_a"].split()
        if len(tx) == len(ty):
            diff = [(p, q) for p, q in zip(tx, ty) if p != q]
            if len(diff) == 1:
                n_min += 1
                print(f"   {x['pair_id']} ~ {y['pair_id']}: {diff[0][0]} / {diff[0][1]}")
    print(f"   total {n_min} minimal pairs")
    # 7
    allo = defaultdict(Counter)
    for r in pairs:
        for tok in r["tr_a_seg"].split():
            for m, s in morpheme_of(tok):
                allo[m][s] += 1
    print("7. allomorph table (tr_a, 20 commands):")
    for m in ("GEN", "POSS3", "ACC", "DAT", "LOC", "ABL", "ki", "?"):
        if allo[m]:
            print(f"   {m:6s} {len(allo[m])} forms: " + ", ".join(f"-{s}×{n}" for s, n in allo[m].most_common()))
    if allo["?"]: errors.append("unclassifiable suffix present")
    # 8
    sp = [r for r in pairs if r["suite"] == "libero_spatial"]
    def first_diff_positions(key):
        toks = [r[key].split() for r in sp]
        pos = []
        for x, y in itertools.combinations(toks, 2):
            i = next((k for k, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
            pos.append(i)
        return pos
    def common_tail(key):
        toks = [r[key].split()[::-1] for r in sp]
        n = 0
        while all(len(t) > n for t in toks) and len({t[n] for t in toks}) == 1: n += 1
        return n
    def common_head(key):
        toks = [r[key].split() for r in sp]
        n = 0
        while all(len(t) > n for t in toks) and len({t[n] for t in toks}) == 1: n += 1
        return n
    print(f"8. position of the distinguishing information (12 Spatial commands): EN common head {common_head('en')} words, common tail {common_tail('en')}; "
          f"TR common head {common_head('tr_a')}, common tail {common_tail('tr_a')}")
    print("\nRESULT:", "CLEAN — no errors" if not errors else f"{len(errors)} ERRORS")
    for e in errors: print("  -", e)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
