"""Lecture 20 — generates the 20-pair EN–TR spatial command set: datasets/04-tr-spatial-commands/data/{pairs,commands}.csv

Design (Section 5 of the lecture):
  A (6)  the TRAINED commands of LIBERO-Spatial — selection: "X'in REL-POSS-LOC-ki siyah kaseyi al ..." (take the black bowl that is REL of X ...)
  B (6)  descriptions in the same Spatial scenes that match NO training sentence (left/right/front, near/far from the robot/plate);
         which bowl is meant is read from referents.csv, which scene_frames.py computes from the 50 init states
  C (6)  the trained commands of LIBERO-Goal — placement: "... X'in REL-POSS-DAT koy" (put ... on REL of X)
  D (2)  untrained placement (to the right / to the left): no predicate -> today only the language side (Lecture 21 tokenization)

5 surface forms per pair: en | tr_a (GENERATED with the suffix generator, with its segmentation) | tr_b (a second, hand-written form) |
  tr_ascii (tr_a without Turkish characters) | tr_bare (bag of words with the suffixes stripped)   -> 20 × 5 = 100 rows.

Note: the cell values written to pairs.csv / commands.csv (relation, evet/hayır, note, ...) are DATA and deliberately stay Turkish.

Usage:
  python make_tr_set.py --out ../../datasets/04-tr-spatial-commands/data [--referents .../referents.csv] [--libero ~/src/LIBERO]
  If --libero is given, en_seen_* is filled in: whether the EN command occurs among LIBERO's 130 task names.
"""
import argparse, csv, pathlib, re
import tr_morph as M

TR2ASCII = str.maketrans("çğıöşüâîû", "cgiosuaiu")


def N(lemma, c="nom", poss=False):
    """Noun + case → (surface, segmentation, bare). poss=True: lemma is already a possessive compound (kurabiye kutusu, şarap şişesi)."""
    s, seg = M.case(lemma, c, poss=poss)
    return (s, seg, lemma)


def R(noun, c, ki=False):
    s, seg = M.rel(noun, c, ki)
    return (s, seg, noun)


def B(noun, c, ki=False):
    s, seg = M.bare(noun, c, ki)
    return (s, seg, noun)


def W(*words):
    return [(w, w, w) for w in words]


def POSSN(owner_gen, head, c, ki=False):
    """Possessed head of a definite compound: (dolab-ın) üst çekmece-si-nde-ki."""
    p, pseg = M.poss3(head)
    s, seg = M.case(p, c, poss=True, seg=pseg)
    if ki:
        s, seg = M.ki(s, seg)
    return (s, seg, head)


TABAK_USTUNE = [N("tabak", "gen"), R("üst", "dat")]
SEL_TAIL = W("siyah") + [N("kase", "acc")] + W("al", "ve") + TABAK_USTUNE + W("koy")   # siyah kaseyi al ve tabağın üstüne koy
BASE = {0: "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        1: "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
        2: "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        4: "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
        7: "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        8: "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
        9: "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"}
PICK = "pick up the black bowl {} and place it on the plate"

# (id, block, tier, suite, scene task, relation, relational noun, case, frame-dependent?, EN, tr_a parts, part index of the relation word,
#  part index of the reference noun (None if absent), tr_b, descriptor (referents.csv key or None), goal predicate, note)
P = [
 ("A01", "A", 1, "libero_spatial", 0, "2 referans", "ara", "LOC+ki", "hayır", PICK.format("between the plate and the ramekin"),
  [N("tabak")] + W("ile") + [N("ramekin", "gen"), R("ara", "loc", True)] + SEL_TAIL, 3, 2,
  "tabakla ramekin arasında duran siyah kâseyi alıp tabağa koy", None, "(On akita_black_bowl_1 plate_1)", "eğitim komutu; = datasets/01 sp0_tr"),
 ("A02", "A", 1, "libero_spatial", 1, "mesafe", "yan", "LOC+ki", "hayır", PICK.format("next to the ramekin"),
  [N("ramekin", "gen"), R("yan", "loc", True)] + SEL_TAIL, 1, 0,
  "ramekinin yanında duran siyah kâseyi alıp tabağa koy", None, "(On akita_black_bowl_1 plate_1)", "eğitim komutu; = sp1_tr"),
 ("A03", "A", 1, "libero_spatial", 2, "bölge", "orta", "LOC+ki", "hayır", PICK.format("from table center"),
  [N("masa", "gen"), R("orta", "loc", True)] + SEL_TAIL, 1, 0,
  "siyah kâseyi masanın ortasından alıp tabağa koy", None, "(On akita_black_bowl_1 plate_1)", "EN 'from' = kaynak; tr_b ayrılma hâli (ABL) kullanır; = sp2_tr"),
 ("A04", "A", 1, "libero_spatial", 4, "NTPP (içinde)", "—", "LOC+ki", "hayır", PICK.format("in the top drawer of the wooden cabinet"),
  W("ahşap") + [N("dolap", "gen"), POSSN(None, "üst çekmece", "loc", True)] + SEL_TAIL, 2, 1,
  "ahşap dolabın üst çekmecesinin içindeki siyah kâseyi alıp tabağa koy", None, "(On akita_black_bowl_1 plate_1)", "EN 'in' → tr_a'da ilişki adı YOK (yalın bulunma hâli); tr_b'de iç-i-nde-ki; = sp4_tr"),
 ("A05", "A", 1, "libero_spatial", 7, "EC (üstünde)", "üst", "LOC+ki", "hayır", PICK.format("on the stove"),
  [N("ocak", "gen"), R("üst", "loc", True)] + SEL_TAIL, 1, 0,
  "ocağın üzerinde duran siyah kâseyi alıp tabağa koy", None, "(On akita_black_bowl_1 plate_1)", "görev 9 ile AYNI sahne: yalnızca komut ayırır (Ders 19)"),
 ("A06", "A", 1, "libero_spatial", 9, "EC (üstünde)", "üst", "LOC+ki", "hayır", PICK.format("on the wooden cabinet"),
  W("ahşap") + [N("dolap", "gen"), R("üst", "loc", True)] + SEL_TAIL, 2, 1,
  "ahşap dolabın üzerinde duran siyah kâseyi alıp tabağa koy", None, "(On akita_black_bowl_1 plate_1)", "görev 7 ile AYNI sahne"),
 ("B07", "B", 2, "libero_spatial", 1, "yön (sol)", "sol", "LOC+ki (iyeliksiz)", "evet", PICK.format("on the left"),
  [B("sol", "loc", True)] + SEL_TAIL, 0, None,
  "sol taraftaki siyah kâseyi alıp tabağa koy", "left", "çerçeveye göre bowl_1 ya da bowl_2", "deiktik: referans nesne yok, çerçeve örtük"),
 ("B08", "B", 2, "libero_spatial", 1, "yön (sağ)", "sağ", "LOC+ki (iyeliksiz)", "evet", PICK.format("on the right"),
  [B("sağ", "loc", True)] + SEL_TAIL, 0, None,
  "sağ taraftaki siyah kâseyi alıp tabağa koy", "right", "çerçeveye göre bowl_1 ya da bowl_2", "B07'nin minimal çifti (tek sözcük)"),
 ("B09", "B", 2, "libero_spatial", 8, "yön (ön)", "ön", "LOC+ki (iyeliksiz)", "evet", PICK.format("in the front"),
  [B("ön", "loc", True)] + SEL_TAIL, 0, None,
  "ön taraftaki siyah kâseyi alıp tabağa koy", "front", "çerçeveye göre bowl_1 ya da bowl_2", "ön/arka: görüntü ile robot çerçevesi ZIT"),
 ("B10", "B", 2, "libero_spatial", 8, "mesafe (açık çerçeve)", "yakın", "DAT + sıfat", "hayır", PICK.format("closer to the robot"),
  [N("robot", "dat")] + W("daha", "yakın", "olan") + SEL_TAIL, 2, 0,
  "robota yakın duran siyah kâseyi alıp tabağa koy", "near_robot", "(On akita_black_bowl_2 plate_1)", "B09'un çerçevesi açık yazılmış hâli; karşı-olgusal hedef"),
 ("B11", "B", 2, "libero_spatial", 0, "mesafe", "uzak", "DAT + sıfat", "hayır", PICK.format("farther from the plate"),
  [N("tabak", "dat")] + W("daha", "uzak", "olan") + SEL_TAIL, 2, 0,
  "tabaktan uzakta duran siyah kâseyi alıp tabağa koy", "far_plate", "(On akita_black_bowl_2 plate_1)", "Ders 19'un vaadi: eğitim cümlesi olmayan karşı-olgusal tarif"),
 ("B12", "B", 1, "libero_spatial", 0, "mesafe", "yakın", "DAT + sıfat", "hayır", PICK.format("closer to the plate"),
  [N("tabak", "dat")] + W("daha", "yakın", "olan") + SEL_TAIL, 2, 0,
  "tabağın yakınındaki siyah kâseyi alıp tabağa koy", "near_plate", "(On akita_black_bowl_1 plate_1)", "B11'in minimal çifti; hedef eğitimdekiyle aynı kâse"),
 ("C13", "C", 1, "libero_goal", 8, "EC (üstüne)", "üst", "DAT", "hayır", "put the bowl on the plate",
  [N("kase", "acc")] + TABAK_USTUNE + W("koy"), 2, 1,
  "tabağın üzerine kâseyi koy", None, "(On akita_black_bowl_1 plate_1)", "Goal: tek sahne; C14/C15 ile minimal üçlü"),
 ("C14", "C", 1, "libero_goal", 1, "EC (üstüne)", "üst", "DAT", "hayır", "put the bowl on the stove",
  [N("kase", "acc"), N("ocak", "gen"), R("üst", "dat")] + W("koy"), 2, 1,
  "ocağın üzerine kâseyi koy", None, "(On akita_black_bowl_1 flat_stove_1_cook_region)", "Ders 19 conditions_goal8.json'daki wrong komutu"),
 ("C15", "C", 1, "libero_goal", 4, "EC (üstüne)", "üst", "DAT", "hayır", "put the bowl on top of the cabinet",
  [N("kase", "acc"), N("dolap", "gen"), R("üst", "dat")] + W("koy"), 2, 1,
  "dolabın üzerine kâseyi koy", None, "(On akita_black_bowl_1 wooden_cabinet_1_top_side)", "BDDL :language 'on the top of the drawer' der; modele dosya adındaki metin gider"),
 ("C16", "C", 1, "libero_goal", 6, "NTPP (içine)", "iç", "DAT", "hayır", "put the cream cheese in the bowl",
  [N("krem peynir", "acc"), N("kase", "gen"), R("iç", "dat")] + W("koy"), 2, 1,
  "kâsenin içine krem peyniri koy", None, "(On cream_cheese_1 akita_black_bowl_1)", "komut 'in' der, predicate On'dur: TR çevirisi iç-i-ne mi üst-ü-ne mi? (dil ≠ dünya)"),
 ("C17", "C", 1, "libero_goal", 9, "EC (üstüne)", "—", "DAT (yalın)", "hayır", "put the wine bottle on the rack",
  [N("şarap şişesi", "acc", poss=True), N("raf", "dat")] + W("koy"), 1, 1,
  "şarap şişesini rafın üstüne yerleştir", None, "(On wine_bottle_1 wine_rack_1_top_region)", "EN 'on' → doğal TR'de ilişki adı yok (raf-a); tr_b açık biçim"),
 ("C18", "C", 1, "libero_goal", 5, "yön (ön)", "ön", "DAT", "evet", "push the plate to the front of the stove",
  [N("tabak", "acc"), N("ocak", "gen"), R("ön", "dat")] + W("it"), 2, 1,
  "ocağın önüne doğru tabağı it", None, "(On plate_1 main_table_stove_front_region)", "LIBERO'da ocağın 'önü' = +x (kameraya bakan yüz; robota göre ARKA)"),
 ("D19", "D", 3, "libero_goal", 8, "yön (sağ)", "sağ", "DAT", "evet", "put the bowl to the right of the plate",
  [N("kase", "acc"), N("tabak", "gen"), R("sağ", "dat")] + W("koy"), 2, 1,
  "tabağın sağ tarafına kâseyi koy", None, "yok — yeni bölge predicate'i gerekir (Ders 7: sağ = yarı düzlem)", "eğitilmemiş; LIBERO-90'da benzeri var (white bowl / right of the plate)"),
 ("D20", "D", 3, "libero_goal", 8, "yön (sol)", "sol", "DAT", "evet", "put the bowl to the left of the plate",
  [N("kase", "acc"), N("tabak", "gen"), R("sol", "dat")] + W("koy"), 2, 1,
  "tabağın sol tarafına kâseyi koy", None, "yok — yeni bölge predicate'i gerekir", "D19'un minimal çifti: sağ-ı-na / sol-u-na (I ünlüsü ı/u)"),
]
COMPLEMENT = {"right": ("left", True), "far_plate": ("near_plate", True)}   # the other bowl


def load_referents(path):
    d = {}
    if path and pathlib.Path(path).exists():
        for r in csv.DictReader(open(path, encoding="utf-8")):
            d[(int(r["task_id"]), r["descriptor"], r["frame"])] = (r["referent"], float(r["min_margin"]), r["unit"], r["unambiguous"] == "True")
    return d


def referent(refs, task, desc, frame):
    flip = False
    if desc in COMPLEMENT:
        desc, flip = COMPLEMENT[desc]
    if (task, desc, frame) not in refs:
        return "", ""
    who, m, unit, ok = refs[(task, desc, frame)]
    if flip and who.startswith("bowl_"):
        who = "bowl_2" if who == "bowl_1" else "bowl_1"
    return (who if ok else who + "?"), f"{m:g} {unit}"


def libero_commands(root):
    out = {}
    if not root:
        return out
    t = (pathlib.Path(root).expanduser() / "libero/libero/benchmark/libero_suite_task_map.py").read_text(encoding="utf-8")
    for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]:
        names = re.findall(r'"([^"]+)"', re.search(r'"%s":\s*\[(.*?)\]' % suite, t, re.S).group(1))
        out[suite] = {re.sub(r"^[A-Z_]+SCENE\d+_", "", n).replace("_", " ") for n in names}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="."); ap.add_argument("--referents"); ap.add_argument("--libero")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    refs = load_referents(a.referents or out / "referents.csv")
    seen = libero_commands(a.libero)
    pairs, cmds = [], []
    for (pid, block, tier, suite, task, relation, relnoun, case, framedep, en, parts, i_rel, i_ref, tr_b, desc, goal, note) in P:
        tr_a = " ".join(p[0] for p in parts)
        tr_seg = " ".join(p[1] for p in parts)
        tr_bare = " ".join(p[2] for p in parts)
        tr_ascii = tr_a.translate(TR2ASCII)
        rl = {f: referent(refs, task, desc, f) for f in ("libero", "image", "robot", "none")} if desc else {}
        if desc and rl["none"][0]:                       # frame-independent description: the same value in all three columns
            rl = {f: rl["none"] for f in rl}
        row = dict(pair_id=pid, block=block, tier=tier, suite=suite, scene_task_id=task, relation=relation, rel_noun=relnoun,
                   case=case, frame_dependent=framedep, en=en, tr_a=tr_a, tr_a_seg=tr_seg, tr_b=tr_b, tr_ascii=tr_ascii, tr_bare=tr_bare,
                   rel_span=parts[i_rel][0], rel_seg=parts[i_rel][1], ref_span=parts[i_ref][0] if i_ref is not None else "",
                   ref_libero=rl.get("libero", ("bowl_1" if block == "A" else "", ""))[0] if desc else ("bowl_1" if block == "A" else "n/a"),
                   ref_image=rl["image"][0] if desc else ("bowl_1" if block == "A" else "n/a"),
                   ref_robot=rl["robot"][0] if desc else ("bowl_1" if block == "A" else "n/a"),
                   ref_min_margin=(rl["libero"][1] + " | " + rl["image"][1] + " | " + rl["robot"][1]) if desc and framedep == "evet" else (rl["libero"][1] if desc else ""),
                   en_seen_suite=("evet" if en in seen.get(suite, ()) else "hayır") if seen else "",
                   en_seen_libero=("evet" if any(en in s for s in seen.values()) else "hayır") if seen else "",
                   goal_predicate=goal, note=note)
        pairs.append(row)
        for variant, lang, text in [("en", "en", en), ("tr_a", "tr", tr_a), ("tr_b", "tr", tr_b), ("tr_ascii", "tr", tr_ascii), ("tr_bare", "tr", tr_bare)]:
            cmds.append(dict(id=f"{pid}_{variant}", pair_id=pid, suite=suite, task_id=task, lang=lang, variant=variant, instruction=text,
                             relation=relation, frame="çerçeveye bağlı" if framedep == "evet" else "n/a",
                             suffixes=row["rel_seg"] if variant == "tr_a" else "", tier=tier))
    for name, rows in [("pairs.csv", pairs), ("commands.csv", cmds)]:
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"written: {out/'pairs.csv'} ({len(pairs)} pairs), {out/'commands.csv'} ({len(cmds)} rows)\n")
    for r in pairs:
        ref = "" if r["ref_libero"] in ("n/a",) else f"  [libero {r['ref_libero']} | image {r['ref_image']} | robot {r['ref_robot']}]"
        print(f"{r['pair_id']} T{r['tier']} {r['en']}\n      {r['tr_a']}\n      {r['tr_a_seg']}{ref}")


if __name__ == "__main__":
    main()
