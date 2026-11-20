"""Ders 24 — Çözüm adayı 4'ün en ucuz biçimi: kural tabanlı Türkçe komut → yapısal çerçeve → İngilizce LIBERO şablonu.

Fikir (Ders 6 / Ders 23: "LLM parser + sembolik çözücü"): Türkçe komutu modele hiç göstermeden önce
  (fiil, hedef nesne, ilişki, referans(lar), varış) çerçevesine ayır, sonra çerçeveden LIBERO'nun eğitimde gördüğü
İngilizce şablonu üret. Bu script LLM bile kullanmaz: ~25 kelimelik sözlük + ek soyma + 3 şablon.
Amaç bir çeviri sistemi yazmak DEĞİL; "çeviri katmanı ne kadar ucuz olabilir ve nerede kırılır" sorusuna
datasets/04'ün 80 Türkçe satırı üzerinde GERÇEK bir sayı vermek (başarı = üretilen İngilizce, `en` sütunuyla birebir aynı).

Kullanım:
  python tr_parse.py                                   # datasets/04 commands.csv → özet tablo + parse_roundtrip.csv
  python tr_parse.py --cmd "soldaki siyah kaseyi al ve tabağın üstüne koy"
  python tr_parse.py --selftest
"""
import argparse, csv, json, re, sys, unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / ".." / ".." / "datasets" / "04-tr-spatial-commands" / "data" / "commands.csv"

# ---------------------------------------------------------------- sözlük (ASCII'ye katlanmış kökler)
ASCII = str.maketrans("çğıöşüâîû", "cgiosuaiu")


def fold(w):
    """Türkçe harfleri ASCII'ye katla, NFC; `tr_ascii` yüzeyi ile `tr_a` aynı anahtara düşsün diye."""
    return unicodedata.normalize("NFC", w).lower().translate(ASCII)


OBJECTS = {  # kök → İngilizce ad
    "kase": "bowl", "tabak": "plate", "ramekin": "ramekin", "masa": "table", "dolap": "cabinet", "cekmece": "drawer",
    "ocak": "stove", "raf": "rack", "sise": "bottle", "peynir": "cheese", "robot": "robot",
}
SOFT = {"tabag": "tabak", "dolab": "dolap", "ocag": "ocak"}            # yumuşama geri alımı
ADJ = {"siyah": "black", "ahsap": "wooden", "ust": "top", "sarap": "wine", "krem": "cream"}
RELATIONS = {  # ilişki adı kökü → (İngilizce seçim ifadesi, İngilizce varış ifadesi)
    "ara": ("between {refs}", None), "yan": ("next to {ref}", None), "orta": ("from {ref} center", None),
    "ust": ("on {ref}", "on {ref}"), "uzer": ("on {ref}", "on {ref}"), "ic": ("in {ref}", "in {ref}"),
    "sol": ("on the left", "to the left of {ref}"), "sag": ("on the right", "to the right of {ref}"),
    "on": ("in the front", "to the front of {ref}"),
    "yakin": ("closer to {ref}", None), "uzak": ("farther from {ref}", None),
}
VERBS = {"al": "pick", "alip": "pick", "koy": "put", "yerlestir": "put", "it": "push"}
STOP = {"ve", "ile", "daha", "olan", "duran", "dogru", "taraftaki", "tarafina", "tarafindaki", "-"}
# ek soyma: en uzun eşleşen kök + izin verilen ek dizisi
SUFFIX = re.compile(r"^(l[ae]|n?[iu]n|[sy]?[iu]|[yn]?[ae]|n?d[ae]n?|t[ae]n?|ki|n)*$")  # izin verilen ek dizisi (ASCII)
STEMS = sorted(set(OBJECTS) | set(SOFT) | set(ADJ) | set(RELATIONS) | set(VERBS), key=len, reverse=True)


def analyze(word):
    """Bir sözcüğü (tür, kök, hâl) üçlüsüne indir. Hâl ∈ {acc, dat, loc, abl, ins, gen, ki, bare}."""
    w = fold(word)
    if w in STOP:
        return ("STOP", w, "bare")
    for stem in STEMS:  # en uzun kök önce (ust > us; sarap > sag...)
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
    """[ADJ...] OBJ → İngilizce 'the wooden cabinet' / 'the top drawer'."""
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
    # yalın "üst": ardından "çekmece" geliyorsa sıfat (top drawer), yoksa ilişki adı (tr_bare: "tabak üst koy")
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
    # --- seçim ilişkisi: ilk REL (varış ilişkisinden önce gelen) ---
    rel_idx = [i for i, (k, s, c) in enumerate(toks) if k == "REL"]
    obj_idx = [i for i, (k, s, c) in enumerate(toks) if k == "OBJ"]
    if frame["verb"] == "pick":  # Spatial şablonu: ... siyah kase ... al ve tabak üst koy
        first_verb = next(i for i, (k, s, c) in enumerate(toks) if k == "VERB")
        before = toks[:first_verb]
        after = toks[first_verb + 1:]
        # hedef: "al"dan önceki son OBJ öbeği (kase)
        # hedef: fiilden önce belirtme hâli (-yi) almış OBJ; yoksa (tr_bare) fiilden önceki son OBJ
        acc_before = [i for i in obj_idx if i < first_verb and toks[i][2] == "acc"]
        tgt_pos = acc_before[-1] if acc_before else max((i for i in obj_idx if i < first_verb), default=None)
        if tgt_pos is None:
            return frame
        tgt_start = tgt_pos
        while tgt_start - 1 >= 0 and toks[tgt_start - 1][0] == "ADJ" and toks[tgt_start - 1][1] == "siyah":
            tgt_start -= 1
        frame["target"] = noun_phrase(toks[tgt_start:tgt_pos + 1])
        rels_sel = [i for i in rel_idx if i < first_verb]          # fiilden önceki ilişki adları (seçim)
        # referans öbekleri: fiilden önce, hedef öbeğin dışında kalan [ADJ...] OBJ dizileri
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
        elif groups:  # ilişki adı yok, yalın bulunma: "dolabın üst çekmecesindeki" → in the top drawer of the cabinet
            frame["relation"] = "ic"
        dest_objs = [i for i in obj_idx if i > first_verb]
        dest_rels = [i for i in rel_idx if i > first_verb]
        if dest_objs:
            frame["dest"] = noun_phrase([toks[dest_objs[0]]])
            frame["dest_rel"] = toks[dest_rels[0]][1] if dest_rels else "ust"
    else:  # Goal şablonu: X-i Y-nin üst-ü-ne koy / Y-nin üst-ü-ne X-i koy
        acc = [i for i in obj_idx if toks[i][2] == "acc"]
        rel = rel_idx[0] if rel_idx else None
        if acc:
            tpos = acc[0]
        else:  # tr_bare: ilk OBJ hedef
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
            # "şarap şişesini rafa koy": ilişki adı yok → DAT = on
            frame["dest_rel"] = "ust"
    return frame


def render(frame):
    """Çerçeveden LIBERO İngilizce şablonu."""
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
            d = "on top of the cabinet"  # LIBERO-Goal görev 4'ün adı böyle
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
    print(f"{'yüzey':<9}{'n':>4}{'birebir':>9}{'yanlış EN':>11}{'parse yok':>11}")
    for v in ("tr_a", "tr_ascii", "tr_bare", "tr_b"):
        s = stats.get(v, {})
        n = sum(s.values())
        print(f"{v:<9}{n:>4}{s.get('ok',0):>9}{s.get('wrong_en',0):>11}{s.get('parse_fail',0):>11}")
    tot = sum(sum(s.values()) for s in stats.values()); ok = sum(s["ok"] for s in stats.values())
    print(f"{'toplam':<9}{tot:>4}{ok:>9}   ({100*ok/tot:.0f}%)   → {out_path}")
    fails = [o for o in out if o["result"] != "ok"]
    if fails:
        print("\nBaşarısızlar:")
        for o in fails:
            print(f"  {o['id']:<14} {o['tr']}\n      → {o['en_pred'] or '∅'}\n      ≠ {o['en_gold']}")
    return stats


def run_heldout(path):
    """Parser dondurulduktan SONRA yazılmış 12 yeni Türkçe yüzey: sözlük/şablon dışına çıkınca ne oluyor?"""
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
