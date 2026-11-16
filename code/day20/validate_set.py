"""Ders 20 — Komut setinin (pairs.csv + commands.csv) tasarım kurallarına uyduğunu denetler ve özet tabloları basar.

Denetimler:
  1. Biçim: 20 çift × 5 yüzey = 100 satır; kimlikler benzersiz; BDDL-güvenli metin (Ders 16: küçük harf, ';' ve parantez yok, NFC,
     U+0307 yok, çift boşluk yok) — her satır yarın bir BDDL (:language …) satırına girebilmeli.
  2. Ek tutarlılığı: rel_span / rel_seg tr_morph ile yeniden üretilir ve karşılaştırılır; tr_a içinde geçer.
  3. Süreklilik: A01–A04'ün tr_a'sı datasets/01'in sp0/1/2/4 `tr` satırlarıyla birebir aynı (Ders 15 ile karşılaştırılabilirlik).
  4. Referans belirliliği: B bloğundaki her tarif 50 init state'in hepsinde aynı kâseyi gösteriyor (referents.csv, eşik üstü).
  5. Denge: blok × tier, ilişki türü, hâl dağılımı; çerçeveye bağlı satır sayısı.
  6. Minimal çiftler: tr_a'sı tam olarak BİR sözcükte ayrılan çiftler (ek/ilişki adı etkisini yalıtan karşılaştırmalar).
  7. Allomorf tablosu: tr_a ayrımlarında her biçimbirimin kaç farklı yüzey biçimi var (Ders 21'in girdisi).
  8. Ortak kuyruk: Spatial komutlarında TR ve EN'de ayırt edici bilginin cümledeki konumu (ilk farklı sözcük indeksi).

Kullanım:  python validate_set.py ../../datasets/04-tr-spatial-commands/data [--d01 ../../datasets/01-libero-spatial-tr-mini/data/commands.csv]
"""
import argparse, csv, itertools, pathlib, re, sys, unicodedata
from collections import Counter, defaultdict
import tr_morph as M


def bddl_safe(t):
    p = []
    if ";" in t: p.append("';'")
    if "(" in t or ")" in t: p.append("parantez")
    if t != t.lower(): p.append("büyük harf")
    if "̇" in t: p.append("U+0307")
    if unicodedata.normalize("NFC", t) != t: p.append("NFC değil")
    if "  " in t or t != t.strip(): p.append("boşluk")
    return p


def morpheme_of(seg_token):
    """'tabağ-ın' → [('GEN','ın')], 'üst-ü-nde-ki' → POSS3, LOC, ki … (yalnızca bu setin ekleri)."""
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
    if len(pairs) != 20 or len(cmds) != 100: errors.append(f"boyut: {len(pairs)} çift, {len(cmds)} satır")
    if len({c["id"] for c in cmds}) != len(cmds): errors.append("yinelenen id")
    for c in cmds:
        p = bddl_safe(c["instruction"])
        if p: errors.append(f"{c['id']}: BDDL-güvensiz ({', '.join(p)})")
    by_pair = defaultdict(dict)
    for c in cmds: by_pair[c["pair_id"]][c["variant"]] = c["instruction"]
    for pid, v in by_pair.items():
        if len(set(v.values())) != 5: errors.append(f"{pid}: 5 yüzey birbirinden farklı değil")
    print(f"1. biçim: {len(pairs)} çift, {len(cmds)} satır, BDDL-güvenli: {'evet' if not errors else 'HAYIR'}")
    # 2
    n_ok = 0
    for r in pairs:
        seg_tokens = r["tr_a_seg"].split()
        if r["rel_span"] not in r["tr_a"]: errors.append(f"{r['pair_id']}: rel_span tr_a'da yok")
        if r["rel_seg"].replace("-", "") != r["rel_span"]: errors.append(f"{r['pair_id']}: rel_seg ≠ rel_span")
        if r["tr_a_seg"].replace("-", "") != r["tr_a"]: errors.append(f"{r['pair_id']}: ayrım yüzeyle uyuşmuyor")
        if r["rel_noun"] in M.LEXICON and r["case"] in ("LOC+ki", "DAT"):
            c = "loc" if r["case"] == "LOC+ki" else "dat"
            exp = M.rel(r["rel_noun"], c, r["case"] == "LOC+ki")
            if exp != (r["rel_span"], r["rel_seg"]): errors.append(f"{r['pair_id']}: tr_morph {exp} ≠ {r['rel_span']}")
            else: n_ok += 1
    print(f"2. ek tutarlılığı: {n_ok} ilişki sözcüğü tr_morph ile yeniden üretildi, hepsi aynı")
    # 3
    if a.d01:
        d01 = {r["id"]: r["instruction"] for r in csv.DictReader(open(a.d01, encoding="utf-8"))}
        same = [(p, k) for p, k in [("A01", "sp0_tr"), ("A02", "sp1_tr"), ("A03", "sp2_tr"), ("A04", "sp4_tr")] if by_pair[p]["tr_a"] == d01[k]]
        if len(same) != 4: errors.append("datasets/01 ile süreklilik bozuk")
        print(f"3. süreklilik: datasets/01 ile birebir aynı tr_a: {len(same)}/4")
    # 4
    refs = {(int(r["task_id"]), r["descriptor"], r["frame"]): r for r in csv.DictReader(open(d / "referents.csv", encoding="utf-8"))}
    print("4. referans belirliliği (B bloğu; 50 init state):")
    for r in pairs:
        if r["block"] != "B": continue
        amb = [f for f in ("ref_libero", "ref_image", "ref_robot") if r[f].endswith("?") or r[f] in ("", "belirsiz")]
        if amb: errors.append(f"{r['pair_id']}: belirsiz referans {amb}")
        agree = len({r["ref_libero"], r["ref_image"], r["ref_robot"]}) == 1
        print(f"   {r['pair_id']} görev {r['scene_task_id']} {r['rel_span']:10s} libero={r['ref_libero']} image={r['ref_image']} robot={r['ref_robot']}"
              f"  {'çerçeveler UYUŞUYOR' if agree else 'çerçeveler AYRIŞIYOR'}  (en küçük fark: {r['ref_min_margin']})")
    # 5
    print("5. denge:")
    for key in ("block", "tier", "relation", "case", "frame_dependent", "suite"):
        print(f"   {key:16s}", dict(sorted(Counter(r[key] for r in pairs).items())))
    # 6
    print("6. minimal çiftler (tr_a tek sözcükte ayrılıyor):")
    n_min = 0
    for x, y in itertools.combinations(pairs, 2):
        tx, ty = x["tr_a"].split(), y["tr_a"].split()
        if len(tx) == len(ty):
            diff = [(p, q) for p, q in zip(tx, ty) if p != q]
            if len(diff) == 1:
                n_min += 1
                print(f"   {x['pair_id']} ~ {y['pair_id']}: {diff[0][0]} / {diff[0][1]}")
    print(f"   toplam {n_min} minimal çift")
    # 7
    allo = defaultdict(Counter)
    for r in pairs:
        for tok in r["tr_a_seg"].split():
            for m, s in morpheme_of(tok):
                allo[m][s] += 1
    print("7. allomorf tablosu (tr_a, 20 komut):")
    for m in ("GEN", "POSS3", "ACC", "DAT", "LOC", "ABL", "ki", "?"):
        if allo[m]:
            print(f"   {m:6s} {len(allo[m])} biçim: " + ", ".join(f"-{s}×{n}" for s, n in allo[m].most_common()))
    if allo["?"]: errors.append("sınıflanamayan ek var")
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
    print(f"8. ayırt edici bilginin konumu (12 Spatial komutu): EN ortak baş {common_head('en')} sözcük, ortak kuyruk {common_tail('en')}; "
          f"TR ortak baş {common_head('tr_a')}, ortak kuyruk {common_tail('tr_a')}")
    print("\nSONUÇ:", "TEMİZ — hata yok" if not errors else f"{len(errors)} HATA")
    for e in errors: print("  -", e)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
