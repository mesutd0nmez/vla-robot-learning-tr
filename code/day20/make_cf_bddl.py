"""Ders 20 — Karşı-olgusal (counterfactual) hedefli BDDL üretir: komut sahnedeki ÖTEKİ kâseyi (akita_black_bowl_2) tarif
ediyorsa, LIBERO'nun success predicate'i (On akita_black_bowl_1 plate_1) modeli haksız yere cezalandırır (Ders 19 §5).
Bu script pairs.csv'de herhangi bir çerçevede bowl_2'yi gösteren her satır için orijinal BDDL'in kopyasını üretir:
  (:goal (And (On akita_black_bowl_2 plate_1)))   +   (:obj_of_interest akita_black_bowl_2 plate_1)   +   (:language <tr_a>)
Sahne (:init) ve bölgeler DEĞİŞMEZ → orijinal görevin 50 init state'i aynen kullanılabilir.

Üretilen her dosya `bddl` paketiyle (pip install bddl==1.0.1 future) geri okunur: :language yazdığımız metinle birebir mi,
:goal bowl_2 mi, :init orijinalle aynı mı. Paket yoksa aynı kontroller regex ile yapılır.

Kullanım:
  python make_cf_bddl.py --libero ~/src/LIBERO --pairs ../../datasets/04-tr-spatial-commands/data/pairs.csv --out .../data/bddl_cf
Not: orijinal LIBERO'da modele giden komut BDDL'den DEĞİL dosya adından gelir (Ders 13); komutu VLANEXT_INSTR_OVERRIDE ile
verirsin, bu dosya yalnızca success predicate'ini değiştirir. Dosya adı = <pair_id>_<çerçeve>_bowl2.bddl
"""
import argparse, csv, pathlib, re

ORDER_FILE = "libero/libero/benchmark/libero_suite_task_map.py"


def official_order(root, suite="libero_spatial"):
    t = (root / ORDER_FILE).read_text(encoding="utf-8")
    return re.findall(r'"([^"]+)"', re.search(r'"%s":\s*\[(.*?)\]' % suite, t, re.S).group(1))


def rewrite(src, text):
    t = src
    t, n1 = re.subn(r"\(:language[^)]*\)", f"(:language {text})", t)
    t, n2 = re.subn(r"\(:goal\s*\(And \(On akita_black_bowl_1 plate_1\)\)\s*\)", "(:goal\n    (And (On akita_black_bowl_2 plate_1))\n  )", t)
    t, n3 = re.subn(r"\(:obj_of_interest\s*akita_black_bowl_1", "(:obj_of_interest\n    akita_black_bowl_2", t)
    assert (n1, n2, n3) == (1, 1, 1), f"şablon tutmadı: language={n1} goal={n2} obj_of_interest={n3}"
    return t


def block(tree, key):
    return next(g for g in tree if isinstance(g, list) and g and g[0] == key)


def verify(path, text, orig_path):
    try:
        from bddl.parsing import scan_tokens
        new, old = scan_tokens(filename=str(path)), scan_tokens(filename=str(orig_path))
        lang = " ".join(block(new, ":language")[1:])
        goal = str(block(new, ":goal")[1:])
        same_init = block(new, ":init") == block(old, ":init") and block(new, ":regions") == block(old, ":regions")
        src = "bddl.parsing.scan_tokens"
    except ImportError:
        raw, raw0 = path.read_text(encoding="utf-8"), orig_path.read_text(encoding="utf-8")
        lang = re.search(r"\(:language\s+([^)]*)\)", raw).group(1).strip().lower()
        goal = raw.split("(:goal")[1]
        init = lambda s: re.sub(r"\s+", " ", re.search(r"\(:init(.*?)\)\s*\(:goal", s, re.S).group(1))
        same_init = init(raw) == init(raw0); src = "regex (bddl paketi yok)"
    return lang == text, "akita_black_bowl_2" in goal and "akita_black_bowl_1" not in goal, same_init, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libero", required=True); ap.add_argument("--pairs", required=True); ap.add_argument("--out", default="bddl_cf")
    a = ap.parse_args()
    root = pathlib.Path(a.libero).expanduser(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    names = official_order(root)
    index = []
    for r in csv.DictReader(open(a.pairs, encoding="utf-8")):
        if r["suite"] != "libero_spatial": continue
        frames = [f for f in ("libero", "image", "robot") if r[f"ref_{f}"] == "bowl_2"]
        if not frames: continue
        tag = "all" if len(frames) == 3 else "-".join(frames)
        orig = root / "libero/libero/bddl_files/libero_spatial" / (names[int(r["scene_task_id"])] + ".bddl")
        dst = out / f"{r['pair_id']}_{tag}_bowl2.bddl"
        dst.write_text(rewrite(orig.read_text(encoding="utf-8"), r["tr_a"]), encoding="utf-8")
        ok_lang, ok_goal, ok_init, src = verify(dst, r["tr_a"], orig)
        index.append(dict(pair_id=r["pair_id"], frames_pointing_to_bowl2=tag, base_task_id=r["scene_task_id"], base_bddl=orig.name,
                          cf_bddl=dst.name, instruction=r["tr_a"], language_roundtrip=ok_lang, goal_is_bowl2=ok_goal, init_unchanged=ok_init))
        print(f"{dst.name:28s} görev {r['scene_task_id']}  dil birebir: {ok_lang}  hedef bowl_2: {ok_goal}  sahne aynı: {ok_init}  [{src}]")
        assert ok_lang and ok_goal and ok_init
    with open(out / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0])); w.writeheader(); w.writerows(index)
    print(f"\n{len(index)} karşı-olgusal BDDL + index.csv → {out}")


if __name__ == "__main__":
    main()
