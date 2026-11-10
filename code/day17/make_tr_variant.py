"""Ders 17 — LIBERO-Plus şablonuyla Türkçe "Language" varyantı üretir: `<temel>_language_trK.bddl` + task map satırı
+ task_classification.json girdisi; dosyayı `bddl` ayrıştırıcısından geçirip modele gidecek metnin yazdığınla
AYNI olduğunu doğrular (Ders 16 Bölüm 4 kuralları: küçük harf, `;` ve parantez yok).

Model/asset GEREKMEZ. Varsayılan olarak klona DOKUNMAZ (çıktılar --out klasörüne yazılır); --install ile klona kopyalar.

Kullanım:
  python make_tr_variant.py <LIBERO-plus kökü> --commands komutlar.csv [--suite libero_spatial] [--out tr_variants/] [--install]
  komutlar.csv sütunları: base_index,tr_id,text   (base_index: 0–9 orijinal LIBERO sırası; tr_id: 1,2,…; text: Türkçe komut)
  Örnek satır: 0,1,tabak ile ramekin arasındaki siyah kâseyi al ve tabağın üstüne koy
"""
import argparse, csv, json, pathlib, re, shutil, sys, unicodedata

BASES = ["pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"]
TAIL = "_view_0_0_100_0_0_initstate_0"  # Language görevlerinin sabit kuyruğu (temiz kamera, robot init 0)


def check_text(t):
    """Ders 16'daki tuzaklar: ';' yorum siler, '()' yapıyı bozar, .lower() I/İ'yi değiştirir, U+0307 sızar."""
    problems = []
    if ";" in t: problems.append("';' var → sonrası yorum sayılır, 'Missing close parenthesis'")
    if "(" in t or ")" in t: problems.append("parantez var → iç içe liste, TypeError")
    if t != t.lower(): problems.append(f"büyük harf var → scan_tokens .lower() değiştirir: {t.lower()!r}")
    if "̇" in t: problems.append("U+0307 birleşik nokta var")
    if unicodedata.normalize("NFC", t) != t: problems.append("NFC normalize değil")
    if "  " in t or t != t.strip(): problems.append("çift/uç boşluk (scan_tokens tek boşluğa indirir)")
    return problems


def parse_language_like_bddl(path):
    """Önce gerçek `bddl` paketi (pip install bddl==1.0.1 future); yoksa aynı iki adımın birebir kopyası."""
    raw = pathlib.Path(path).read_text(encoding="utf-8")
    try:
        from bddl.parsing import scan_tokens  # bddl 1.0.1 parsing.py:14-41 → iç içe liste
        tree = scan_tokens(filename=str(path)); src = "bddl.parsing.scan_tokens"
        lang = next(group for group in tree if isinstance(group, list) and group and group[0] == ":language")
        words = lang[1:]
    except ImportError:
        s = re.sub(r";.*$", "", raw, flags=re.MULTILINE).lower()
        toks = re.findall(r"[()]|[^\s()]+", s); src = "yerel kopya (bddl paketi yok)"
        i = toks.index(":language"); j = i + 1; words = []
        while toks[j] != ")": words.append(toks[j]); j += 1
    # bddl_utils.py:83 — kelimeler " ".join ile birleştirilir
    return " ".join(words), src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--commands", required=True); ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--out", default="tr_variants"); ap.add_argument("--install", action="store_true")
    a = ap.parse_args()
    lib = pathlib.Path(a.root) / "libero/libero"; bddl_dir = lib / "bddl_files" / a.suite
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tmap_lines, json_entries, report = [], [], []
    for row in csv.DictReader(open(a.commands, encoding="utf-8")):
        base = BASES[int(row["base_index"])]; k = row["tr_id"].strip(); text = row["text"].strip()
        probs = check_text(text)
        src = (bddl_dir / f"{base}.bddl").read_text(encoding="utf-8")
        n = len(re.findall(r"\(:language .*?\)", src)); assert n == 1, f"{base}: :language satırı {n} kez"
        new = re.sub(r"\(:language .*?\)", f"(:language {text})", src, count=1)
        fname = f"{base}_language_tr{k}.bddl"; (out / fname).write_text(new, encoding="utf-8")
        try:
            got, parser = parse_language_like_bddl(out / fname)
        except Exception as e:  # ';' → Missing close parenthesis, '(' → iç içe liste (TypeError)
            got, parser = f"AYRIŞTIRICI HATASI: {type(e).__name__}: {e}", "bddl.parsing.scan_tokens"
        same = got == text
        task_name = f"{base}_language_tr{k}{TAIL}"
        tmap_lines.append(f'    "{task_name}",')
        json_entries.append({"id": None, "name": task_name, "category": "Language Instructions", "difficulty_level": None, "lang": "tr"})
        report.append((fname, text, got, same, probs, parser))
        if a.install:
            shutil.copy(out / fname, bddl_dir / fname)
    # --- rapor
    print(f"# {len(report)} Türkçe varyant → {out}/   (ayrıştırıcı: {report[0][5]})\n")
    print("| dosya | yazılan | modele gidecek (task.language) | aynı mı | uyarı |\n|---|---|---|---|---|")
    for fname, text, got, same, probs, _ in report:
        print(f"| `{fname.replace(BASES[0], '…')}` | {text} | {got} | {'evet' if same else 'HAYIR'} | {'; '.join(probs) or '—'} |")
    print("\n# libero_suite_task_map.py → libero_task_map[\"%s\"] listesinin SONUNA ekle (indeksler kaymasın):" % a.suite)
    print("\n".join(tmap_lines))
    print("\n# task_classification.json → aynı suite listesinin sonuna (id = son id + 1; difficulty_level: 4 referans modelle ölçülmedi → null):")
    print(json.dumps(json_entries, ensure_ascii=False, indent=1))
    print("\n# Sonra: VLANEXT_TASK_IDS olarak yeni görevlerin 0 tabanlı indeksleri = eski liste uzunluğu, +1, …  (Ders 16: id-1 == indeks)")
    (out / "task_map_lines.txt").write_text("\n".join(tmap_lines) + "\n", encoding="utf-8")
    json.dump(json_entries, open(out / "task_classification_entries.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if a.install: print(f"\nKlona kopyalandı: {bddl_dir}  (task map ve JSON'u elle güncelle)")
    bad = [r for r in report if not r[3] or r[4]]
    if bad: print(f"\nUYARI: {len(bad)} varyantta metin değişti ya da uyarı var — Türkçe'yi baştan küçük harfle, ';' ve parantezsiz yaz.", file=sys.stderr)


if __name__ == "__main__":
    main()
