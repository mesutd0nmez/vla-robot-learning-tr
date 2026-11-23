"""Lecture 17 — Produces a Turkish "Language" variant using the LIBERO-Plus template: `<base>_language_trK.bddl` + a task map line
+ a task_classification.json entry; passes the file through the `bddl` parser and verifies that the text that will go to the model is
THE SAME as what you wrote (Lecture 16 Section 4 rules: lowercase, no `;` and no parentheses).

NO model/assets needed. By default it does NOT touch the clone (outputs go to the --out folder); --install copies into the clone.

Usage:
  python make_tr_variant.py <LIBERO-plus root> --commands commands.csv [--suite libero_spatial] [--out tr_variants/] [--install]
  commands.csv columns: base_index,tr_id,text   (base_index: 0–9 original LIBERO order; tr_id: 1,2,…; text: Turkish command)
  Example row: 0,1,tabak ile ramekin arasındaki siyah kâseyi al ve tabağın üstüne koy
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
TAIL = "_view_0_0_100_0_0_initstate_0"  # fixed tail of the Language tasks (clean camera, robot init 0)


def check_text(t):
    """The pitfalls from Lecture 16: ';' deletes as a comment, '()' breaks the structure, .lower() changes I/İ, U+0307 leaks in."""
    problems = []
    if ";" in t: problems.append("';' present → everything after it is treated as a comment, 'Missing close parenthesis'")
    if "(" in t or ")" in t: problems.append("parenthesis present → nested list, TypeError")
    if t != t.lower(): problems.append(f"uppercase letter present → scan_tokens .lower() changes it: {t.lower()!r}")
    if "̇" in t: problems.append("U+0307 combining dot present")
    if unicodedata.normalize("NFC", t) != t: problems.append("not NFC-normalized")
    if "  " in t or t != t.strip(): problems.append("double/edge whitespace (scan_tokens collapses it to a single space)")
    return problems


def parse_language_like_bddl(path):
    """First the real `bddl` package (pip install bddl==1.0.1 future); if missing, an exact copy of the same two steps."""
    raw = pathlib.Path(path).read_text(encoding="utf-8")
    try:
        from bddl.parsing import scan_tokens  # bddl 1.0.1 parsing.py:14-41 → nested list
        tree = scan_tokens(filename=str(path)); src = "bddl.parsing.scan_tokens"
        lang = next(group for group in tree if isinstance(group, list) and group and group[0] == ":language")
        words = lang[1:]
    except ImportError:
        s = re.sub(r";.*$", "", raw, flags=re.MULTILINE).lower()
        toks = re.findall(r"[()]|[^\s()]+", s); src = "local copy (no bddl package)"
        i = toks.index(":language"); j = i + 1; words = []
        while toks[j] != ")": words.append(toks[j]); j += 1
    # bddl_utils.py:83 — the words are joined with " ".join
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
        n = len(re.findall(r"\(:language .*?\)", src)); assert n == 1, f"{base}: :language line appears {n} times"
        new = re.sub(r"\(:language .*?\)", f"(:language {text})", src, count=1)
        fname = f"{base}_language_tr{k}.bddl"; (out / fname).write_text(new, encoding="utf-8")
        try:
            got, parser = parse_language_like_bddl(out / fname)
        except Exception as e:  # ';' → Missing close parenthesis, '(' → nested list (TypeError)
            got, parser = f"PARSER ERROR: {type(e).__name__}: {e}", "bddl.parsing.scan_tokens"
        same = got == text
        task_name = f"{base}_language_tr{k}{TAIL}"
        tmap_lines.append(f'    "{task_name}",')
        json_entries.append({"id": None, "name": task_name, "category": "Language Instructions", "difficulty_level": None, "lang": "tr"})
        report.append((fname, text, got, same, probs, parser))
        if a.install:
            shutil.copy(out / fname, bddl_dir / fname)
    # --- report
    print(f"# {len(report)} Turkish variants → {out}/   (parser: {report[0][5]})\n")
    print("| file | written | goes to the model (task.language) | same? | warning |\n|---|---|---|---|---|")
    for fname, text, got, same, probs, _ in report:
        print(f"| `{fname.replace(BASES[0], '…')}` | {text} | {got} | {'yes' if same else 'NO'} | {'; '.join(probs) or '—'} |")
    print("\n# libero_suite_task_map.py → append to the END of the libero_task_map[\"%s\"] list (so the indices do not shift):" % a.suite)
    print("\n".join(tmap_lines))
    print("\n# task_classification.json → to the end of the same suite list (id = last id + 1; difficulty_level: not measured with the 4 reference models → null):")
    print(json.dumps(json_entries, ensure_ascii=False, indent=1))
    print("\n# Then: as VLANEXT_TASK_IDS use the 0-based indices of the new tasks = old list length, +1, …  (Lecture 16: id-1 == index)")
    (out / "task_map_lines.txt").write_text("\n".join(tmap_lines) + "\n", encoding="utf-8")
    json.dump(json_entries, open(out / "task_classification_entries.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if a.install: print(f"\nCopied into the clone: {bddl_dir}  (update the task map and JSON by hand)")
    bad = [r for r in report if not r[3] or r[4]]
    if bad: print(f"\nWARNING: in {len(bad)} variants the text changed or there is a warning — write the Turkish in lowercase from the start, without ';' and parentheses.", file=sys.stderr)


if __name__ == "__main__":
    main()
