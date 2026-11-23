"""Lecture 16 — Read the LIBERO-Plus task map, select the variants of one base task, produce VLANEXT_TASK_IDS.

NO model / GPU / assets needed: it only reads two text files in the LIBERO-Plus clone
(it does not run the code; it parses the task map with `ast.literal_eval`).

Usage:
  python select_plus_tasks.py <LIBERO-plus root>                      # suite × category summary
  python select_plus_tasks.py <root> --suite libero_spatial --base 0 --category Language --pick 3
  python select_plus_tasks.py <root> --base 0 --category Language --pick 3 --json selection.json

--base N   : original LIBERO task order (same as list_tasks.py in Lecture 10, 0–9)
--pick K   : the K easiest variants (difficulty_level ascending; ties by id order)
The indices in the output are 0-based = `task_suite.get_task(i)` = the number to put in VLANEXT_TASK_IDS.
The "id" field in task_classification.json is 1-based (id - 1 = index; this script verifies it).
"""
import argparse, ast, json, pathlib, re, sys
from collections import Counter

# Original LIBERO-Spatial / Object / Goal / 10 task order (LIBERO libero_suite_task_map.py; Lecture 10)
BASE = {
    "libero_spatial": [
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
    ],
}
SHORT = {"Language": "Language Instructions", "Camera": "Camera Viewpoints", "Robot": "Robot Initial States",
         "Light": "Light Conditions", "Background": "Background Textures", "Noise": "Sensor Noise",
         "Layout": "Objects Layout"}


def load_task_map(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "libero_task_map":
            return ast.literal_eval(node.value)
    sys.exit(f"ERROR: {path} does not contain libero_task_map")


def bddl_language(path):
    """The command LIBERO-Plus sees: like bddl.parsing.scan_tokens, everything after ';' is deleted, .lower(), joined with spaces."""
    raw = path.read_text(encoding="utf-8")
    raw = re.sub(r";.*$", "", raw, flags=re.MULTILINE).lower()
    m = re.search(r"\(:language([^()]*)\)", raw)
    return " ".join(m.group(1).split()) if m else None


def variant_of(name, base):
    """Which base task a task name belongs to: if it starts with base + '_' (or is exactly equal)."""
    return name == base or name.startswith(base + "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="LIBERO-plus repo root (the one containing libero/libero/benchmark/)")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--base", type=int, default=None)
    ap.add_argument("--category", default=None, help="Language | Camera | Robot | Light | Background | Noise | Layout")
    ap.add_argument("--pick", type=int, default=0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    lib = pathlib.Path(a.root) / "libero" / "libero"
    cls = json.loads((lib / "benchmark" / "task_classification.json").read_text(encoding="utf-8"))
    tmap = load_task_map(lib / "benchmark" / "libero_suite_task_map.py")

    # 1) Consistency: is json id-1 == task map order?
    for s in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        names, js = tmap[s], cls[s]
        ok = all(names[x["id"] - 1] == x["name"] for x in js)
        print(f"{s:15s} tasks={len(names):5d}  json={len(js):5d}  id-1 == index: {ok}")
    print(f"TOTAL {sum(len(cls[s]) for s in cls)} tasks\n")

    if a.base is None:
        cats = list(SHORT.values())
        print(f"{'suite':15s} " + " ".join(f"{c.split()[0][:10]:>10s}" for c in cats))
        for s in cls:
            c = Counter(x["category"] for x in cls[s])
            print(f"{s:15s} " + " ".join(f"{c[k]:10d}" for k in cats))
        return

    base = BASE[a.suite][a.base]
    names = tmap[a.suite]
    info = {x["name"]: x for x in cls[a.suite]}
    cat = SHORT.get(a.category, a.category) if a.category else None
    rows = []
    for i, n in enumerate(names):
        if not variant_of(n, base):
            continue
        x = info[n]
        if cat and x["category"] != cat:
            continue
        rows.append((x["difficulty_level"] or 0, i, n, x["category"]))
    clean_in_map = base in names
    print(f"base task {a.base}: {base}")
    print(f"  variants of this task ({cat or 'all categories'}): {len(rows)}  | is the unperturbed version in the benchmark: {clean_in_map}")
    print("  difficulty distribution:", dict(sorted(Counter(r[0] for r in rows).items())))
    rows.sort()
    pick = rows[: a.pick] if a.pick else rows
    out = []
    print(f"\n{'index':>6} {'difficulty':>10}  command (as LIBERO-Plus gives it to the model)")
    for d, i, n, c in pick:
        if "_language_" in n:
            bddl = lib / "bddl_files" / a.suite / (n.split("_view_")[0] + ".bddl")
        else:
            bddl = None
        lang = bddl_language(bddl) if bddl and bddl.exists() else " ".join(base.split("_"))
        print(f"{i:6d} {d:10d}  {lang}")
        out.append({"index": i, "difficulty": d, "category": c, "name": n, "language": lang})
    ids = ",".join(str(r["index"]) for r in out)
    clean = " ".join(base.split("_"))
    print(f"\nVLANEXT_TASK_IDS=\"{ids}\"")
    print(f"clean command (original LIBERO task.language): \"{clean}\"")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps({"suite": a.suite, "base": a.base, "clean_language": clean,
                                                    "task_ids": ids, "tasks": out}, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
        print("written:", a.json)


if __name__ == "__main__":
    main()
