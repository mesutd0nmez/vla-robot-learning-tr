"""Lecture 18 — Produce the condition JSON of one task from datasets/01-libero-spatial-tr-mini/data/commands.csv.

  python make_conditions.py --task 0                      # → conditions_task0.json (clean/empty/wrong/tr)
  python make_conditions.py --task 0 --extra para,tr2     # 6 conditions: + EN paraphrase + second Turkish surface form
  python make_conditions.py --task 3 --csv ../../datasets/01-libero-spatial-tr-mini/data/commands.csv

The output is in the format VLANEXT_CONDITIONS expects: {"name": command | null}. null = task.language (filled in by the eval code;
so the text of the "clean" condition comes from the benchmark rather than from a file, and a typo cannot corrupt clean).
Mapping: clean ← en (written only for checking, null in the JSON), empty ← empty (""), wrong ← distractor, tr ← tr.
No model / GPU needed.
"""
import argparse, csv, json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--task", type=int, default=0)
ap.add_argument("--csv", default=str(HERE / "../../datasets/01-libero-spatial-tr-mini/data/commands.csv"))
ap.add_argument("--extra", default="", help="comma-separated: para,tr2 (condition names in commands.csv)")
ap.add_argument("--out", default=None)
a = ap.parse_args()

rows = {r["condition"]: r for r in csv.DictReader(open(a.csv, encoding="utf-8")) if int(r["task_id"]) == a.task}
if not rows:
    sys.exit(f"ERROR: no task_id={a.task} in commands.csv (0–4 exist)")
need = {"en": "clean", "empty": "empty", "distractor": "wrong", "tr": "tr"}
missing = [k for k in need if k not in rows]
if missing:
    sys.exit(f"ERROR: missing condition(s) for task {a.task}: {missing}")

conds = {"clean": None, "empty": "", "wrong": rows["distractor"]["instruction"], "tr": rows["tr"]["instruction"]}
for name in [x for x in a.extra.split(",") if x.strip()]:
    if name not in rows:
        sys.exit(f"ERROR: --extra {name} is not in commands.csv; available: {sorted(rows)}")
    conds[name] = rows[name]["instruction"]

for name, text in conds.items():          # Lecture 16/17 rules: the override path preserves uppercase but the check is harmless
    if text and any(ch in text for ch in ";()"):
        print(f"WARNING: condition {name} contains ';' or a parenthesis — if you switch to the BDDL path, scan_tokens breaks (Lecture 16)")

out = pathlib.Path(a.out or HERE / f"conditions_task{a.task}.json")
out.write_text(json.dumps(conds, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"task {a.task} — clean (task.language, null in the JSON): {rows['en']['instruction']}")
for k, v in conds.items():
    print(f"  {k:6s} → {v!r}")
print(f"{len(conds)} conditions → {out}")
print(f'Colab: os.environ["VLANEXT_CONDITIONS"] = open("{out.name}").read()')
