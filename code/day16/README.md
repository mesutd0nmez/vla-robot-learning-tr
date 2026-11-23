# day16

Lecture 16 — Phase 2 synthesis + LIBERO-Plus: installation in a separate runtime via `PYTHONPATH`, the code path of Language tasks, the effect of the BDDL parser on Turkish, a closed-loop trial with VLANeXt on `rewritten` / `empty` (optionally `clean` / `tr`).

| File | What | Status |
|---|---|---|
| `colab_libero_plus_t4.ipynb` | 14-cell Colab/Kaggle T4 notebook (new runtime!) | Not run on Colab (no GPU/HF) |
| `select_plus_tasks.py` | `task_classification.json` + task map → suite × category summary, variants of a base task, `VLANEXT_TASK_IDS` | ran (real data) |
| `bddl_lang_probe.py` | What happens when a Turkish command passes through `bddl.parsing.scan_tokens` (`;`, `.lower()`, parentheses) | ran with the real `bddl==1.0.1` |
| `vlanext_plus_patch.py` | 5 patches: json path, `VLANEXT_INSTR_OVERRIDE`, `VLANEXT_RUN_TAG`, log line, `np.float_` | verified on a copy of VLANeXt d911e6b + LIBERO-plus 4976dc3 (idempotent) |
| `parse_plus_log.py` | LIBERO-Plus `eval.log` → table + `results/runs.csv` | `--selftest` (synthetic log) |
| `t4_plus_config.yaml` | Small-budget eval config | — |
| `plot_day16.py` | Lecture figures 02, 04, 05, 06 | ran |

Usage (on the Mac, no model needed):
```bash
git clone --depth 1 https://github.com/sylvestf/LIBERO-plus ~/vla-work/LIBERO-plus
curl -O https://raw.githubusercontent.com/mesutd0nmez/vla-robot-learning-tr/main/code/day16/select_plus_tasks.py
python select_plus_tasks.py ~/vla-work/LIBERO-plus --base 0 --category Language --pick 3
pip install bddl==1.0.1 future && curl -O https://raw.githubusercontent.com/mesutd0nmez/vla-robot-learning-tr/main/code/day16/bddl_lang_probe.py
python bddl_lang_probe.py
```
Colab: <https://colab.research.google.com/github/mesutd0nmez/vla-robot-learning-tr/blob/main/code/day16/colab_libero_plus_t4.ipynb>


## select_plus_tasks.py

```python
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
```

## bddl_lang_probe.py

```python
"""Lecture 16 — When a Turkish command passes through LIBERO-Plus's BDDL parser, what reaches the model?

For `_language_` tasks, LIBERO-Plus reads the command from the `(:language ...)` field of the BDDL file
(libero/libero/benchmark/__init__.py → grab_language_from_filename → bddl_utils.get_problem_info).
That path uses `bddl.parsing.scan_tokens`: (1) treats everything from ';' to the end of the line as a COMMENT and deletes it,
(2) lowercases the whole file with Python `str.lower()`, (3) reads the '(' and ')' characters as structure,
(4) rejoins the words with single spaces.

This script uses the real `bddl` package (pip install bddl==1.0.1 future); if it is missing it imitates the same 3 steps.
Usage:  python bddl_lang_probe.py [--csv output.csv]
"""
import argparse, csv, re, unicodedata

try:
    from bddl.parsing import scan_tokens          # the function LIBERO / LIBERO-Plus use
    SRC = "bddl.parsing.scan_tokens (real package)"
except Exception:                                  # pragma: no cover
    SRC = "imitation (bddl is not installed)"

    def scan_tokens(string):
        s = re.sub(r";.*$", "", string, flags=re.MULTILINE).lower()
        stack, tokens = [], []
        for t in re.findall(r"[()]|[^\s()]+", s):
            if t == "(":
                stack.append(tokens); tokens = []
            elif t == ")":
                if not stack:
                    raise Exception("Missing open parenthesis")
                toks, tokens = tokens, stack.pop(); tokens.append(toks)
            else:
                tokens.append(t)
        if stack:
            raise Exception("Missing close parenthesis")
        return tokens[0]

TEMPLATE = "(define (problem LIBERO_Tabletop_Manipulation)\n  (:domain robosuite)\n  (:language {cmd})\n)\n"

CASES = [
    ("EN reference", "pick up the black bowl between the plate and the ramekin and place it on the plate"),
    ("TR lowercase", "tabak ile ramekinin arasındaki siyah kaseyi al ve tabağın üstüne koy"),
    ("TR uppercase I", "Işığın altındaki kâseyi al"),
    ("TR uppercase İ", "İki tabağın arasındaki kâseyi al"),
    ("TR ALL CAPS", "SIRADAKİ KASEYİ AL"),
    ("semicolon", "kâseyi al; tabağa koy"),
    ("parentheses", "kâseyi (siyah olanı) tabağa koy"),
    ("apostrophe", "ramekin'in yanındaki kâseyi al"),
    ("empty", ""),
]


def parse_language(cmd):
    tokens = scan_tokens(string=TEMPLATE.format(cmd=cmd))
    # libero/libero/envs/bddl_utils.py get_problem_info: tokens.pop(0)=="define", groups from the end, the ":language" group
    tokens = list(tokens)
    assert tokens.pop(0) == "define"
    lang = ""
    while tokens:
        g = tokens.pop()
        if isinstance(g, list) and g and g[0] == ":language":
            lang = " ".join(g[1:])   # same as bddl_utils.py:83; TypeError if there is a nested list
    return lang


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--csv"); a = ap.parse_args()
    print("parser:", SRC)
    rows = []
    for label, cmd in CASES:
        try:
            out = parse_language(cmd); err = ""
        except Exception as e:
            out, err = "", f"{type(e).__name__}: {e}"
        same = out == cmd
        extra = [f"U+{ord(c):04X}" for c in out if unicodedata.combining(c)]
        note = err or ("same" if same else "CHANGED") + (f" (combining mark {','.join(extra)})" if extra else "")
        rows.append({"durum": label, "girdi": cmd, "modele_giden": out, "len_in": len(cmd), "len_out": len(out), "not": note})
        print(f"- {label:16s} | {cmd!r}\n  {'':16s} → {out!r}  [{note}]")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print("written:", a.csv)


if __name__ == "__main__":
    main()
```

## vlanext_plus_patch.py

```python
"""Lecture 16 — Prepare VLANeXt's LIBERO-Plus eval for Colab/Kaggle and for the experiment conditions.
First apply Lecture 12's patch (sdpa, mmap, VLANEXT_TASK_IDS), then this one:

  python vlanext_t4_patch.py   /content/VLANeXt
  python vlanext_plus_patch.py /content/VLANeXt [/content/VLANeXt/third_party/LIBERO-plus]

It is idempotent; if it cannot find an expected line it stops and says so (the repo may have changed). Source: VLANeXt d911e6b,
LIBERO-plus 4976dc3 (Nov 2026 clone).

Patches
 1. task_classification.json path   The eval script hard-codes the path to the authors' server
                                    (/data/NTU_slab/...). If it cannot find it, the category statistics stay EMPTY
                                    (only a warning). → LIBERO_PLUS_ROOT environment variable.
 2. VLANEXT_INSTR_OVERRIDE          If defined, every episode's command becomes this string ("" = empty command).
                                    The only line changed: `task_description = task.language`
                                    inside build_episode_specs (Lecture 13).
 3. VLANEXT_RUN_TAG                 Suffix for the output folder; so the videos/logs of the conditions do not get mixed up.
 4. Log line                        The override value is written to eval.log (read by parse_plus_log.py).
 5. np.float_ → np.float64          LIBERO-plus env_wrapper.py:105, numpy 2 (VLANeXt COMMON_ISSUES #3).
"""
import pathlib, sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
plus = pathlib.Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else root / "third_party" / "LIBERO-plus"
ev = root / "scripts" / "libero_plus_bench_eval.py"
par = root / "src" / "evaluation" / "libero_bench" / "parallel_eval_utils.py"
envw = plus / "libero" / "libero" / "envs" / "env_wrapper.py"
for p in (ev, par):
    if not p.exists():
        sys.exit(f"ERROR: {p} does not exist — the first argument must be the VLANeXt repo root")


def patch(path, old, new, label):
    src = path.read_text(encoding="utf-8")
    if new in src:
        print(f"  [=] {label}: already patched"); return
    if old not in src:
        sys.exit(f"ERROR: {label}: expected line not found ({path.name}). Looked for: {old!r}")
    path.write_text(src.replace(old, new, 1), encoding="utf-8")
    print(f"  [+] {label}: applied")


print(f"VLANeXt root: {root}\nLIBERO-plus: {plus}")
# 1) json path
patch(ev,
      'json_path = Path("/data/NTU_slab/draven/proj/third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json")',
      'json_path = Path(os.environ.get("LIBERO_PLUS_ROOT", "third_party/LIBERO-plus")) / "libero/libero/benchmark/task_classification.json"  # Lecture 16 patch',
      "task_classification.json path")
# 2) command override (parallel_eval_utils; Lecture 12's patch adds `import os`; if it is missing we add it)
if "\nimport os\n" not in "\n" + par.read_text(encoding="utf-8"):
    patch(par, "import numpy as np\n", "import os\nimport numpy as np\n", "import os (parallel_eval_utils)")
patch(par,
      "        task_description = task.language\n",
      '        task_description = os.environ["VLANEXT_INSTR_OVERRIDE"] if "VLANEXT_INSTR_OVERRIDE" in os.environ else task.language  # Lecture 16 patch\n',
      "VLANEXT_INSTR_OVERRIDE")
# 3) output folder tag
patch(ev,
      '        f"_libero_plus"\n    )',
      '        f"_libero_plus"\n        + os.environ.get("VLANEXT_RUN_TAG", "")  # Lecture 16 patch\n    )',
      "VLANEXT_RUN_TAG")
# 4) write the override to the log
patch(ev,
      '    logger.info("Task suite: %s", cfg.eval.task_suite_name)\n',
      '    logger.info("Task suite: %s", cfg.eval.task_suite_name)\n'
      '    logger.info("Instruction override: %r", os.environ.get("VLANEXT_INSTR_OVERRIDE", "<none: task.language>"))  # Lecture 16 patch\n',
      "override log line")
# 5) numpy 2
if envw.exists():
    patch(envw, "dtype=np.float_)", "dtype=np.float64)  # Lecture 16 patch (numpy 2)", "np.float_ -> np.float64")
else:
    print(f"  [!] {envw} does not exist — run again after cloning LIBERO-plus")

print("Done. Check:")
for p in (ev, par, envw):
    if p.exists():
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if "Lecture 16 patch" in line or "T4 patch" in line:
                print(f"  {p.name}:{i}: {line.strip()[:100]}")
```

## parse_plus_log.py

```python
"""Lecture 16 — VLANeXt LIBERO-Plus eval.log → episode table + runs.csv rows.

Usage:
  python parse_plus_log.py <eval.log> --run-id vlanext_plus_t4_01 --condition rewritten --lang en --csv results/runs.csv
  python parse_plus_log.py --selftest      # tests with a log produced by the script's own format string

The log format is from scripts/libero_plus_bench_eval.py (DIFFERENT from libero_bench_eval in Lecture 12: no trial/episode_time,
there is a category):
  Episode result: task=%d/%d category=%s success=%s rollout_steps=%d total_episodes=%d/%d total_successes=%d total_success_rate=%.1f%% video=%s
runs.csv convention (CLAUDE.md): run_id, model, checkpoint, suite, task_id, lang, condition, n_episodes, success_rate
task_id = LIBERO-Plus index (0-based, same as VLANEXT_TASK_IDS).
"""
import argparse, csv, pathlib, re, sys, tempfile
from collections import defaultdict

EP = re.compile(r"Episode result: task=(\d+)/(\d+) category=(\S+) success=(True|False) rollout_steps=(\d+) "
                r"total_episodes=(\d+)/(\d+) total_successes=(\d+) total_success_rate=([\d.]+)% video=(\S+)")
OVR = re.compile(r"Instruction override: (.*)")
SUITE = re.compile(r"Task suite: (\S+)")
CKPT = re.compile(r"checkpoint=(\S+)")
HEADER = ["run_id", "model", "checkpoint", "suite", "task_id", "lang", "condition", "n_episodes", "success_rate"]


def parse(text):
    eps = [m.groups() for m in EP.finditer(text)]
    s = SUITE.search(text); c = CKPT.search(text); o = OVR.search(text)
    return {"suite": s.group(1) if s else "?", "checkpoint": pathlib.Path(c.group(1)).name if c else "?",
            "override": o.group(1).strip() if o else "(no line — patch 4 may not have been applied)", "episodes": eps}


def report(info, run_id, lang, condition, csv_path=None):
    eps = info["episodes"]
    if not eps:
        sys.exit("No 'Episode result' line in the log — the eval may have crashed before starting; look at the end of the log.")
    print(f"suite={info['suite']}  checkpoint={info['checkpoint']}  override={info['override']}  episodes={len(eps)}")
    per = defaultdict(lambda: [0, 0]); cats = {}
    print(f"{'index':>6} {'category':>10} {'success':>7} {'steps':>5}  video")
    for task, n, cat, ok, steps, *_rest, video in eps:
        i = int(task) - 1; per[i][0] += 1; per[i][1] += ok == "True"; cats[i] = cat
        print(f"{i:6d} {cat:>10} {ok:>7} {int(steps):5d}  {pathlib.Path(video).name[:60]}")
    tn = sum(v[0] for v in per.values()); ts = sum(v[1] for v in per.values())
    print(f"Total: {ts}/{tn} = {100 * ts / tn:.1f}%")
    if csv_path:
        out = pathlib.Path(csv_path); out.parent.mkdir(parents=True, exist_ok=True)
        new = not out.exists()
        with out.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(HEADER)
            for i, (n, s) in sorted(per.items()):
                w.writerow([run_id, "VLANeXt-B", info["checkpoint"], info["suite"] + "_plus", i, lang, condition, n,
                            round(s / n, 4)])
        print(f"{len(per)} rows appended → {out}")
    return ts, tn


def selftest():
    fmt = ("Episode result: task=%d/%d category=%s success=%s rollout_steps=%d total_episodes=%d/%d "
           "total_successes=%d total_success_rate=%.1f%% video=%s")
    lines = ["2026-11-09 10:00:00 INFO Task suite: libero_spatial",
             "2026-11-09 10:00:00 INFO Instruction override: ''",
             "2026-11-09 10:00:01 INFO Eval setup: checkpoint=checkpoints/VLANeXt_libero_spatial.pt tasks=2402"]
    demo = [(989, True, 131), (995, False, 220), (996, True, 152)]
    for k, (t, ok, st) in enumerate(demo, 1):
        succ = sum(d[1] for d in demo[:k])
        lines.append("2026-11-09 10:0%d:00 INFO " % k + fmt % (t, 2402, "Language", ok, st, k, 3, succ, 100 * succ / k,
                                                              f"/x/episode={k}--success={ok}--task=.mp4"))
    with tempfile.TemporaryDirectory() as d:
        csvp = pathlib.Path(d) / "runs.csv"
        ts, tn = report(parse("\n".join(lines)), "selftest", "none", "empty", csvp)
        rows = list(csv.DictReader(csvp.open(encoding="utf-8")))
    assert (ts, tn) == (2, 3) and len(rows) == 3 and rows[0]["task_id"] == "988", rows
    print("SELFTEST OK (synthetic log; not a real result)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?")
    ap.add_argument("--run-id", default="vlanext_plus_t4_01")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--condition", default="rewritten", help="clean | rewritten | empty | tr")
    ap.add_argument("--csv")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.log:
        selftest()
    else:
        report(parse(pathlib.Path(a.log).read_text(encoding="utf-8", errors="replace")), a.run_id, a.lang, a.condition, a.csv)
```

## plot_day16.py

```python
"""Lecture 16 figures (02, 04, 05, 06).

Usage:  python plot_day16.py <output_folder/> [--plus <LIBERO-plus root>] [--probe probe.csv]
  If --plus is given, the numbers of 02 and 04 are RECOMPUTED from task_classification.json; otherwise the embedded
  numbers below are used (taken from the LIBERO-plus 4976dc3 clone with this script's --plus mode).
  --probe: the --csv output of bddl_lang_probe.py (for 05).
The numbers in 06 come from the LIBERO-plus README leaderboard (transcribed by hand).
"""
import argparse, csv, json, os, pathlib
from collections import Counter, defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--plus"); ap.add_argument("--probe")
a = ap.parse_args(); out = a.out.rstrip("/") + "/"; os.makedirs(out, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
CATS = ["Camera Viewpoints", "Robot Initial States", "Language Instructions", "Light Conditions",
        "Background Textures", "Sensor Noise", "Objects Layout"]
SHORT = ["Camera", "Robot", "Language", "Light", "Background", "Noise", "Layout"]
COUNTS = {  # suite -> category order of CATS
    "libero_spatial": [376, 350, 390, 292, 258, 351, 385],
    "libero_object": [396, 398, 354, 297, 248, 422, 403],
    "libero_goal": [408, 409, 410, 279, 281, 379, 425],
    "libero_10": [419, 393, 383, 274, 289, 449, 312],
}
LANG_DIFF = {  # suite -> counts of difficulty 1..5 in the Language category
    "libero_spatial": [115, 100, 58, 82, 35], "libero_object": [103, 67, 72, 63, 49],
    "libero_goal": [45, 49, 52, 79, 185], "libero_10": [65, 64, 77, 104, 73],
}
if a.plus:
    d = json.loads((pathlib.Path(a.plus) / "libero/libero/benchmark/task_classification.json").read_text(encoding="utf-8"))
    COUNTS = {s: [Counter(x["category"] for x in d[s])[c] for c in CATS] for s in SUITES}
    LANG_DIFF = {}
    for s in SUITES:
        c = Counter(x["difficulty_level"] for x in d[s] if x["category"] == "Language Instructions")
        LANG_DIFF[s] = [c[i] for i in range(1, 6)]
    print("recomputed:", COUNTS, LANG_DIFF)

# --- 02: 10,030 tasks, suite × category
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(SUITES)); bottom = np.zeros(len(SUITES))
cols = ["#9ecae1", "#c6dbef", "#ffd166", "#fdd0a2", "#d9d9d9", "#e5c3e8", "#c7e9c0"]
for j, (name, c) in enumerate(zip(SHORT, cols)):
    v = np.array([COUNTS[s][j] for s in SUITES])
    ax.bar(x, v, 0.62, bottom=bottom, color=c, edgecolor="white", label=name, zorder=3)
    for xi, (b, val) in enumerate(zip(bottom, v)):
        ax.text(xi, b + val / 2, str(val), ha="center", va="center", fontsize=8.5,
                fontweight="bold" if name == "Language" else None)
    bottom += v
for xi, t in enumerate(bottom):
    ax.text(xi, t + 25, f"{int(t)}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(["Spatial", "Object", "Goal", "Long (10)"])
ax.set_ylabel("number of tasks"); ax.set_ylim(0, 2800); ax.yaxis.grid(True, color="#eee", zorder=0)
ax.legend(ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False, fontsize=9)
ax.set_title(f"LIBERO-Plus: {int(bottom.sum())} tasks = 4 suites × 7 perturbations — source: task_classification.json (LIBERO-plus 4976dc3)",
             fontsize=10)
fig.text(0.5, 0.005, "Each task is a single perturbation instance and is run for 1 episode (num_trials_per_task: 1). "
         "The unperturbed original task is NOT in this list.", ha="center", fontsize=8.5, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(out + "02-task-map.png", dpi=170); plt.close(fig)

# --- 04: difficulty distribution of the Language tasks (difficulty = how many of 4 models solved it)
fig, ax = plt.subplots(figsize=(11, 4.6))
lv_cols = ["#d4f5d4", "#e8f5c8", "#fff3c4", "#ffe0c2", "#ffd6d6"]
labels = ["L1: 4 of 4 models solved it", "L2: 3 of 4", "L3: 2 of 4", "L4: 1 of 4", "L5: none"]
y = np.arange(len(SUITES))[::-1]; left = np.zeros(len(SUITES))
tot = np.array([sum(LANG_DIFF[s]) for s in SUITES], dtype=float)
for k in range(5):
    v = np.array([LANG_DIFF[s][k] for s in SUITES]) / tot * 100
    ax.barh(y, v, 0.6, left=left, color=lv_cols[k], edgecolor="#999", linewidth=0.5, label=labels[k])
    for yi, l, val, n in zip(y, left, v, [LANG_DIFF[s][k] for s in SUITES]):
        if val > 4:
            ax.text(l + val / 2, yi, f"{val:.0f}%\n({n})", ha="center", va="center", fontsize=8)
    left += v
ax.set_yticks(y); ax.set_yticklabels([f"{n} ({int(t)} tasks)" for n, t in zip(["Spatial", "Object", "Goal", "Long"], tot)])
ax.set_xlim(0, 100); ax.set_xlabel("Percentage of Language tasks")
ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False, fontsize=8.5)
ax.set_title("Language perturbation: distribution of difficulty levels — source: task_classification.json; level definition: arXiv:2510.13626 Appendix C.3",
             fontsize=9.5)
fig.text(0.5, 0.005, "Difficulty is not a property of the task itself but the outcome of 4 models (OpenVLA-OFT, π0, π0-fast, UniVLA): "
         "on Goal, none of them could solve 45% of the Language tasks.", ha="center", fontsize=8.5, color="#555")
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(out + "04-language-difficulty.png", dpi=170); plt.close(fig)

# --- 05: a Turkish command passing through the BDDL parser (real bddl 1.0.1 output)
if a.probe:
    rows = list(csv.DictReader(open(a.probe, encoding="utf-8")))
    import textwrap
    wr = lambda t, n: "\n".join(textwrap.wrap(t, n)) or t
    fig, ax = plt.subplots(figsize=(13, 0.62 * len(rows) + 1.0)); ax.axis("off")
    cell = [[r["durum"], wr(r["girdi"] or "(empty)", 44), wr(r["modele_giden"] or "—", 44), wr(r["not"], 34)] for r in rows]
    tb = ax.table(cellText=cell, colLabels=["case", "written to BDDL", "task.language (goes to the model)", "note"],
                  loc="center", cellLoc="left", colWidths=[0.13, 0.33, 0.33, 0.27])
    tb.auto_set_font_size(False); tb.set_fontsize(8.6); tb.scale(1, 2.6)
    for (r, c), cl in tb.get_celld().items():
        cl.set_edgecolor("#bbb")
        if r == 0:
            cl.set_facecolor("#cfe8ff"); cl.set_text_props(fontweight="bold")
        elif c == 3:
            t = rows[r - 1]["not"]
            cl.set_facecolor("#d4f5d4" if t == "same" else "#ffd6d6")
        elif c == 2:
            cl.set_facecolor("#fffbea")
    ax.set_title("bddl.parsing.scan_tokens: comment after ';', str.lower(), parentheses = structure — real output (bddl 1.0.1, bddl_lang_probe.py)",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(out + "05-bddl-turkish-probe.png", dpi=170, bbox_inches="tight"); plt.close(fig)

# --- 06: LIBERO-Plus leaderboard: Language column vs the average of the other 6 perturbations (README)
LB = {  # model: Camera, Robot, Language, Light, Background, Noise, Layout, Total
    "OpenVLA": [0.8, 3.5, 23.0, 8.1, 34.8, 15.2, 28.5, 15.6],
    "OpenVLA-OFT": [56.4, 31.9, 79.5, 88.7, 93.3, 75.8, 74.2, 69.6],
    "OpenVLA-OFT_w": [10.4, 38.7, 70.5, 76.8, 93.6, 49.9, 69.9, 55.8],
    "NORA": [2.2, 37.0, 65.1, 45.7, 58.6, 12.8, 62.1, 39.0],
    "WorldVLA": [0.1, 27.9, 41.6, 43.7, 17.1, 10.9, 38.0, 25.0],
    "UniVLA": [1.8, 46.2, 69.6, 69.0, 81.0, 21.2, 31.9, 43.9],
    "π0": [13.8, 6.0, 58.8, 85.0, 81.4, 79.0, 68.9, 53.6],
    "π0-Fast": [65.1, 21.6, 61.0, 73.2, 73.2, 74.4, 68.8, 61.6],
    "RIPT-VLA": [55.2, 31.2, 77.6, 88.4, 91.6, 73.5, 74.2, 68.4],
    "OpenVLA-OFT_m": [55.6, 21.7, 81.0, 92.7, 91.0, 78.6, 68.7, 67.9],
    "OpenVLA-OFT+": [92.8, 30.3, 85.8, 94.9, 93.9, 89.3, 77.6, 79.6],
}
fig, ax = plt.subplots(figsize=(10.5, 4.8))
names = list(LB); lang = np.array([LB[n][2] for n in names]); others = np.array([np.mean(LB[n][:2] + LB[n][3:7]) for n in names])
rank = np.array([1 + sum(v > LB[n][2] for v in LB[n][:2] + LB[n][3:7]) for n in names])
order = np.argsort(lang)
yy = np.arange(len(names))
ax.hlines(yy, others[order], lang[order], color="#bbb", zorder=1)
ax.scatter(others[order], yy, color="#cfe8ff", edgecolor="#3d7fc4", s=60, zorder=3, label="average of the other 6 perturbations")
ax.scatter(lang[order], yy, color="#ffd166", edgecolor="#b07d00", s=70, zorder=3, label="Language perturbation")
for k, i in enumerate(order):
    ax.text(102, k, f"{rank[i]}.", va="center", fontsize=8.5, color="#555")
ax.text(2, -0.9, "number in the right column: rank of Language among the 7 perturbations by ease (1 = easiest)", fontsize=8, color="#555", va="center")
ax.set_yticks(yy); ax.set_yticklabels([names[i] for i in order]); ax.set_xlim(0, 110)
ax.set_xlabel("success rate (%) — average over 4 suites"); ax.xaxis.grid(True, color="#eee", zorder=0)
ax.set_ylim(-1.6, len(names) - 0.4)
ax.legend(loc="upper left", frameon=False, fontsize=9)
ax.set_title("LIBERO-Plus: when the command is rewritten, success drops LESS than for the other perturbations\nsource: LIBERO-plus README leaderboard",
             fontsize=10)
fig.tight_layout(); fig.savefig(out + "06-leaderboard-language.png", dpi=170); plt.close(fig)
print("ok")
```

## t4_plus_config.yaml

```yaml
# Lecture 16 — VLANeXt LIBERO-Plus eval, small budget for the free T4.
# Orijinal: VLANeXt/config/libero_plus_bench_config.yaml (num_parallel_envs: 32, num_trials_per_task: 1, 2402 tasks)
# Usage (from the VLANeXt root; PYTHONPATH=third_party/LIBERO-plus, LIBERO_CONFIG_PATH=~/.libero_plus):
#   VLANEXT_TASK_IDS="988,994,995" python -m scripts.libero_plus_bench_eval --config config/t4_plus_config.yaml
data:
  augmentation:
    center_crop: true
    center_crop_ratio: 0.9

model:
  diffusion_steps: 5
  scheduler_type: "flow_match"

project:
  use_wandb: false

eval:
  finetuned_checkpoint: "checkpoints/VLANeXt_libero_spatial.pt"   # the file from Lecture 12 (trained on LIBERO)
  task_suite_name: "libero_spatial"
  num_steps_wait: 10
  num_steps_execute: 8
  num_parallel_envs: 1         # LIBERO-Plus envs load extra assets; start with 1 on a T4 + 12.7 GB RAM
  num_trials_per_task: 1       # in LIBERO-Plus every task is already a single perturbation instance; README: 50 → 1
  seed: 42
  image_size: 256
  resume_episodes: 0
  resume_successes: 0
```
