# day18

Lecture 18 — Same task, four commands: on LIBERO-Spatial task 0, 10 init states × {clean, empty, wrong, tr} = 40 episodes, with a single model load on a Colab/Kaggle T4. The condition becomes a field of `EpisodeSpec` in the eval code (Lecture 18 patch); the analysis produces a 4-row table with Wilson intervals + paired (McNemar) and unpaired (Fisher) tests. None of the scripts in this folder needs a model/GPU on the Mac; the run is in the notebook.

| File | What | Status |
|---|---|---|
| `colab_4cond_t4.ipynb` | 13-cell Colab/Kaggle T4 notebook: Lecture 12 setup + Lecture 18 patch + single run + analysis + videos | Not run on Colab (no GPU/HF) |
| `vlanext_cond_patch.py` | `VLANEXT_CONDITIONS` (JSON) support for VLANeXt: `EpisodeSpec.condition`, init × condition loop, `condition=` in the log line, condition in the video name, `VLANEXT_RUN_TAG`, condition log line (6 patches, idempotent) | applied to a repo copy in 3 scenarios (clean / Lecture 12 / Lecture 12+16), the 2nd run says "already patched", `py_compile` clean |
| `make_conditions.py` → `conditions_task0.json` | Condition JSON from `commands.csv` in `datasets/01` (clean=null, empty="", wrong=distractor, tr=sp0_tr; `--extra para,tr2`) | ran (tasks 0 and 1) |
| `t4_cond_config.yaml` | `num_trials_per_task: 10`, `num_parallel_envs: 2` (Kaggle 4) | — |
| `spec_order_demo.py` | Shows the episode order and batches produced by the patched loop without a model | ran |
| `analyze_4cond.py` | `eval.log` → `episodes.csv` + `runs.csv` + `table.md` (k/n, Wilson 95%, Δ, discordant pairs, McNemar exact p, Fisher p, mean steps/time); Plan B with `--logs condition=path` | `--selftest` (synthetic 40 rows produced with the eval script's format string) |
| `power_day18.py` | Figure 03 (Wilson intervals, n = 10/20/50) and 04 (Fisher / McNemar exact power curves) | ran (pure computation) |
| `plot_day18.py` | Figure 02 (empty command in the literature, Spatial), 06 (LIBERO-CF wrong command: faithful/biased), 07 (four possible patterns — ILLUSTRATIVE) | ran (paper values by hand) |

Colab: <https://colab.research.google.com/github/mesutd0nmez/vla-robot-learning-tr/blob/main/code/day18/colab_4cond_t4.ipynb>

On the Mac (no model needed):
```bash
cd code/day18
python make_conditions.py --task 0            # conditions_task0.json
python spec_order_demo.py conditions_task0.json --parallel 2
python analyze_4cond.py --selftest
python power_day18.py ../../lectures/images/lecture-18/
python plot_day18.py  ../../lectures/images/lecture-18/
# With the eval.log you downloaded from Colab:
python analyze_4cond.py eval.log --run-id vlanext_sp0_4cond_01 --out ../../projects/05-spatial0-4kosul/results/
```

## vlanext_cond_patch.py
```python
"""Lecture 18 — Add the notion of a "condition" to VLANeXt's LIBERO eval: with a single model load, on the same init states,
run several command conditions (clean / empty / wrong / tr).

  python vlanext_t4_patch.py   /content/VLANeXt      # Lecture 12 (sdpa, mmap, VLANEXT_TASK_IDS) — FIRST
  python vlanext_cond_patch.py /content/VLANeXt      # this file

It is idempotent; if it cannot find an expected line it stops and says so. If Lecture 16's patch (VLANEXT_INSTR_OVERRIDE) has been applied
it works with that too: if a condition list is defined the conditions win, otherwise the Lecture 16 behavior continues.
Source: VLANeXt d911e6b (28 Sep 2026), scripts/libero_bench_eval.py + src/evaluation/libero_bench/parallel_eval_utils.py.

Environment variables (after the patch)
  VLANEXT_CONDITIONS  JSON: {"clean": null, "empty": "", "wrong": "...", "tr": "..."}  or the path of a .json file.
                      null = task.language (the original command). For every init state one episode per condition is produced;
                      order: init 0 × {clean, empty, wrong, tr}, init 1 × {...}, ... → with num_parallel_envs=4 the four conditions
                      of the same init state run in the SAME batch.
  VLANEXT_RUN_TAG     suffix of the output folder (so the logs/videos of the runs do not get mixed up; same name as in Lecture 16).

Patches (6; all marked with a "Lecture 18 patch" comment)
 1. EpisodeSpec.condition field                 parallel_eval_utils.py
 2. build_episode_specs: condition × init state parallel_eval_utils.py  (task_description line + for loop + spec fields)
 3. condition= in the "Episode result" log line libero_bench_eval.py
 4. Condition name at the start of the video name libero_bench_eval.py  (so that "task=" does not stay empty for an empty command)
 5. VLANEXT_RUN_TAG                             libero_bench_eval.py
 6. Write the condition list to the log         libero_bench_eval.py   (read by analyze_4cond.py)
"""
import pathlib, sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
ev = root / "scripts" / "libero_bench_eval.py"
par = root / "src" / "evaluation" / "libero_bench" / "parallel_eval_utils.py"
for p in (ev, par):
    if not p.exists():
        sys.exit(f"ERROR: {p} does not exist — the argument must be the VLANeXt repo root")


def patch(path, old, new, label, alt_old=None):
    src = path.read_text(encoding="utf-8")
    if new in src:
        print(f"  [=] {label}: already patched"); return
    if old not in src and alt_old and alt_old in src:
        old = alt_old
    if old not in src:
        sys.exit(f"ERROR: {label}: expected line not found ({path.name}); the repo may have changed. Looked for: {old!r}")
    path.write_text(src.replace(old, new, 1), encoding="utf-8")
    print(f"  [+] {label}: applied")


print(f"VLANeXt root: {root}")

# --- parallel_eval_utils.py -------------------------------------------------------------------------------
if "\nimport os\n" not in "\n" + par.read_text(encoding="utf-8"):
    patch(par, "import numpy as np\n", "import os\nimport numpy as np\n", "import os (parallel_eval_utils)")
patch(par, "import numpy as np\n",
      "import numpy as np\n"
      "import json as _json  # Lecture 18 patch\n"
      "\n"
      "\n"
      "def _load_conditions(default_text):  # Lecture 18 patch: [(name, command)] ; null = task.language\n"
      '    raw = os.environ.get("VLANEXT_CONDITIONS")\n'
      "    if not raw:\n"
      '        return [(None, default_text)]\n'
      "    if raw.strip().endswith(\".json\") and os.path.exists(raw.strip()):\n"
      '        raw = open(raw.strip(), encoding="utf-8").read()\n'
      "    conds = _json.loads(raw)\n"
      "    return [(str(k), default_text if v is None else str(v)) for k, v in conds.items()]\n",
      "_load_conditions definition")
# 1) spec field
patch(par, "    task_category: str = None\n",
      "    task_category: str = None\n    condition: str = None  # Lecture 18 patch\n",
      "EpisodeSpec.condition")
# 2) condition list: the original line or Lecture 16's override line
_ORIG = "        task_description = task.language\n"
_D16 = '        task_description = os.environ["VLANEXT_INSTR_OVERRIDE"] if "VLANEXT_INSTR_OVERRIDE" in os.environ else task.language  # Lecture 16 patch\n'
_line = _D16 if _D16 in par.read_text(encoding="utf-8") else _ORIG   # if the Lecture 16 patch is present it is kept (the condition list goes on top of it)
patch(par, _line, _line + "        _conds = _load_conditions(task_description)  # Lecture 18 patch\n", "condition list (task_description)")
patch(par,
      "        for episode_idx in range(num_trials_per_task):\n",
      "        for episode_idx, (_cond_name, _cond_text) in ((e, c) for e in range(num_trials_per_task) for c in _conds):  # Lecture 18 patch\n",
      "init state × condition loop")
patch(par,
      "                    task_description=task_description,\n                    task_category=task_category,\n",
      "                    task_description=_cond_text,  # Lecture 18 patch\n                    task_category=task_category,\n                    condition=_cond_name,  # Lecture 18 patch\n",
      "spec fields")

# --- libero_bench_eval.py ---------------------------------------------------------------------------------
# 3) log line
patch(ev,
      '"Episode result: task=%d/%d trial=%d success=%s rollout_steps=%d',
      '"Episode result: task=%d/%d trial=%d condition=%s success=%s rollout_steps=%d',
      "log line (format)")
patch(ev,
      "                spec.task_id + 1,\n                num_tasks_in_suite,\n                spec.episode_idx,\n                result.success,\n",
      "                spec.task_id + 1,\n                num_tasks_in_suite,\n                spec.episode_idx,\n                spec.condition,  # Lecture 18 patch\n                result.success,\n",
      "log line (argument)")
# 4) video name
patch(ev,
      "                task_description=spec.task_description,\n",
      '                task_description=((spec.condition + "__") if spec.condition else "") + spec.task_description,  # Lecture 18 patch\n',
      "video name")
# 5) output folder tag
patch(ev,
      '        f"_libero"\n    )',
      '        f"_libero"\n        + os.environ.get("VLANEXT_RUN_TAG", "")  # Lecture 18 patch\n    )',
      "VLANEXT_RUN_TAG")
# 6) write the conditions to the log
patch(ev,
      '    logger.info("Task suite: %s", cfg.eval.task_suite_name)\n',
      '    logger.info("Task suite: %s", cfg.eval.task_suite_name)\n'
      '    logger.info("Conditions: %s", os.environ.get("VLANEXT_CONDITIONS", "<none: task.language>"))  # Lecture 18 patch\n',
      "condition log line")

print("Done. Check:")
for p in (par, ev):
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if "Lecture 18 patch" in line:
            print(f"  {p.name}:{i}: {line.strip()[:110]}")
```

## make_conditions.py
```python
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
```

## spec_order_demo.py
```python
"""Lecture 18 — Show the order produced by the patched build_episode_specs without a model (same loop, fake task_suite).

  python spec_order_demo.py conditions_task0.json [--trials 10] [--parallel 2]

It answers the questions "with num_parallel_envs=2, which episodes are in the same batch?" and "which condition does global_episode_id correspond to?"
(The patch's loop: for episode_idx, (name, text) in ((e, c) for e in range(trials) for c in conds) → init state is the outer loop, condition the inner one.)
No model / GPU / LIBERO needed.
"""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("conditions")
ap.add_argument("--trials", type=int, default=10)
ap.add_argument("--parallel", type=int, default=2)
ap.add_argument("--task-language", default="pick up the black bowl between the plate and the ramekin and place it on the plate")
a = ap.parse_args()

conds = [(k, a.task_language if v is None else v) for k, v in json.load(open(a.conditions, encoding="utf-8")).items()]
specs = []
seen = 0
for episode_idx, (cond_name, cond_text) in ((e, c) for e in range(a.trials) for c in conds):   # identical to the line in the patch
    seen += 1
    specs.append((seen, episode_idx, cond_name, cond_text))

print(f"{len(conds)} conditions × {a.trials} init states = {len(specs)} episodes; num_parallel_envs={a.parallel}\n")
print(f"{'batch':>5} {'episode':>7} {'init':>4} {'cond':6}  command")
for i in range(0, len(specs), a.parallel):
    for g, init, name, text in specs[i:i + a.parallel]:
        print(f"{i // a.parallel + 1:5d} {g:7d} {init:4d} {name:6s}  {text[:60]!r}")
    print()
same = all(len({init for _, init, _, _ in specs[i:i + a.parallel]}) == 1 for i in range(0, len(specs), a.parallel))
print("Does every batch contain a single init state?", "YES" if same else "NO (the number of conditions does not divide parallel)")
print(f"In the log: trial=<init>, condition=<name>; video name '<name>__<command>' — no need to infer the condition from the episode number.")
```

## analyze_4cond.py
```python
"""Lecture 18 — Read the eval.log of the 4-condition experiment → episodes.csv + runs.csv + result table with Wilson intervals + paired tests.

Usage:
  python analyze_4cond.py <eval.log> --run-id vlanext_sp0_4cond_01 --out results/
  python analyze_4cond.py --logs clean=a/eval.log empty=b/eval.log wrong=c/eval.log tr=d/eval.log --out results/
        (Plan B: a separate run per condition using Lecture 16's VLANEXT_INSTR_OVERRIDE; the logs have no condition= field)
  python analyze_4cond.py --selftest      # tests with a synthetic log produced with the eval script's format string

Log format (Lecture 18 patch 3): Episode result: task=%d/%d trial=%d condition=%s success=%s rollout_steps=%d episode_time=%s ...
trial = init state index (build_episode_specs: initial_states[episode_idx]) → the conditions see the SAME scene in the same trial → paired.

Outputs (--out folder):
  episodes.csv  run_id, suite, task_id, condition, init_state, success, rollout_steps, episode_time_s
  runs.csv      CLAUDE.md convention: run_id, model, checkpoint, suite, task_id, lang, condition, n_episodes, success_rate (1 row per condition; appended if the file exists)
  table.md      condition | k/n | SR | Wilson 95% | Δ vs clean | discordant pairs (clean✓x✗ / clean✗x✓) | McNemar exact p | Fisher p
No model / GPU needed; only scipy (Fisher) — if missing the Fisher column becomes "—".
"""
import argparse, csv, math, pathlib, re, sys, tempfile
from collections import OrderedDict, defaultdict

EP = re.compile(r"Episode result: task=(\d+)/(\d+) trial=(\d+)(?: condition=(\S+))? success=(True|False) "
                r"rollout_steps=(\d+) episode_time=(\S+)")
SUITE = re.compile(r"Task suite: (\S+)")
CKPT = re.compile(r"checkpoint=(\S+)")
CONDS = re.compile(r"Conditions: (.*)")
OVR = re.compile(r"Instruction override: (.*)")
LANG = {"clean": "en", "empty": "none", "wrong": "en", "para": "en", "tr": "tr", "tr2": "tr"}
HEADER_RUNS = ["run_id", "model", "checkpoint", "suite", "task_id", "lang", "condition", "n_episodes", "success_rate"]


def parse_duration(s):
    """The output of format_duration (src/utils/logging_utils.py) is 'HH:MM:SS'; for flexibility '1m23s' / '45.2s' are also accepted."""
    s = s.strip()
    m = re.fullmatch(r"(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:([\d.]+)s)?", s)
    if m and any(m.groups()):
        h, mi, se = m.groups()
        return 3600 * float(h or 0) + 60 * float(mi or 0) + float(se or 0)
    m = re.fullmatch(r"(\d+):(\d\d):(\d\d(?:\.\d+)?)", s)
    if m:
        return 3600 * int(m.group(1)) + 60 * int(m.group(2)) + float(m.group(3))
    try:
        return float(s)
    except ValueError:
        return float("nan")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def binom_two_sided(b, c):
    """McNemar exact: discordant pair counts b (clean✓, x✗) and c (clean✗, x✓); H0: b ~ Binom(b+c, 0.5)."""
    n = b + c
    if n == 0:
        return float("nan")
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def fisher(k1, n1, k2, n2):
    try:
        from scipy.stats import fisher_exact
    except ImportError:
        return float("nan")
    return fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]


def read_log(path, forced_condition=None):
    text = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
    s = SUITE.search(text); c = CKPT.search(text); cl = CONDS.search(text); o = OVR.search(text)
    eps = []
    for m in EP.finditer(text):
        task, _n, trial, cond, ok, steps, t = m.groups()
        cond = forced_condition or cond or "clean"
        eps.append(dict(task_id=int(task) - 1, init_state=int(trial), condition=cond, success=ok == "True",
                        rollout_steps=int(steps), episode_time_s=parse_duration(t)))
    return {"suite": s.group(1) if s else "?", "checkpoint": pathlib.Path(c.group(1)).name if c else "?",
            "conditions_line": (cl.group(1).strip() if cl else None), "override": (o.group(1).strip() if o else None),
            "episodes": eps}


def summarize(eps, order=None):
    by = OrderedDict()
    for e in eps:
        by.setdefault(e["condition"], {})[(e["task_id"], e["init_state"])] = e
    conds = order or list(by)
    if "clean" in conds:
        conds = ["clean"] + [c for c in conds if c != "clean"]
    rows = []
    base = by.get("clean", {})
    for cnd in conds:
        d = by[cnd]; n = len(d); k = sum(e["success"] for e in d.values())
        lo, hi = wilson(k, n)
        r = dict(condition=cnd, k=k, n=n, sr=k / n if n else float("nan"), lo=lo, hi=hi,
                 mean_steps=sum(e["rollout_steps"] for e in d.values()) / n if n else float("nan"),
                 mean_time=sum(e["episode_time_s"] for e in d.values()) / n if n else float("nan"))
        if cnd != "clean" and base:
            common = sorted(set(base) & set(d))
            b = sum(base[key]["success"] and not d[key]["success"] for key in common)
            c = sum((not base[key]["success"]) and d[key]["success"] for key in common)
            kb = sum(base[key]["success"] for key in common); nb = len(common)
            r.update(delta=(k / n - kb / nb) if n and nb else float("nan"), b=b, c=c, n_pairs=len(common),
                     p_mcnemar=binom_two_sided(b, c), p_fisher=fisher(kb, nb, k, n))
        rows.append(r)
    return rows


def fmt_p(p):
    return "—" if p != p else ("<0.001" if p < 0.001 else f"{p:.3f}")


def table_md(rows, suite, task_id):
    out = [f"Suite `{suite}`, task {task_id}; n = number of init states; Wilson 95%; paired tests relative to clean (same init state).", "",
           "| condition | k/n | SR | Wilson 95% | Δ vs clean | discordant (clean✓x✗ / clean✗x✓) | McNemar exact p | Fisher p | mean steps | mean time |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["condition"] == "clean":
            d = "—"; u = "—"; pm = "—"; pf = "—"
        else:
            d = f"{100 * r['delta']:+.0f}"; u = f"{r['b']} / {r['c']} (n={r['n_pairs']})"; pm = fmt_p(r["p_mcnemar"]); pf = fmt_p(r["p_fisher"])
        out.append(f"| {r['condition']} | {r['k']}/{r['n']} | {100 * r['sr']:.0f}% | {100 * r['lo']:.0f}–{100 * r['hi']:.0f} | {d} | {u} | {pm} | {pf} | "
                   f"{r['mean_steps']:.0f} | {r['mean_time']:.0f} s |")
    return "\n".join(out)


def write_outputs(info, eps, rows, run_id, out_dir, model="VLANeXt-B"):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    task_ids = sorted({e["task_id"] for e in eps})
    with (out / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "suite", "task_id", "condition", "init_state", "success", "rollout_steps", "episode_time_s"])
        for e in sorted(eps, key=lambda e: (e["task_id"], e["init_state"], e["condition"])):
            w.writerow([run_id, info["suite"], e["task_id"], e["condition"], e["init_state"], int(e["success"]), e["rollout_steps"],
                        f"{e['episode_time_s']:.1f}"])
    runs = out / "runs.csv"; new = not runs.exists()
    with runs.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HEADER_RUNS)
        for r in rows:
            w.writerow([run_id, model, info["checkpoint"], info["suite"], ",".join(map(str, task_ids)), LANG.get(r["condition"], "?"),
                        r["condition"], r["n"], f"{r['sr']:.3f}"])
    md = table_md(rows, info["suite"], ",".join(map(str, task_ids)))
    (out / "table.md").write_text(md + "\n", encoding="utf-8")
    return md


def selftest():
    # fake log produced with the format string of Lecture 18 patch 3 (libero_bench_eval.py): 10 init × 4 conditions; results are ILLUSTRATIVE
    fmt = ("Episode result: task=%d/%d trial=%d condition=%s success=%s rollout_steps=%d episode_time=%s total_episodes=%d/%d "
           "total_successes=%d task_success_rate=%.1f%% total_success_rate=%.1f%% elapsed=%s avg_episode=%.2fs eta=%s video=%s")
    pattern = {"clean": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0], "empty": [1, 1, 0, 1, 0, 1, 0, 0, 1, 0],
               "wrong": [1, 1, 1, 0, 1, 1, 1, 0, 1, 0], "tr": [1, 0, 0, 1, 0, 0, 1, 0, 1, 0]}
    lines = ["2026-11-11 07:00:00 INFO Task suite: libero_spatial",
             '2026-11-11 07:00:00 INFO Conditions: {"clean": null, "empty": "", "wrong": "pick up ...", "tr": "tabak ile ..."}',
             "2026-11-11 07:00:01 INFO Eval setup: checkpoint=checkpoints/VLANeXt_libero_spatial.pt tasks=10 trials_per_task=10 ..."]
    g = 0; succ = 0
    for init in range(10):
        for cnd in ["clean", "empty", "wrong", "tr"]:
            ok = bool(pattern[cnd][init]); g += 1; succ += ok
            lines.append("2026-11-11 07:05:00 INFO " + fmt % (1, 10, init, cnd, ok, 96 if ok else 220, "00:01:31" if ok else "00:03:04",
                                                             g, 40, succ, 0.0, 100 * succ / g, "00:10:00", 91.0, "00:30:00", "x.mp4"))
    lines.append("2026-11-11 08:00:00 INFO Eval finished: total_episodes=40 total_successes=%d final_success_rate=%.2f%% elapsed=01:02:00" % (succ, 2.5 * succ))
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "eval.log"; p.write_text("\n".join(lines), encoding="utf-8")
        info = read_log(p); eps = info["episodes"]
        assert len(eps) == 40, len(eps)
        rows = summarize(eps)
        assert [r["condition"] for r in rows] == ["clean", "empty", "wrong", "tr"]
        assert rows[0]["k"] == 9 and rows[1]["k"] == 5 and rows[3]["k"] == 4
        tr = rows[3]; assert (tr["b"], tr["c"]) == (5, 0), (tr["b"], tr["c"])       # clean✓tr✗ 5, clean✗tr✓ 0
        assert abs(tr["p_mcnemar"] - 2 * 0.5 ** 5) < 1e-9                          # 0.0625: even if all 5 discordant pairs are one-directional, p = 0.0625
        assert abs(wilson(9, 10)[0] - 0.5958) < 1e-3 and abs(wilson(9, 10)[1] - 0.9821) < 1e-3
        assert abs(parse_duration("00:01:31") - 91) < 1e-9 and abs(parse_duration("1m23s") - 83) < 1e-9
        assert abs(rows[0]["mean_time"] - (9 * 91 + 184) / 10) < 1e-9
        md = write_outputs(info, eps, rows, "selftest", pathlib.Path(d) / "results")
        assert (pathlib.Path(d) / "results" / "runs.csv").read_text().count("\n") == 5
        # Plan B: a log without a condition= field + a forced condition name
        p2 = pathlib.Path(d) / "e2.log"; p2.write_text("\n".join(lines).replace(" condition=tr", "").replace(" condition=clean", ""), encoding="utf-8")
        info2 = read_log(p2, forced_condition="empty"); assert {e["condition"] for e in info2["episodes"]} == {"empty"}
        print(md)
        print("\nselftest OK (numbers are ILLUSTRATIVE; run with a real log)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?")
    ap.add_argument("--logs", nargs="*", default=[], help="Plan B: condition=path pairs")
    ap.add_argument("--run-id", default="vlanext_sp0_4cond_01")
    ap.add_argument("--out", default="results")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.log:
        info = read_log(a.log); eps = info["episodes"]
        if eps and all(e["condition"] == "clean" for e in eps) and info["conditions_line"] in (None, "<none: task.language>"):
            print("WARNING: no condition= field in the log — the Lecture 18 patch was not applied or VLANEXT_CONDITIONS is empty; all counted as 'clean'")
    elif a.logs:
        info = None; eps = []
        for item in a.logs:
            cnd, _, path = item.partition("=")
            inf = read_log(path, forced_condition=cnd); eps += inf["episodes"]; info = info or inf
    else:
        ap.error("give an eval.log or --logs (or --selftest)")
    if not eps:
        sys.exit("No 'Episode result' line in the log — the eval may have crashed before starting; look at the end of the log.")
    print(f"suite={info['suite']}  checkpoint={info['checkpoint']}  episodes={len(eps)}  conditions={info['conditions_line'] or info['override'] or '—'}")
    rows = summarize(eps)
    print(write_outputs(info, eps, rows, a.run_id, a.out))
    print(f"\n→ {a.out}/episodes.csv, runs.csv ({len(rows)} rows appended), table.md")
    inits = defaultdict(set)
    for e in eps:
        inits[e["condition"]].add(e["init_state"])
    if len({frozenset(v) for v in inits.values()}) > 1:
        print("WARNING: the conditions have different sets of init states — the paired test was computed only on the common init states")


if __name__ == "__main__":
    main()
```

## power_day18.py
```python
"""Lecture 18 — figures 03 and 04: what do 10 episodes show, and what do they not show? (pure math; no model/GPU needed)

  python power_day18.py <output_folder/>

03-wilson-n10.png : Wilson 95% interval for k/n successes, n = 10 / 20 / 50 (all k values). Exact computation.
04-power.png      : with the true success probability of the clean condition p_clean = 0.95, if the true probability of the second
                    condition p_x is lower, do the two tests catch this difference at α = 0.05?
                    (a) unpaired: Fisher exact (two independent binomials; exact power over all (k1,k2) pairs),
                    (b) paired: McNemar exact — two conditions on the same init state; in the model the probability of a "clean succeeds, x fails" pair
                        is p_clean − p_x, with the reverse assumed 0 (the most optimistic paired case: the condition can only make things worse).
                    Power = total probability of the outcomes with p < 0.05. Exact computation (not a simulation).
No source (computation); the educational assumptions are written in the title.
"""
import math, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom, fisher_exact

out = (sys.argv[1] if len(sys.argv) > 1 else ".").rstrip("/") + "/"; os.makedirs(out, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
Y, B, G, R = "#ffd166", "#cfe8ff", "#d4f5d4", "#ffd6d6"
YD, BD, GD, RD = "#c79a00", "#3b6ea8", "#2e8b57", "#c0392b"


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


# ---------------- 03: Wilson ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
for ax, n, col in zip(axes, [10, 20, 50], [RD, YD, GD]):
    ks = np.arange(0, n + 1)
    lo = np.array([wilson(k, n)[0] for k in ks]); hi = np.array([wilson(k, n)[1] for k in ks])
    x = ks / n * 100
    ax.fill_between(x, lo * 100, hi * 100, color=col, alpha=0.18, lw=0)
    ax.plot(x, lo * 100, color=col, lw=1.6); ax.plot(x, hi * 100, color=col, lw=1.6)
    ax.plot(x, x, color="#555", lw=0.8, ls="--")
    ax.set_title(f"n = {n} episodes", fontsize=11)
    ax.set_xlabel("observed success rate (%)")
    ax.grid(alpha=0.25)
    for k in ([5, 9] if n == 10 else [10, 18] if n == 20 else [25, 45]):
        l, h = wilson(k, n)
        high = k / n > 0.8
        ax.annotate(f"{k}/{n}: {100*l:.0f}–{100*h:.0f}", (k / n * 100, (l * 100 - 7) if high else (h * 100 + 2)),
                    ha="right" if high else "center", fontsize=9, color=col)
    # 10 episodes: do the 9/10 and 5/10 intervals overlap?
    if n == 10:
        l9, h9 = wilson(9, 10); l5, h5 = wilson(5, 10)
        ax.axhspan(l9 * 100, h5 * 100, color=R, alpha=0.5, lw=0)
        ax.text(2, (l9 + h5) / 2 * 100, f"the 9/10 and 5/10\nintervals overlap\n({100*l9:.0f}–{100*h5:.0f})", fontsize=8.5, color=RD, va="center")
axes[0].set_ylabel("Wilson 95% interval (%)")
fig.suptitle("Lecture 18 — Wilson 95% confidence interval for k/n: 10 episodes give a 'direction', not a 'number' (exact computation)", fontsize=12)
fig.tight_layout(); fig.savefig(out + "03-wilson-n10.png", dpi=160); plt.close(fig)


# ---------------- 04: power ----------------
def power_fisher(p1, p2, n, alpha=0.05):
    pw = 0.0
    for k1 in range(n + 1):
        q1 = binom.pmf(k1, n, p1)
        if q1 < 1e-12:
            continue
        for k2 in range(n + 1):
            q2 = binom.pmf(k2, n, p2)
            if q2 < 1e-12:
                continue
            if fisher_exact([[k1, n - k1], [k2, n - k2]])[1] < alpha:
                pw += q1 * q2
    return pw


def mcnemar_p(b, c):
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)


def power_mcnemar(p1, p2, n, alpha=0.05):
    # pair types: (✓,✓) p2 ; (✓,✗) p1-p2 ; (✗,✗) 1-p1 ; (✗,✓) 0  — optimistic paired model
    pb = p1 - p2
    pw = 0.0
    for b in range(n + 1):  # b discordant pairs (clean✓, x✗); c = 0 → p = 2 * 0.5**b
        pr = binom.pmf(b, n, pb)
        if mcnemar_p(b, 0) < alpha:
            pw += pr
    return pw


P1 = 0.95
px = np.round(np.arange(0.05, 0.951, 0.05), 2)
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
for ax, (fn, title) in zip(axes, [(power_fisher, "(a) unpaired: Fisher exact (two independent groups)"),
                                  (power_mcnemar, "(b) paired: McNemar exact (same init state; optimistic model: one-directional discordance)")]):
    for n, col, ls in [(10, RD, "-"), (20, YD, "-"), (50, GD, "-")]:
        pw = [fn(P1, p, n) for p in px]
        ax.plot((P1 - px) * 100, np.array(pw) * 100, color=col, lw=2, ls=ls, label=f"n = {n}")
        # the smallest difference at which power exceeds 80% for this n
        ok = [(P1 - p) * 100 for p, w in zip(px, pw) if w >= 0.8]
        if ok:
            ax.annotate(f"≥80% power: Δ ≥ {min(ok):.0f} points", (min(ok), 80), xytext=(min(ok) + 3, 80 - (0 if n == 10 else 9 if n == 20 else 18)),
                        fontsize=9, color=col, arrowprops=dict(arrowstyle="-", color=col, lw=0.8))
    ax.axhline(80, color="#777", lw=0.8, ls=":")
    ax.set_xlabel("true difference: p_clean − p_x (points); p_clean = 0.95")
    ax.set_title(title, fontsize=10.5); ax.grid(alpha=0.25); ax.legend(frameon=False, loc="lower right")
    for ref, yy, lab in [(0.95 - 0.66, 30, "π0.5 Spatial empty command\nΔ≈29 (97.5→66, Chen et al.)"), (0.95 - 0.80, 8, "OpenVLA-OFT Spatial\nempty Δ≈14 (93.5→80)")]:
        ax.axvline(ref * 100, color="#999", lw=0.8, ls="--"); ax.text(ref * 100 + 0.8, yy, lab, fontsize=7.5, color="#555")
axes[0].set_ylabel("power: probability of catching the difference at α=0.05 (%)")
fig.suptitle("Lecture 18 — Which difference do 10 / 20 / 50 episodes reliably catch? (exact power computation; assuming p_clean = 0.95)", fontsize=12)
fig.tight_layout(); fig.savefig(out + "04-power.png", dpi=160); plt.close(fig)

# text summary (numbers used in the lecture)
for n in (10, 20, 50):
    f29 = power_fisher(P1, 0.66, n); m29 = power_mcnemar(P1, 0.66, n)
    f15 = power_fisher(P1, 0.80, n); m15 = power_mcnemar(P1, 0.80, n)
    f55 = power_fisher(P1, 0.40, n); m55 = power_mcnemar(P1, 0.40, n)
    print(f"n={n:2d}  Δ=15: Fisher {100*f15:4.0f}%  McNemar {100*m15:4.0f}% | Δ=29: Fisher {100*f29:4.0f}%  McNemar {100*m29:4.0f}% | Δ=55: Fisher {100*f55:4.0f}%  McNemar {100*m55:4.0f}%")
print("Wilson 9/10: %.3f–%.3f  5/10: %.3f–%.3f  10/10: %.3f–%.3f  0/10: %.3f–%.3f" % (*wilson(9, 10), *wilson(5, 10), *wilson(10, 10), *wilson(0, 10)))
print("McNemar smallest p, n=10, b discordant: " + ", ".join(f"b={b}: {mcnemar_p(b,0):.4f}" for b in (3, 4, 5, 6, 10)))
```

## plot_day18.py
```python
"""Lecture 18 — figures 02 and 06: the 'empty command' and 'wrong command' numbers in the literature (transcribed by hand; no model/GPU needed).

  python plot_day18.py <output_folder/>

02-empty-literature.png  Correct English command vs empty command (Ø) on LIBERO-Spatial, per model.
   Sources: Chen et al. 2026 (arXiv:2606.15714) Table 6 — Spatial columns en / Ø (the OpenVLA-OFT row in the paper is the same as
   ABot-M0; it contradicts the paper's own Table 1 → marked 'suspicious'); Zhan et al. 2026 (arXiv:2601.04052) Table 8/9 — π0 and π0.5
   correct / blank; Fang et al. 2026 (arXiv:2602.17659) Table I — π0.5 vision-only (language input removed) Spatial 68.2.
06-wrong-literature.png  LIBERO-CF CF-Spatial (Fang et al. Table III, without CAG): which task does the model perform under a wrong (counterfactual) command?
   Grounding/Success × Faithful (faithful to the command) / Biased (performs the task from training).
"""
import os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out = (sys.argv[1] if len(sys.argv) > 1 else ".").rstrip("/") + "/"; os.makedirs(out, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
Y, B, G, R = "#ffd166", "#cfe8ff", "#d4f5d4", "#ffd6d6"
YD, BD, GD, RD = "#c79a00", "#3b6ea8", "#2e8b57", "#c0392b"

# ---------- 02 ----------
# (label, en, empty, source abbreviation, note)
ROWS = [
    ("OpenVLA-OFT", 93.5, 80.0, "Chen T6", "row suspicious"),
    ("π0.5", 97.5, 66.0, "Chen T6", ""),
    ("π0.5", 95.4, 65.0, "Zhan T8/9", ""),
    ("π0.5 (vision-only)", 97.2, 68.2, "Fang T1", "no language input"),
    ("π0", 96.8, 29.8, "Zhan T8/9", ""),
    ("Cosmos Policy", 96.5, 55.5, "Chen T6", ""),
    ("Qwen3-VL-OFT", 92.5, 34.0, "Chen T6", ""),
    ("Qwen3-VL-FAST", 89.0, 12.5, "Chen T6", ""),
    ("Qwen3-VL-GR00T", 94.0, 64.5, "Chen T6", ""),
    ("Qwen3-VL-π", 93.0, 41.0, "Chen T6", ""),
    ("Qwen2.5-VL-OFT", 91.5, 66.5, "Chen T6", ""),
    ("Qwen2.5-VL-FAST", 86.5, 2.0, "Chen T6", ""),
    ("Qwen2.5-VL-GR00T", 95.5, 62.5, "Chen T6", ""),
    ("Qwen2.5-VL-π", 88.0, 52.5, "Chen T6", ""),
]
fig, ax = plt.subplots(figsize=(15, 6.4))
y = np.arange(len(ROWS))[::-1]
for yi, (lab, en, em, src, note) in zip(y, ROWS):
    ax.plot([em, en], [yi, yi], color="#bbb", lw=3, solid_capstyle="round", zorder=1)
    ax.scatter(en, yi, s=80, color=GD, zorder=3, edgecolor="white", lw=1)
    ax.scatter(em, yi, s=80, color=RD, zorder=3, edgecolor="white", lw=1)
    ax.text(en + 1.2, yi, f"{en:.1f}", va="center", fontsize=9, color=GD)
    ax.text(em - 1.2, yi, f"{em:.1f}", va="center", ha="right", fontsize=9, color=RD)
    ax.text(103, yi, f"Δ −{en - em:.0f}  ·  {src}" + (f"  ({note})" if note else ""), va="center", fontsize=8.5, color="#555")
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in ROWS])
ax.set_xlim(-2, 135); ax.set_xlabel("LIBERO-Spatial success rate (%)")
ax.axvspan(29.8, 68.2, color=R, alpha=0.25, lw=0)
ax.set_ylim(-1.6, len(ROWS) - 0.4)
ax.text(49, -1.1, "share of Spatial 'solvable from the image' with an empty command: 30–68% for most models", ha="center", fontsize=9, color=RD)
ax.scatter([], [], s=80, color=GD, label="correct English command"); ax.scatter([], [], s=80, color=RD, label="empty command (Ø) / no language input")
ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.grid(axis="x", alpha=0.25)
ax.set_title("Lecture 18 — the 'empty command' effect in the literature, LIBERO-Spatial (real paper values; Chen et al. 2606.15714 T6, Zhan et al. 2601.04052 T8–9, Fang et al. 2602.17659 T1)",
             fontsize=10.5)
fig.tight_layout(); fig.savefig(out + "02-empty-literature.png", dpi=160); plt.close(fig)

# ---------- 06 ----------
CF = {  # CF-Spatial, without CAG: (grounding faithful, grounding biased, success faithful, success biased)
    "OpenVLA-OFT": (6.8, 84.7, 1.1, 80.7),
    "π0": (70.9, 37.1, 34.4, 19.1),
    "π0.5": (39.3, 61.3, 24.4, 56.9),
}
fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
for ax, (title, i_f, i_b) in zip(axes, [("Grounding: which object did the gripper touch?", 0, 1), ("Success: which task was completed?", 2, 3)]):
    x = np.arange(len(CF)); w = 0.36
    f = [v[i_f] for v in CF.values()]; b = [v[i_b] for v in CF.values()]
    ax.bar(x - w / 2, f, w, color=B, edgecolor=BD, label="Faithful — target of the wrong command (read the command)")
    ax.bar(x + w / 2, b, w, color=R, edgecolor=RD, label="Biased — the original task from training (went by the image)")
    for xi, (fv, bv) in enumerate(zip(f, b)):
        ax.text(xi - w / 2, fv + 1.5, f"{fv:.1f}", ha="center", fontsize=9, color=BD)
        ax.text(xi + w / 2, bv + 1.5, f"{bv:.1f}", ha="center", fontsize=9, color=RD)
    ax.set_xticks(x); ax.set_xticklabels(list(CF)); ax.set_ylim(0, 100); ax.set_title(title, fontsize=11); ax.grid(axis="y", alpha=0.25)
axes[0].set_ylabel("%"); axes[0].legend(frameon=False, fontsize=9, loc="upper right")
fig.suptitle("Lecture 18 — What happens when a 'wrong command' is given? LIBERO-CF, CF-Spatial (Fang et al. 2026, arXiv:2602.17659, Table III, without CAG)\n"
             "In our 'wrong' condition, success = the Biased column; you can only count Faithful from the video (which bowl?)", fontsize=10.5)
fig.tight_layout(); fig.savefig(out + "06-wrong-literature.png", dpi=160); plt.close(fig)
print("02-empty-literature.png, 06-wrong-literature.png →", out)

# ---------- 07: possible patterns (ILLUSTRATIVE — not real data; the reading key for Lecture 19) ----------
PAT = [
    ("A) Language is ignored\n(pure shortcut)", [90, 85, 85, 85], "clean ≈ empty ≈ wrong ≈ tr\n→ Spatial task 0 is solved from the scene"),
    ("B) Language is read, Turkish is not", [90, 40, 40, 40], "tr ≈ empty: nothing is extracted from Turkish\n(Chen et al.: the OFT/π0.5 pattern)"),
    ("C) Wrong command is 'biased'", [90, 40, 80, 40], "wrong ≈ clean but empty is low:\nthe command says 'pick a bowl', the scene picks which one"),
    ("D) The multilingual backbone helps", [90, 40, 40, 75], "tr > empty: Qwen3-VL extracts at least the\nobject/action from Turkish (hypothesis)"),
]
fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), sharey=True)
cols = [GD, RD, RD, YD]
for ax, (title, vals, note) in zip(axes, PAT):
    ax.bar(["clean", "empty", "wrong", "tr"], vals, color=[G, R, R, Y], edgecolor=cols, lw=1.2)
    ax.set_ylim(0, 128); ax.set_yticks([0, 20, 40, 60, 80, 100]); ax.set_title(title, fontsize=10.5); ax.grid(axis="y", alpha=0.25)
    ax.text(1.5, 124, note, ha="center", va="top", fontsize=8.5, color="#444")
axes[0].set_ylabel("success rate (%) — ILLUSTRATIVE")
fig.suptitle("Lecture 18 — four possible patterns of the 4-row table (ILLUSTRATIVE values; the real numbers will come from your run, interpretation in Lecture 19)", fontsize=11)
fig.tight_layout(); fig.savefig(out + "07-expected-patterns.png", dpi=160); plt.close(fig)
print("07-expected-patterns.png →", out)
```

## conditions_task0.json
```json
{
  "clean": null,
  "empty": "",
  "wrong": "pick up the black bowl next to the ramekin and place it on the plate",
  "tr": "tabak ile ramekinin arasındaki siyah kaseyi al ve tabağın üstüne koy"
}
```

## t4_cond_config.yaml
```yaml
# Lecture 18 — 4-condition experiment: LIBERO-Spatial task 0, 10 init states × {clean, empty, wrong, tr}, free T4.
# Difference from Lecture 12's t4_bench_config.yaml: num_trials_per_task 1 → 10 (init states 0..9), num_parallel_envs 2 (Kaggle 29 GB RAM: 4 →
# the four conditions of the same init state run in the same batch). The conditions are not in the config but in an environment variable: VLANEXT_CONDITIONS (Lecture 18 patch).
# Usage (from the VLANeXt root, with the Lecture 12 + Lecture 18 patches applied):
#   VLANEXT_TASK_IDS=0 VLANEXT_CONDITIONS="$(cat conditions_task0.json)" VLANEXT_RUN_TAG=_4cond \
#   python -m scripts.libero_bench_eval --config config/t4_cond_config.yaml
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
  finetuned_checkpoint: "checkpoints/VLANeXt_libero_spatial.pt"
  task_suite_name: "libero_spatial"
  num_steps_wait: 10
  num_steps_execute: 8
  num_parallel_envs: 2         # Colab 12.7 GB RAM: 2; Kaggle: 4 (same as the number of conditions → paired batch)
  num_trials_per_task: 10      # init states 0..9; each with 4 conditions → 40 episodes per task
  seed: 42
  image_size: 256
  resume_episodes: 0
  resume_successes: 0
```
