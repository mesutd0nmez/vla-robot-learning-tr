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
