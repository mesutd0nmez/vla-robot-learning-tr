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
