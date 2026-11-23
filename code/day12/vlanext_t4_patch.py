"""Day 12 — Patch the VLANeXt codebase for the free T4 (Colab/Kaggle).
Usage:  python vlanext_t4_patch.py /content/VLANeXt        (idempotent; a second run says "already patched")

Three patches (all one-liners, source: VLANeXt repo, Nov 2026):
 1. flash_attention_2 -> sdpa      The T4 (sm75) does not support FlashAttention-2; it is hard-coded in the eval code.
 2. torch.load(..., mmap=True)     The checkpoint is 16.4 GB; it does not fit in Colab's 12.7 GB RAM. With mmap the file
                                   is read from disk piece by piece instead of being loaded into RAM entirely.
 3. VLANEXT_TASK_IDS env variable  The default eval runs all 10 tasks; with "0,2" only those tasks run.
"""
import os, sys, re, pathlib

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
utils = root / "src/evaluation/libero_bench/VLANeXt_utils.py"
par   = root / "src/evaluation/libero_bench/parallel_eval_utils.py"
for p in (utils, par):
    if not p.exists():
        sys.exit(f"ERROR: {p} does not exist — the argument must be the VLANeXt repo root")

def patch(path, old, new, label):
    src = path.read_text(encoding="utf-8")
    if new in src:
        print(f"  [=] {label}: already patched"); return
    if old not in src:
        sys.exit(f"ERROR: {label}: expected line not found; the repo may have changed. Looked for: {old!r}")
    path.write_text(src.replace(old, new, 1), encoding="utf-8")
    print(f"  [+] {label}: applied")

print(f"VLANeXt root: {root}")
# 1) T4: no FlashAttention-2 -> PyTorch SDPA
patch(utils,
      'attn_implementation = "flash_attention_2"',
      'attn_implementation = os.environ.get("VLANEXT_ATTN", "sdpa")  # T4 patch (original: flash_attention_2)',
      "attn_implementation -> sdpa")
if "import os" not in utils.read_text(encoding="utf-8").split("\n\n")[0]:
    patch(utils, "from pathlib import Path", "import os\nfrom pathlib import Path", "import os (VLANeXt_utils)")

# 2) Open the 16.4 GB .pt file without copying it into RAM
patch(utils,
      'ckpt = torch.load(str(path), map_location="cpu")',
      'ckpt = torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)  # T4 patch: RAM',
      "torch.load mmap=True")

# 3) Task filter: VLANEXT_TASK_IDS="0,2,5"
patch(par,
      "import numpy as np\n",
      'import os\nimport numpy as np\n_TASK_FILTER = {int(x) for x in os.environ.get("VLANEXT_TASK_IDS", "").split(",") if x.strip()}  # T4 patch\n',
      "VLANEXT_TASK_IDS definition")
patch(par,
      "    for task_id in range(num_tasks):\n        task = task_suite.get_task(task_id)",
      "    for task_id in range(num_tasks):\n        if _TASK_FILTER and task_id not in _TASK_FILTER:\n            continue  # T4 patch\n        task = task_suite.get_task(task_id)",
      "task filter")

print("Done. Check:")
for p in (utils, par):
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if "T4 patch" in line:
            print(f"  {p.name}:{i}: {line.strip()[:90]}")
