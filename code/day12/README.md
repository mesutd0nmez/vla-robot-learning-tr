# day12

Lecture 12 — Running VLANeXt on LIBERO Spatial on the free T4 (Colab/Kaggle). Order: notebook → patch → config → log parsing.

| File | What |
|---|---|
| `colab_vlanext_t4.ipynb` | 8-cell notebook: hardware check, install, patches, checkpoint, eval, result |
| `vlanext_t4_patch.py` | 3 one-line patches to the VLANeXt source (sdpa, mmap, task filter); idempotent |
| `t4_bench_config.yaml` | Small-budget version of `libero_bench_config.yaml` (2 envs, 1 episode per task) |
| `parse_eval_log.py` | `eval.log` → episode table + `results/runs.csv` rows |
| `plot_paper_results.py` | Produces the lecture's 03/04 plots from the paper's numbers |

To open the notebook in Colab: https://colab.research.google.com/github/mesutd0nmez/vla-robot-learning-tr/blob/main/code/day12/colab_vlanext_t4.ipynb

## vlanext_t4_patch.py

```python
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
```

## parse_eval_log.py

```python
"""Day 12 — Read a VLANeXt eval.log file; produce an episode table + runs.csv rows.
Usage:  python parse_eval_log.py <eval.log> [--run-id vlanext_spatial_t4_01] [--lang en] [--condition correct] [--csv results/runs.csv]
The log format is the "Episode result: ..." and "Eval finished: ..." lines from scripts/libero_bench_eval.py.
The runs.csv columns follow the convention in the repo's CLAUDE.md:
  run_id, model, checkpoint, suite, task_id, lang, condition, n_episodes, success_rate
"""
import re, sys, csv, argparse, pathlib
from collections import defaultdict

p = argparse.ArgumentParser()
p.add_argument("log")
p.add_argument("--run-id", default="vlanext_t4_01")
p.add_argument("--lang", default="en")
p.add_argument("--condition", default="correct", help="correct | empty | wrong | paraphrase | tr")
p.add_argument("--csv", default=None, help="if given, appends the rows to this CSV (writes the header if missing)")
args = p.parse_args()

text = pathlib.Path(args.log).read_text(encoding="utf-8", errors="replace")
ep_re = re.compile(r"Episode result: task=(\d+)/(\d+) trial=(\d+) success=(True|False) rollout_steps=(\d+) episode_time=(\S+)")
suite = re.search(r"Task suite: (\S+)", text)
ckpt  = re.search(r"checkpoint=(\S+)", text)
suite = suite.group(1) if suite else "?"
ckpt  = pathlib.Path(ckpt.group(1)).name if ckpt else "?"

rows, per_task = [], defaultdict(lambda: [0, 0])
for m in ep_re.finditer(text):
    task, n_tasks, trial, ok, steps, t = m.groups()
    task_id = int(task) - 1
    ok = ok == "True"
    per_task[task_id][0] += 1; per_task[task_id][1] += int(ok)
    rows.append((task_id, int(trial), ok, int(steps), t))

if not rows:
    sys.exit("No 'Episode result' line in the log — the eval may have crashed before starting; look at the end of the file.")

print(f"suite={suite}  checkpoint={ckpt}  episodes={len(rows)}")
print(f"{'task':>4} {'trial':>5} {'success':>7} {'steps':>5} {'time':>8}")
for task_id, trial, ok, steps, t in rows:
    print(f"{task_id:>4} {trial:>5} {str(ok):>7} {steps:>5} {t:>8}")
tot_n = sum(v[0] for v in per_task.values()); tot_s = sum(v[1] for v in per_task.values())
print(f"\nTotal: {tot_s}/{tot_n} = {100*tot_s/tot_n:.1f}%")
fin = re.search(r"Eval finished: .*final_success_rate=([\d.]+)%.*elapsed=(\S+)", text)
if fin:
    print(f"Script's own summary: SR={fin.group(1)}%  time={fin.group(2)}")

if args.csv:
    out = pathlib.Path(args.csv); out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["run_id", "model", "checkpoint", "suite", "task_id", "lang", "condition", "n_episodes", "success_rate"])
        for task_id in sorted(per_task):
            n, s = per_task[task_id]
            w.writerow([args.run_id, "VLANeXt-B", ckpt, suite, task_id, args.lang, args.condition, n, f"{s/n:.3f}"])
    print(f"{len(per_task)} rows appended → {out}")
```

## plot_paper_results.py

```python
"""Day 12 — Plots the LIBERO and LIBERO-plus (Language) numbers from the VLANeXt paper.
Usage: python plot_paper_results.py [output_folder/]   → 03-libero-results.png, 04-libero-plus-language.png
Numbers: tables of arXiv:2602.18532v2 (VLANeXt); transcribed by hand.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
import os, sys
out = (sys.argv[1] if len(sys.argv) > 1 else "./") ; os.makedirs(out, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# --- 03: LIBERO per-suite, VLANeXt paper Table (arXiv 2602.18532v2)
suites = ["Spatial", "Object", "Goal", "Long", "Avg."]
models = [("OpenVLA (2024)", "#8c6bb1", [84.7, 88.4, 79.2, 53.7, 76.5]),
          ("OpenVLA-OFT (2025)", "#3a9a5c", [97.6, 98.4, 97.9, 94.5, 97.1]),
          ("π0.5 (2025)", "#3d7fc4", [98.8, 98.2, 98.0, 92.4, 96.9]),
          ("VLANeXt-B (2026)", "#e09a00", [99.0, 99.2, 96.6, 94.8, 97.4])]
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(suites)); w = 0.2
for i, (name, c, v) in enumerate(models):
    b = ax.bar(x + (i - 1.5) * w, v, w * 0.92, color=c, label=name, zorder=3)
    for r, val in zip(b, v):
        ax.text(r.get_x() + r.get_width() / 2, val + 0.8, f"{val:.1f}", ha="center", va="bottom", fontsize=8, color="#333")
ax.set_xticks(x); ax.set_xticklabels(suites)
ax.set_ylim(40, 104); ax.set_ylabel("Success rate (%)")
ax.yaxis.grid(True, color="#e5e5e5", zorder=0)
ax.set_title("LIBERO: 4 suites, 500 episodes/suite — source: VLANeXt paper table (arXiv:2602.18532v2)", fontsize=10)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False, fontsize=9)
fig.text(0.5, 0.005, "Each suite is 10 tasks × 50 episodes = 500 episodes. Next to these numbers, the 3 episodes you will run today are only a 'does it work?' test.",
        ha="center", fontsize=8.5, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(out + "03-libero-results.png", dpi=180); plt.close(fig)

# --- 04: LIBERO-plus Language perturbation per suite, OFT vs VLANeXt (same paper, LIBERO-plus table)
suites = ["Spatial", "Object", "Goal", "Long"]
oft   = [80.5, 99.0, 53.2, 87.0]
vla   = [86.9, 98.6, 51.5, 90.3]
clean_oft = [97.6, 98.4, 97.9, 94.5]; clean_vla = [99.0, 99.2, 96.6, 94.8]
fig, ax = plt.subplots(figsize=(10, 5.2))
x = np.arange(4); w = 0.36
b1 = ax.bar(x - w/2, oft, w * 0.94, color="#3a9a5c", label="OpenVLA-OFT, Language perturbation", zorder=3)
b2 = ax.bar(x + w/2, vla, w * 0.94, color="#e09a00", label="VLANeXt-B, Language perturbation", zorder=3)
ax.scatter(x - w/2, clean_oft, marker="_", s=700, color="#222", zorder=4, linewidths=2, label="same model, clean LIBERO (line)")
ax.scatter(x + w/2, clean_vla, marker="_", s=700, color="#222", zorder=4, linewidths=2)
for bars, vals in ((b1, oft), (b2, vla)):
    for r, v in zip(bars, vals):
        ax.text(r.get_x() + r.get_width()/2, v - 6, f"{v:.1f}", ha="center", fontsize=9, color="white", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(suites); ax.set_ylim(0, 105); ax.set_ylabel("Success rate (%)")
ax.yaxis.grid(True, color="#e5e5e5", zorder=0)
ax.set_title("LIBERO-plus 'Language' perturbation: how much does success drop when the command is changed? — source: arXiv:2602.18532v2", fontsize=10)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, fontsize=9)
fig.text(0.5, 0.005, "On Goal both models drop to ~50% (language is really used); on Object they stay at ~99% (the command is almost unnecessary, the shortcut from Lectures 3–4).",
         ha="center", fontsize=8.5, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(out + "04-libero-plus-language.png", dpi=180); plt.close(fig)
print("ok")
```

## t4_bench_config.yaml

```yaml
# Day 12 — VLANeXt LIBERO eval, small budget for the free T4.
# Original: VLANeXt/config/libero_bench_config.yaml (num_parallel_envs: 32, num_trials_per_task: 50)
# Usage: from the VLANeXt root  python -m scripts.libero_bench_eval --config config/t4_bench_config.yaml
data:
  augmentation:
    center_crop: true          # training used random crop; eval uses center crop (original setting)
    center_crop_ratio: 0.9

model:
  diffusion_steps: 5           # flow-matching steps; original is 5. Do not lower it to speed up, quality drops
  scheduler_type: "flow_match"

project:
  use_wandb: false             # do not ask for a wandb account

eval:
  finetuned_checkpoint: "checkpoints/VLANeXt_libero_spatial.pt"   # the file you downloaded with hf download (16.4 GB)
  task_suite_name: "libero_spatial"   # libero_spatial | libero_object | libero_goal | libero_10
  num_steps_wait: 10           # first 10 steps are no-ops (let objects settle) — same as the OpenVLA protocol
  num_steps_execute: 8         # all 8 actions of the 8-step chunk are executed, then the model is called again
  num_parallel_envs: 2         # T4 + 12 GB RAM: 1–2. Each env is a separate process (~1 GB RAM)
  num_trials_per_task: 1       # 1 episode per task (original 50). Also narrow the tasks with VLANEXT_TASK_IDS
  seed: 42
  image_size: 256
  resume_episodes: 0
  resume_successes: 0
```

## colab_vlanext_t4.ipynb (cells)

# Lecture 12 — Run VLANeXt on LIBERO on the free T4

**First:** Runtime → Change runtime type → **T4 GPU**. If you are on Kaggle: Settings → Accelerator → **GPU T4 x2** (29 GB RAM; safer than Colab's 12.7 GB).

Flow: (1) verify the hardware → (2) system libraries → (3) install VLANeXt + LIBERO → (4) T4 patches → (5) download the checkpoint (16.4 GB) → (6) eval on 2–3 episodes → (7) read the result, watch the video, produce a `runs.csv` row.

Total time (hypothesis, T4): install ~10 min + download ~5 min + 3 episodes ~5 min. Run the cells in order; if you get an error, paste the **whole** cell output into the chat.

```python
# 1) Hardware check — compute capability 7.5 = T4 (NO FlashAttention-2, NO hardware bf16 support → patches needed)
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
import torch, psutil, shutil
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| GPU", torch.cuda.get_device_name(0), "| cc", torch.cuda.get_device_capability(0))
print("RAM %.1f GB | disk free %.0f GB" % (psutil.virtual_memory().total/1e9, shutil.disk_usage("/").free/1e9))
print("bf16 (including emulation):", torch.cuda.is_bf16_supported())
```

```python
# 2) System libraries MuJoCo needs for headless (EGL) rendering — VLANeXt COMMON_ISSUES #1
!apt-get -qq update && apt-get -qq install -y libgl1-mesa-dev libegl1-mesa-dev libxrandr2 libxcursor1 libxinerama1 libxrender1 > /dev/null
import os; os.environ["MUJOCO_GL"] = "egl"; os.environ["PYOPENGL_PLATFORM"] = "egl"
print("ok")
```

```python
# 3) VLANeXt + LIBERO. We install what the eval needs, NOT the whole requirements.txt
#    (no torch 2.4 / flash-attn / deepspeed / dm-reverb). mujoco 3.8.1 is VLANeXt's own pin; it works with robosuite 1.4.1 on Linux.
%cd /content
!git clone -q https://github.com/DravenALG/VLANeXt && echo "clone ok"
%cd /content/VLANeXt
!mkdir -p third_party && git clone -q https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
!pip install -q "transformers==5.5.0" "diffusers==0.36.0" accelerate huggingface_hub sentencepiece h5py opencv-python "imageio[ffmpeg]" \
    "robosuite==1.4.1" "mujoco==3.8.1" bddl easydict cloudpickle gym pyyaml wandb tqdm tensorflow-datasets rlds 2>&1 | tail -2
!pip install -q ./third_party/LIBERO 2>&1 | tail -1
# On first import LIBERO asks "custom path? (Y/N)"; we feed the N answer through a pipe
!printf 'N\n' | python -c "import libero.libero; print('LIBERO ok')"
```

```python
# 4) T4 patches: flash_attention_2 → sdpa, torch.load mmap=True, VLANEXT_TASK_IDS task filter
!curl -sO https://raw.githubusercontent.com/mesutd0nmez/vla-robot-learning-tr/main/code/day12/vlanext_t4_patch.py
!curl -s -o config/t4_bench_config.yaml https://raw.githubusercontent.com/mesutd0nmez/vla-robot-learning-tr/main/code/day12/t4_bench_config.yaml
!python vlanext_t4_patch.py /content/VLANeXt
```

```python
# 5) Checkpoint (16.4 GB; trained for the Spatial suite). The Qwen3-VL-2B backbone downloads automatically on the first model load (~4.5 GB).
!hf download DravenALG/VLANeXt VLANeXt_libero_spatial.pt --local-dir checkpoints
!ls -lh checkpoints
```

```python
# 6) Eval: only tasks 0, 2 and 8 (between / table center / next to the plate), 1 episode per task.
#    The first model load takes a few minutes (Qwen download + copying the 16.4 GB state_dict).
import os
os.environ["VLANEXT_TASK_IDS"] = "0,2,8"
os.environ["PYTHONPATH"] = "/content/VLANeXt/third_party/LIBERO"
os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
os.environ.pop("LIBERO_CONFIG_PATH", None)     # COMMON_ISSUES #6
!CUDA_VISIBLE_DEVICES=0 python -m scripts.libero_bench_eval --config config/t4_bench_config.yaml 2>&1 | grep -v Warning | tee eval_stdout.txt
```

```python
# 7) Result: turn the log into a table, produce a runs.csv row, list the videos
!curl -sO https://raw.githubusercontent.com/mesutd0nmez/vla-robot-learning-tr/main/code/day12/parse_eval_log.py
import glob
logs = sorted(glob.glob("checkpoints/libero_spatial_checkpoint_*/eval.log"))
print(logs)
!python parse_eval_log.py {logs[-1]} --run-id vlanext_spatial_t4_01 --csv results/runs.csv
!ls checkpoints/libero_spatial_checkpoint_*/*.mp4
```

```python
# 8) Watch a rollout video inside the notebook (pick a failed one: where does it get stuck?)
from IPython.display import Video
import glob
vids = sorted(glob.glob("checkpoints/libero_spatial_checkpoint_*/*.mp4"))
Video(vids[0], embed=True, width=384)
```

## Next
- Download `results/runs.csv` and `eval.log` (Files panel) → put them in an experiment folder under `projects/` in the repo.
- In Lecture 13 we will read the inside of this code: where the command enters (`spec.task_description`), how the action is decoded (`_denormalize_action`), where success is computed (`env.step` → `done`).
- The only line to change for the Turkish experiment is `task_description = task.language` inside `build_episode_specs` — but we will do that in Lecture 18.
