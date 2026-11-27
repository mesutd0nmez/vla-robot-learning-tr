# Day 1 — Setup and smoke test (≈ 60 min)

**Goal of the day:** by the end of the hour the T4 notebook has run **one** complete episode of task 0 and you have an `eval.log` with an `Episode result:` line. Nothing else. Success rate is tomorrow's job.

**Before you start (2 min):** on the Mac, `git pull`; open this file and `progress.md` side by side. You will write to `progress.md` at minute 55 whatever happens.

## Step 1 — Open the notebook on a T4 (5 min)
1. Open https://colab.research.google.com/github/mesutd0nmez/vla-robot-learning-tr/blob/main/code/day12/colab_vlanext_t4.ipynb
2. Runtime → Change runtime type → **T4 GPU** → Save. (Kaggle alternative: Settings → Accelerator → GPU T4 x2; 29 GB RAM is more comfortable than Colab's 12.7 GB.)
3. Run **cell 1**. Expected: `Tesla T4`, `cc (7, 5)`, RAM ≥ 12 GB, disk free ≥ 60 GB. If the GPU line says anything else, you are not on a T4 — fix the runtime before going on.

**Checkpoint A:** cell 1 output pasted into `progress.md` (3 lines are enough).

## Step 2 — System libraries + install (12–15 min)
4. Run **cell 2** (apt + EGL). Expected: `ok`.
5. Run **cell 3** (clone VLANeXt + LIBERO, pip). This is the slow cell. Expected last line: an `import` check without a traceback.
   - If pip ends with `ERROR: ... numpy`, rerun the cell once; the pins in the cell are the ones that worked in Lecture 12.
   - If `ModuleNotFoundError: libero` appears later, the `PYTHONPATH` line in cell 6 fixes it; do not reinstall.
6. Run **cell 4** (patches + config). Expected: three `[+] ... applied` lines (or `[=] already patched` on a rerun) and no `ERROR:`. If it says `expected line not found`, the VLANeXt repo changed since Lecture 12 — stop, paste the message into `progress.md`, continue with Step 3 anyway (the download does not depend on the patch).

**Checkpoint B:** the three patch lines in `progress.md`.

## Step 3 — Checkpoint download (5–10 min, runs in the background of your attention)
7. Run **cell 5**. Expected: `VLANeXt_libero_spatial.pt` ≈ 16.4 GB in `checkpoints/`. While it downloads, read Step 4 and open `code/day13/README.md` in the repo to recall where `task_description` enters the eval loop.

## Step 4 — Smoke: ONE episode of task 0 (10–15 min)
8. Before running cell 6, edit two things:
   - In the cell, change `os.environ["VLANEXT_TASK_IDS"] = "0,2,8"` to `"0"`.
   - `config/t4_bench_config.yaml` already has `num_trials_per_task: 1` — leave it.
9. Run **cell 6**. The first model load is slow (Qwen3-VL-2B download ≈ 4.5 GB + 16.4 GB state_dict copy): 3–6 minutes of nothing visible is normal. Then one episode. Expected somewhere in the output:
   `Episode result: task=1/10 trial=0 success=True|False rollout_steps=... episode_time=...`
   and at the end `Eval finished: ... final_success_rate=... elapsed=...`.
10. Note the `episode_time` value — that is H2's first data point. Write it into `progress.md`.

**Checkpoint C (the day's goal):** the `Episode result:` line exists.

## Step 5 — Save the evidence (5 min)
11. Run **cell 7**. It writes `results/runs.csv` in the Colab filesystem and prints a table. Download `eval.log` and `results/runs.csv` from the Files panel (left bar) to your Mac, into `projects/03-vlanext-t4-first-eval/results/` as `eval-day1-smoke.log` and `runs-day1-smoke.csv` (keep the day-1 files separate; tomorrow's run overwrites nothing).
12. Run **cell 8** once to see the video. Do not download it today.

## Step 6 — Close the hour (5 min)
13. `progress.md` → Day 1: Checkpoint A/B/C status, episode_time, anything that broke (last 20 lines of the error, verbatim).
14. Commit and push from the Mac:
    ```bash
    git add projects/03-vlanext-t4-first-eval && git commit -m "Project 03 day 1: T4 setup + smoke episode" && git push
    ```
15. **Keep the Colab tab open** if you will continue within ~90 minutes; otherwise accept that tomorrow starts with cells 2–5 again (≈ 20 min). Kaggle keeps `/kaggle/working` between sessions only if you save a version — Colab does not.

## If you get stuck
| Symptom | Do |
|---|---|
| `CUDA out of memory` during model load | Runtime → Restart; rerun cells 2, 4, 6 (not 3 and 5); make sure `num_parallel_envs: 2` → set it to `1` in the yaml |
| Kernel dies silently in cell 6 | RAM, not GPU: Colab 12.7 GB is tight. Check cell 4 really applied the `mmap=True` patch; otherwise switch to Kaggle |
| `EGL` / `GLEW` / `cannot open display` | cell 2 did not run in this session; rerun it, then cell 6 |
| Download in cell 5 stops at ~10 GB | Colab disk full: `!df -h /` ; delete `third_party/LIBERO/.git` and the pip cache (`!rm -rf ~/.cache/pip`), rerun cell 5 |
| No `Episode result:` line but `Eval finished` | the task filter removed everything: `VLANEXT_TASK_IDS` must be exactly `"0"` |

Whatever happens, Checkpoint A–C status goes into `progress.md`. Tomorrow's `coach/day-2.md` starts from there.
