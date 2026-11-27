# Project 03 — VLANeXt on the free T4: first real success rate (LIBERO-Spatial task 0, clean, 10 init states)

**Status:** in progress · **Start:** 2026-11-28 · **Days:** 3 · **Difficulty:** ★ · **Related lectures:** [12](../../lectures/lecture-12.md), [13](../../lectures/lecture-13.md), [19](../../lectures/lecture-19.md) · **Closes handoff item:** Lecture 12 first T4 eval

## Goal
Run a published VLA (VLANeXt-B, Spatial checkpoint) on LIBERO-Spatial **task 0** (*pick up the black bowl between the plate and the ramekin and place it on the plate*) on a free T4, with the benchmark's own English command (`clean`), over the **first 10 init states**, and produce the first real success rate of this thesis — logged in `results/runs.csv`, with `eval.log` and two rollout videos as evidence. This number is the ceiling that Project 04's `empty / wrong / tr` conditions will be compared against.

## Hypothesis (written before the run)
- H1: success on task 0, 10 init states, is **≥ 8/10**. Reason: the paper reports ≈97% on the whole Spatial suite (Lecture 12, figure 03) and Lecture 19's structural reading gives task 0 a high ceiling. Falsified if ≤ 6/10.
- H2: one episode on the T4 takes **2–6 minutes** wall-clock (Lecture 27's episode-time model, mid scenario). Falsified if the median is > 10 min (then Project 04's budget must be recomputed).
- Expected failure mode, if any: wrong bowl picked (the one *next to* the plate instead of *between*), not a grasp failure.

## Setup
Colab (Runtime → T4 GPU) or Kaggle (GPU T4 x2, 29 GB RAM). Everything is in the Lecture 12 notebook:
https://colab.research.google.com/github/mesutd0nmez/vla-robot-learning-tr/blob/main/code/day12/colab_vlanext_t4.ipynb
Patches (`code/day12/vlanext_t4_patch.py`): flash-attention → SDPA, `torch.load(mmap=True)`, `VLANEXT_TASK_IDS` task filter. Config: `code/day12/t4_bench_config.yaml`.

## Running
```bash
# Day 1 — smoke: 1 episode, task 0
VLANEXT_TASK_IDS=0  (num_trials_per_task: 1)   python -m scripts.libero_bench_eval --config config/t4_bench_config.yaml
# Day 2 — the run: 10 episodes, task 0
VLANEXT_TASK_IDS=0  (num_trials_per_task: 10)  python -m scripts.libero_bench_eval --config config/t4_bench_config.yaml
python parse_eval_log.py <eval.log> --run-id vlanext_sp0_clean_01 --lang en --condition correct --csv results/runs.csv
```
Day-by-day instructions, adjusted each morning to the real state: [`coach/day-1.md`](coach/day-1.md), `coach/day-2.md`, `coach/day-3.md`. Your own notes and blockers: [`progress.md`](progress.md).

## Results
_(filled on Day 3)_

| run_id | checkpoint | suite | task | condition | n | successes | success rate | Wilson 95% CI | median episode time |
|---|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | — | — |

Raw: `results/runs.csv`, `results/eval.log`, `results/episodes.csv`; videos: `images/ep-success.mp4`, `images/ep-fail.mp4` (or frame strips if the mp4 is too big).

## Findings and notes
_(Day 3: answer H1 and H2 explicitly; describe the failure mode you saw; what this number changes in `thesis/experiment-matrix.md` (t_call measured scenario) and in Project 04's plan.)_

## Sources
- Tang et al. (2026). VLANeXt. [arXiv:2602.18532](https://arxiv.org/abs/2602.18532) · [github.com/DravenALG/VLANeXt](https://github.com/DravenALG/VLANeXt)
- Liu et al. (2023). LIBERO. [arXiv:2306.03310](https://arxiv.org/abs/2306.03310)
- Repo: Lecture 12 notebook and patches (`code/day12/`), Lecture 19 init-state features (`datasets/03-libero-spatial0-4cond/data/init_features_task0.csv`)
