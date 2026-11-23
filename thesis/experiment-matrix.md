# Experiment matrix — v0.1 (Lecture 27)

Machine-read file: `experiment-matrix.csv` (19 cells). Budget: `python code/day27/budget.py` (medium scenario pilot + core ≈ 258 T4 hours; lean matrix, 20 inits per task ≈ 108 hours). Details and assumptions: `lectures/lecture-27.md`.

Cell = system × suite × block × condition set × inits per task × repetition. `tier`: pilot / core / ext (extension). `status`: ready or blocked (prerequisite written).

Open decisions (Mesut / advisor):
- **Claim unit:** is the main claim of RQ-a at the task level (n_init 50, MDE 23) or the block level (n_init 20, pool n=120, MDE 13)?
- **Repetition:** 1 in the core, PI only in the pilot (P1) — a change proposal that will go into `rq-candidates.md` v0.2.
- **S-OV = 4-bit NF4** (T4); the K1 (against the published 84.7) and K3 (paired with Mac bf16) calibrations are mandatory.

Claim unit decision and quota (Mesut fills in):
```
claim unit: task / block   ; n_init (core): __   ; budget.py medium total: ___ h ; my Kaggle quota: ___ h/week
```

Version: v0.1 — Lecture 27 (automatic draft). Changes are added here with a date.
