# Project ladder — Stage 1 (hands-on)

Stage 0 produced 31 lectures, a thesis proposal draft and an experiment matrix, but the hands-on items were left undone
(`code/day31/results/handoff.md`). Stage 1 closes that gap with a **ladder of 3-day projects**, each harder than the last,
each ending in a written report with goal / hypothesis / results / findings and a row in `results/runs.csv`.

## Rules
- **3 days per project, ~1 hour per day.** Day 1 = setup + smoke test, Day 2 = the real run, Day 3 = analysis + report + commit.
- Every morning at 06:51 (TR) the coach (cloud task, or local Claude Code with `tools/COACH_PROMPT.md` when the task did not run) reads the repo, checks what was committed since yesterday, and writes
  `projects/<project>/coach/day-N.md`: today's step-by-step instructions adjusted to the actual state. On the morning after
  Day 3 it grades the project (rubric below), writes the verdict into the README, and opens the next project from this ladder.
- Mesut works locally with Claude Code: `git pull` → follow `coach/day-N.md` → commit at the end of the hour
  (`Project NN day N: <what happened>`) → push. Blocked? Write the blocker into `progress.md`; the next morning's coach note starts from it.
- A project that is not finished in 3 days gets **one** catch-up day, then is closed as `shelved` with what exists. Difficulty keeps rising regardless.
- Nothing is faked: a result that was not produced is written as "not produced", with the reason.

## Rubric (graded by the coach on the closing morning; 0–2 each, max 10)
| Criterion | 0 | 1 | 2 |
|---|---|---|---|
| Runs | nothing ran | smoke test only | the planned run completed |
| Record | no `runs.csv` row | rows exist, incomplete | every run logged with run_id, checkpoint, n, success_rate |
| Report | README template empty | results filled | hypothesis stated **before** the run, findings answer it |
| Reproducibility | commands missing | commands present | a stranger could rerun from the README alone |
| Thesis link | none | mentioned | result changes a decision in `thesis/` (matrix, ablation plan, RQ) |

## The ladder

| # | Project (folder) | Days | Difficulty | What you can do afterwards | Closes handoff item |
|---|---|---|---|---|---|
| 03 | `03-vlanext-t4-first-eval` — VLANeXt on the free T4: LIBERO-Spatial task 0, clean condition, 10 init states | 1–3 | ★ | run a published VLA on LIBERO and read a success rate | Lecture 12 |
| 04 | `04-spatial0-four-conditions` — the S1 pilot: clean / empty / wrong / tr × 10 init states, Wilson intervals | 4–6 | ★★ | run the thesis's core 4-condition design; first real evidence on the empty-command baseline | Lecture 18 (S1) |
| 05 | `05-openvla-mac-tatar-mini` — OpenVLA-7B on the Mac (MPS): single-frame actions, 5 frames × 6 conditions, L2 + cosine | 7–9 | ★★ | run a second VLA system locally; reproduce Tatar et al. at mini scale | Lectures 14–15 |
| 06 | `06-libero-plus-language-perturbation` — LIBERO-Plus setup on T4; language perturbation on task 0 with the Turkish rewrites | 10–12 | ★★★ | use the robustness benchmark's own perturbation machinery | Lectures 16–17 |
| 07 | `07-two-systems-four-conditions` — OpenVLA-LIBERO checkpoint vs VLANeXt, same 4 conditions, task 0 + one Goal task (design P1) | 13–15 | ★★★ | compare two tokenizers/systems on the same cells; first RQ-a data point | Lecture 26 RQ-a |
| 08 | `08-tr-ascii-letter-cost` — `tr_ascii` surface ablation (letter × suffix 2×2) on the stronger system (design P2) | 16–18 | ★★★★ | isolate letter cost from suffix cost | Lecture 28 ABL-surface |
| 09 | `09-parser-mt-closure` — Turkish → structured → English (parser) vs MT oracle: κ and success on 10 Spatial tasks (design P3) | 19–21 | ★★★★ | build and evaluate the first mitigation layer | Lecture 24 |
| 10 | `10-spatial-block-scale-up` — the lean matrix: 10 Spatial tasks × 4 conditions × n_init per the claim-unit decision, both systems; T4 budget ≈ the Lecture 27 lean total | 22–30 | ★★★★★ | produce the thesis's main table | Lecture 27 matrix |

After 10: Goal block, LoRA fine-tune pilot (Lecture 24 option 4), real-robot/Webots bridge — planned when 10 closes.

## Status
| # | Status | Start | Close | Score |
|---|---|---|---|---|
| 03 | in progress | 2026-11-28 | — | — |
