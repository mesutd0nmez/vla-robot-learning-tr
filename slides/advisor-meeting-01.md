# Advisor meeting 01 — agenda, presentation skeleton, question list, decision log

Prepared by: Mesut Bahtiyar Dönmez · Draft: Lecture 30 (automatic) · Meeting date: ____ · Duration: 30 min
Attachments (by e-mail 2–3 days before): `thesis/proposal-v0.1.md` (Lecture 29, 1 page), this file, `thesis/experiment-matrix.md`.
Question source: `code/day30/questions.csv` (11 questions; each traced to file:line in the repo, `code/day30/results/questions_raw.csv`).

---

## 1. Agenda (30 min)

| min | topic | slide | type |
|---|---|---|---|
| 0–3 | Status: Stage 0 is done; the pipeline is set up, no closed-loop Turkish run (S1) yet | 1 | context |
| 3–7 | Three RQs and the experiment matrix; budget 108 / 258 T4 hours | 2–3 | context |
| 7–12 | **Q1** main suite (S7) + **Q2** claim unit | 4 | decision |
| 12–17 | **Q3** contribution criterion: weak or strong reading (S10) | 5 | decision |
| 17–21 | **Q4** Turkish LoRA / GPU access (S9) | 5 | decision |
| 21–24 | **Q11** thesis proposal schedule and form | 7 | decision |
| 24–27 | Quick approvals: **Q5** 4-bit OpenVLA, **Q6** second reader, **Q10** repetition policy | 6 | approval |
| 27–30 | Wrap-up: owner + date for each decision (template below) | — | closing |

To be asked in writing before the meeting (e-mail): **Q7** role of Webots (S8), **Q8** position relative to Tatar et al., **Q9** the Cohn line.

---

## 2. Presentation skeleton (7 slides + 1 backup)

**Slide 1 — Thesis axis and where we are**
- One sentence: how much do Turkish spatial commands drop in open VLAs (OpenVLA, VLANeXt) in the same scene, why, and how does it close.
- Ready: LIBERO / LIBERO-Plus pipeline, T4 notebooks, the 80-row TR–EN spatial set (`datasets/04`), tokenization analysis.
- Missing: the first closed-loop four-condition table (S1); no model run was done.
- Figure: Lecture 25 figure 01 (Phase 3 map).

**Slide 2 — Three research questions**
- RQ-a is there a difference (tr vs clean vs empty; two systems).
- RQ-b where does it break (relation family, surface, frame; block B).
- RQ-c how does it close (MT, parser, LoRA; weak/strong reading).
- Figure: Lecture 26 figure 03 (experiment space).

**Slide 3 — Experiment matrix and budget**
- 19 cells; pilot + core ≈ 8,400 episodes.
- Medium scenario 258 T4 hours; 108 hours if the claim moves to the block (~2 Kaggle weeks with T4×2).
- S-OV is 63% of the cost (chunk 1, 4-bit on the T4).
- Figure: Lecture 27 figure 03 (hours per cell).

**Slide 4 — Decision: suite and claim unit (Q1, Q2)**
- Spatial: CDC relations but the no-command ceiling is 9/10 (shortcut risk); Goal: ceiling 1/10, the Turkish token side is clean.
- 50 inits per task: the smallest visible difference on a single task is 23 points; in a 6-task block it is 13 points (n=120).
- Our proposal: blocks A + C, claim at block level.
- Figure: Lecture 27 figure 04 (inverse power).

**Slide 5 — Decision: contribution criterion and LoRA (Q3, Q4)**
- Weak reading: correct behavior with a Turkish command (MT/parser suffices). Strong reading: above `empty` on an untrained Turkish recipe.
- Closing the strong reading needs LoRA: 65–98 T4 hours (estimate, 4-bit).
- Our proposal: weak floor + strong target; C3 in parallel if the department has a GPU.
- Figure: Lecture 30 figure 01 (decision tree).

**Slide 6 — Risks and quick approvals (Q5, Q6, Q10)**
- OpenVLA only fits 4-bit on the T4 → K1/K3 calibration; "OpenVLA (4-bit)" in the thesis.
- The Turkish set has a single author → second reader + agreement (κ), 20 rows.
- Repetition only in the pilot (S-OV greedy; a repetition is not an independent sample).
- Figure: Lecture 27 figure 05 (where it runs).

**Slide 7 — Schedule and the next 4 weeks**
- Week 1: S1 (P1) + K1/K2 → real call time → budget updated.
- Weeks 2–3: core cells of the primary RQ; week 4: first table + advisor meeting 02.
- Question: thesis proposal submission date and the institute form (Q11).
- Figure: Lecture 29 Gantt (proposal-v0.1).

**Backup — Position in the literature**
- The Turkish × closed loop × empty-command baseline cell is empty; Tatar et al. is open-loop, single frame.
- Multilingual VLA (Chen 2606.15714, Dong 2606.11906) does not include Turkish.
- Figure: Lecture 26 figure 05 (literature map).

---

## 3. Question list (in priority order; impact × urgency, 1–5 subjective)

| # | question | context | options | our proposal | what it changes | priority |
|---|---|---|---|---|---|---|
| Q1 | Is the main suite Spatial, Goal, or both? (S7) | Spatial ceiling 9/10, Goal 1/10 | Spatial / Goal / both | both (A + C); if `empty` ≥ 80 in Spatial, weight Goal | A1–A4, B1–B2 | 25 |
| Q2 | Is the claim unit the task or the block? | task MDE ≥ 23, block n=120 → 13 | task (258 h) / block (108 h) | block; task Δs in "direction" mode | budget ×2.4, TOST thresholds | 25 |
| Q3 | Is the contribution criterion weak or strong reading? (S10) | strong reading = block B + counterfactual BDDL | weak / strong / both | weak floor + strong target | RQ-c H0-c3, B3/B4/C4 | 20 |
| Q11 | Schedule and form of the thesis proposal? | not written in the repo | — | learn at the meeting | proposal-v0.1 format | 16 |
| Q4 | Should LoRA be done; is there GPU access? (S9) | 65–98 T4 h estimate | eval-only / Kaggle 4-bit / university GPU | eval-only first; C3 in parallel if there is a GPU | C3/C4 ≈ 150 h | 12 |
| Q5 | Is OpenVLA 4-bit acceptable? | bf16 does not fit the T4 | 4-bit + K1/K3 / Mac bf16 / drop S-OV | 4-bit + calibration | system label | 12 |
| Q6 | A second reader for the Turkish set? | single author; `human_check.csv` empty | single / two + κ / survey | two readers, 20 rows | `datasets/04` v0.2, H0-b4 | 9 |
| Q10 | Repetition only in the pilot? | S-OV greedy | 3 per cell / in the pilot / none | in the pilot | rq-candidates v0.2 | 6 |
| Q7 | Role of Webots? (S8) | Lecture 21 decision note + 3 questions | LIBERO / small Webots set / both | LIBERO; Webots conditional | new cell group | 6 |
| Q8 | Relationship to Tatar et al.? | open loop, no null | extension / independent / joint | closed-loop extension | introduction chapter | 6 |
| Q9 | Is the Cohn line part of the thesis? | text LLMs, gold labels from a solver | part / side paper / out | side product | literature, extra paper | 4 |

---

## 4. Decision log template (ADR style; filled in before the meeting ends)

```
### ADR-001 — <decision title>            (question: Q_)
Date: ____   Status: accepted / rejected / deferred
Decision: <one sentence>
Rationale: <the advisor's reasoning, in their own words>
Consequence: <which file / cell / RQ changes — e.g. experiment-matrix.csv A2 n_init 50 → 20>
Owner: Mesut   Deadline: ____
Left open: <if any>
```

Post-meeting summary e-mail (same day): a one-line list of the decisions + "please correct me if I misunderstood anything" + a proposed date for the next meeting.

Version: v0.1 — Lecture 30 (automatic draft). Mesut edits before the meeting.
