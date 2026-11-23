# Robot Learning and VLA — MSc Thesis Journal

An open record of the **whole working process** of my Computer Engineering MSc thesis (ALKU — Alanya Alaaddin Keykubat University):
from learning the concepts to the literature survey, from setups to experiments, from the research question to thesis writing,
every step is kept in this repo in English (Turkish only for the research material itself: Turkish commands, segmentations, examples).

Field: **Vision-Language-Action (VLA)** models, robot learning, and connecting language and spatial commands to robot behavior
(language grounding). Since there are very few Turkish-language sources, the notes are written so that others can use them too.

**Site:** https://mesutd0nmez.github.io/vla-robot-learning-tr/ — lectures are reached from the left menu (GitHub Pages + Docsify; no build step, markdown is rendered in the browser).

## Stages

| Stage | Scope | Status |
|---|---|---|
| 0. Start — Accelerated introduction (Lectures 1–31) | Concepts → environment setup → first experiments → reasoning about the project (`plan/lecture-plan.md`) | ✅ Lectures 1–31 ready (complete) |
| 1. Literature and topic selection | Settling the thesis axis with the advisor, literature survey | ⬜ |
| 2. Experiment infrastructure | Setting up the benchmark, model, metrics and dataset | ⬜ |
| 3. Experiments | Main experiments, ablations, result analysis | ⬜ |
| 4. Thesis writing and defense | Thesis text, paper, presentation | ⬜ |

This table and the `plan/` folder are updated as the stages become clear.

## Structure

| Folder | Content |
|---|---|
| `plan/` | Lecture plan (`lecture-plan.md`) and `daily-log.md` (status, task and output of each working day) |
| `lectures/` | Lectures `lecture-NN.md` (images in `lectures/images/lecture-NN/`); each lecture is reading + a short task |
| `resources/` | Source list and introductory notes (links instead of copyrighted PDFs) |
| `literature/` | VLA and multilingual VLA literature: list of 15 peer-reviewed papers (`makaleler.md`) and a 2-page report (`rapor.md`) |
| `notes/` | Cumulative topic notes (OpenVLA, LIBERO, spatial reasoning, tokenization…) |
| `projects/` | Demo, experiment and research projects done by hand (one project per folder, `_template` template) |
| `datasets/` | Dataset cards and small data files (Turkish command sets, result tables; external data with a card only) |
| `code/` | Setup scripts, experiment code, Colab notebooks (each folder has a README with the source code) |
| `index.html`, `_sidebar.md`, `assets/` | Documentation site (Docsify); `tools/gen_sidebar.py` generates the menu |

| `thesis/` | Thesis drafts: candidate research questions (`rq-candidates.md`, Lecture 26) — later the proposal and thesis text |

To be added later: `experiments/` (experiment logs and results), `slides/`.

## How I work

- At most 1 hour a day. One topic, one small output per day. The lecture notes are researched and written by Claude;
  the outputs and answers are mine.
- Every lecture has the same skeleton: ~35 min of reading (with figures) → short task.

## Lectures

| # | Topic | Phase |
|---|---|---|
| [1](lectures/lecture-01.md) | The robot learning loop and the source map | 1 |
| [2](lectures/lecture-02.md) | Inside OpenVLA: image + language → action | 1 |
| [3](lectures/lecture-03.md) | Benchmark ≠ simulator: LIBERO, MuJoCo, LIBERO-Plus | 1 |
| [4](lectures/lecture-04.md) | Language grounding and shortcut learning | 1 |
| [5](lectures/lecture-05.md) | Deep paper read: Tatar et al. (2026) | 1 |
| [6](lectures/lecture-06.md) | Foundations of spatial reasoning | 1 |
| [7](lectures/lecture-07.md) | RoomSpace → LIBERO | 1 |
| [8](lectures/lecture-08.md) | Week 1 synthesis and VLA model families | 1 |
| [9](lectures/lecture-09.md) | Mac: conda + MuJoCo + first scene | 2 |
| [10](lectures/lecture-10.md) | Mac: LIBERO setup + first episode | 2 |
| [11](lectures/lecture-11.md) | LIBERO HDF5 demo data | 2 |
| [12](lectures/lecture-12.md) | VLANeXt on a Colab T4: first success rate | 2 |
| [13](lectures/lecture-13.md) | Code reading: the command → action → success chain | 2 |
| [14](lectures/lecture-14.md) | OpenVLA-7B on the Mac (MPS): image + command → 7 tokens → action | 2 |
| [15](lectures/lecture-15.md) | Mini reproduction of Tatar et al.: 5 frames × 6 conditions, L2 + cosine | 2 |
| [16](lectures/lecture-16.md) | Phase 2 synthesis + LIBERO-Plus: separate runtime, the command's code path, empty-command trial | 2 |
| [17](lectures/lecture-17.md) | LIBERO-Plus's seven perturbations in code: three carriers, camera math, 500 Pandas, R1/R2, Turkish variant | 3 |
| [18](lectures/lecture-18.md) | Same task, four commands: clean / empty / wrong / tr × 10 init states — experiment design, the Lecture 18 patch, Wilson + power calculation | 3 |
| [19](lectures/lecture-19.md) | Result analysis: is there shortcut learning, in which suite? — reading protocol, structural upper bound from BDDL, the real geometry of the 50 init states | 3 |
| [20](lectures/lecture-20.md) | Turkish spatial command set: 20 EN–TR pairs × 5 surfaces — relation name + suffix chain, three reference frames (LIBERO / image / robot), reference computed from 50 init states, counterfactual BDDL | 3 |
| [21](lectures/lecture-21.md) | Webots: the advisor's second simulator — its place in the MuJoCo/robosuite/LIBERO stack, macOS Apple Silicon setup + conda `vla` link, first scene with UR10e (Supervisor, camera, FK), webots.cloud / ROS 2, decision note for the thesis + 3 questions for the advisor | 3 |
| [22](lectures/lecture-22.md) | Tokenization analysis: the same 100 commands, two tokenizers — LLaMA-2 (OpenVLA) vs Qwen3 (VLANeXt): inflation 1.63× / 1.41×, fertility 3.1 / 2.4, the cost of Turkish letters, token shapes of the relation suffix, morpheme↔token boundary, *soldaki* = `sold`+`aki`, Qwen sentence-initial effect, the `lower()` trap, action range overlap 0, the Lecture 15 hypothesis closed | 3 |
| [23](lectures/lecture-23.md) | The Cohn line: seven papers at abstract level — StepGame correction (AAAI-24), RoomSpace (IJCAI-24), cardinal direction (COSIT-24 → Revisited 2025: 0.60 → 0.92, T4 still at the bottom), uncertainty in benchmark scores (temp 0 is not deterministic; prediction interval), DecompSR (k=1…100; EN/HI/SV; nonce), QSTRBench (9 calculi, English only); three shared findings and their LIBERO/Turkish counterparts, `datasets/04` relation families, adapting the PI to LIBERO runs | 3 |
| [24](lectures/lecture-24.md) | Solution approaches: MT translation layer / parser + symbolic / multilingual backbone / Turkish LoRA / hidden-state alignment — what changes, does it fit the Mac–T4 budget, in which Lecture 18 pattern does it work, "does it understand Turkish" or "does it reduce to English"; Dong et al. (EN-CoT does not help, step-wise shift +6…+26, ZH fine-tune 51.7 → 86.8) and Chen et al. (Qwen3-VL good only in Chinese, E-FT erases the capability, M-CT closes the gap but EN −2.7); OpenVLA LoRA memory map (72 / 27 GB; T4 only 4-bit, ≈7.7 GB estimate); rule-based Turkish parser real output (in-set 79/80, held-out 5/12, "wrong but valid" H04); decision tree | 3 |
| [25](lectures/lecture-25.md) | Phase 3 synthesis: the chain of eight lectures (benchmark / experiment / Turkish / solution sides + the Webots side branch); balance matrix — 21 measurements = 11 reproduced in this session with real output, 6 in Mesut's run, 4 hypotheses; `phase3_smoke.py` ran the 30 scripts/checks of Phase 3 as a dry run (28 OK / 0 FAIL / 2 skipped; Lecture 22 `tokens.csv` 201/201 exact); the four numbers of Phase 3 (no-command ceiling vs empty-command measurements, two tokenizers, Chen Table 1, Wilson n=10 vs 50); LangGap (Hou & Zhao 2026: when the command component changes in the same scene π0.5 drops 93.8 → 21.4, Spatial 97 → 5.9); the three sentences of the thesis; open questions S1–S10 (`plan/faz3-open-questions.md`) → RQ-a/b/c seeds | 3 |
| [26](lectures/lecture-26.md) | Start of Phase 4 — three candidate research questions: from seed to testable question; six-field template (null hypothesis + equivalence bound δ, variables, measure and numeric threshold, suite + block + n, systems, "what the thesis says if the answer is yes/no"), FINER/PICO and preregistration (Bertinetto et al.); exact computation in `rq_power.py`: with paired McNemar, n=50 (r=0.05) catches ≥23 points, n≈100 for Δ=15; TOST (Lakens) δ=15 → n≈25, δ=10 → n≈55; RQ-a (is there a difference; two nulls = A/B/C/D), RQ-b (where; family × surface × frame, B block, `tr_ascii` opposite-direction test), RQ-c (how; closure rate κ, weak/strong reading, English side effect); conditional ordering by the S1 pattern; `rq_lint.py` 3 × 7 fields COMPLETE; `thesis/rq-candidates.md` v0.1 | 4 |
| [27](lectures/lecture-27.md) | Experiment matrix — data, model, benchmark, metric, cost for each RQ: 19 cells (`thesis/experiment-matrix.csv`; calibration K1–K3, pilot, RQ-a/b/c); episode time model T_ep = reset + step·t_sim + call·t_call (max_steps 220/300 real code, chunk S-VN 8 / S-OV 1); `sim_step_bench.py` real LIBERO measurement (physics 12–23 ms, reset ~1–3 s, 50 init states on 4 tasks); S-OV in 4-bit on the T4 (OpenVLA Table 2; VLAQuantBench OFT W4 Goal 96 → 82) → K1/K3 calibration; budget pilot+core ≈ 258 T4 h (149–497), S-OV 63%, the string cache cuts C2 from 1,200 to 15 episodes; inverse power: task MDE ≥ 23, block pool n=120 → 13 → lean matrix ≈ 108 h | 4 |
| [28](lectures/lecture-28.md) | Baseline + ablation design: baselines (no-command ceiling/1/k, empty, wrong, clean, EN-B + LangGap 11%, published numbers); ablation rule (single factor → ablation, otherwise descriptive; Meyes et al., Lipton & Steinhardt); `ablation_check.py` real analysis: the letter step is 20/20 pure, the suffix step also deletes 68% of the Turkish letters → the surface axis is a 2×2 (letter × suffix), missing cell `tr_bare_ascii`; token 2×2 (real tokenizers, 160/160 against datasets/05): LLaMA-2 letter +2.7 / suffix +5.4 / interaction +4.3, Qwen3 +0.4 / +4.2 / +0.9; MT/parser oracles = the `clean` cell (re-reading; the parser is a tautology on A/C); code-switching `tr_cs` (Chen Table 3); K3 two-factor; system difference descriptive (StarVLA out of budget); `thesis/ablation-plan.md`, matrix v0.2 (AB1–AB3) ≈ 303 T4 h, lean 126 | 4 |
| [29](lectures/lecture-29.md) | One-page thesis proposal draft (`thesis/proposal-v0.1.md`): title TR + EN (no model name), argument chain problem → gap → aim → RQ-a/b/c → method → three contributions; mapping of the DARPA Heilmeier questions to the sections; 5 decision-relevant numbers in the method, claim unit = block (task MDE 23 / block 13), the "system-level" and "preregistration" sentences; schedule in T4 hours (Lecture 27 budget: WP1–WP4 lean ≈ 108 h, full ≈ 258; 16-week Gantt, midterm in week 5); 7 risks + plan B (risk matrix); `proposal_lint.py` 680/700 words, 8/8 sections, 12/12 numeric claims labeled (COMPLETE); ALKU and TÜBİTAK 2210-C form contents could not be read → ask the advisor | 4 |
| [30](lectures/lecture-30.md) | Question/presentation list for the advisor meeting: `collect_questions.py` gathered the repo lines on "advisor / S7–S10 / decision / claim unit / 4-bit" with file:line (184 lines / 37 files at the end of Lecture 27); 11 questions with the "what does the answer change?" filter (`code/day30/questions.csv`); impact × urgency: Q1 suite (S7), Q2 claim unit, Q3 weak/strong reading (S10), Q11 schedule, Q4 LoRA/GPU (S9) in the meeting, Q5 4-bit / Q6 second reader / Q10 repetition quick approval, Q7 Webots / Q8 Tatar et al. / Q9 Cohn by e-mail; RQ decision tree according to the advisor's answers; 30-min agenda; 7-slide skeleton; ADR-style decision log; `slides/advisor-meeting-01.md` | 4 |
| [31](lectures/lecture-31.md) | Final — the "let's do a project" simulation: Claude makes three short proposals (P1 Turkish vs English four conditions S-VN Spatial A; P2 `tr_ascii` opposite-direction/letter cost; P3 parser/MT closure κ), Mesut fills in the eight-field design template; `design_lint.py` structure + n↔MDE consistency, rubric; `phase0_inventory.py` real repo scan; competency radar (narrated vs proven by hand); carried-over items (S1, projects/01–05, runs.csv, prediction block, claim unit, K1/K2); **end of Stage 0 — the last lecture** | 4 |
- Each lecture is committed as `lectures/lecture-NN.md` together with `plan/daily-log.md`; more than one lecture can be worked in a day; this way my place is never lost.
- If I get stuck, the plan slips; it is not skipped.

## Hardware

MacBook Pro M2 (32 GB) — simulation (LIBERO/MuJoCo) and OpenVLA inference locally (MPS); CUDA-requiring
eval and training on Colab/Kaggle T4 or a rented GPU.

## Resources

See [`resources/README.md`](resources/README.md).

## License

Notes and code: MIT. Third-party papers and theses are not included in this repo; only links are given.
