# CLAUDE.md — working in this repo

This repo is Mesut Bahtiyar Dönmez's MSc thesis journal (ALKU, Computer Engineering): Robot Learning,
Vision-Language-Action (VLA) models, the LIBERO benchmark, and grounding of Turkish spatial commands.
Open source; site: https://mesutd0nmez.github.io/vla-robot-learning-tr/

**Project language: English.** All lectures, notes, plans, code comments and docs are written in English.
Turkish is kept only for the research material itself: Turkish commands, morpheme segmentations, examples,
Turkish paper titles and data files.

## Read first
1. `tools/LECTURE_GUIDE.md` — lecture format and all rules (mandatory)
2. `plan/lecture-plan.md` — Lectures 1–31 topic table and source map
3. `plan/daily-log.md` — progress; Mesut's notes/questions land here
4. Latest lecture: `lectures/lecture-NN.md` (largest NN)

## Structure
- `lectures/lecture-NN.md` — lectures (sequentially numbered, undated); images in `lectures/images/lecture-NN/`
- `code/dayNN/` — that lecture's scripts + README.md (each script as `## name.py` + a python block)
- `projects/NN-short-name/` — demo/experiment/research projects Mesut builds locally with Claude Code; start from `projects/_template/README.md` and add a row to the `projects/README.md` table. Lectures do not modify this folder; projects may cite lectures.
- `datasets/NN-short-name/` — dataset card (`dataset-card.md`, mandatory) + small data files (`data/`, <10 MB). Our own Turkish command sets and result tables live here; external data (LIBERO HDF5, checkpoints) get a card only (source, version, license, sha256, transformations).
- `literature/` — VLA and multilingual VLA literature: `makaleler.md` (15 peer-reviewed papers + 2 preprints, citations verified against the publication pages) and `rapor.md` (≤ 2 pages). Adding/removing papers is Mesut's decision.
- `resources/`, `notes/`, `plan/`; later `thesis/`, `slides/`
- Site: `index.html` + `assets/site.css` (Docsify, no build step), `home.md` (home page), `_sidebar.md` (generated)

## Automation
- Stage 0 (Lectures 1–31) is **complete**; the daily lecture task was closed on 23 Nov 2026. Lecture 31 is the last lecture —
  do not write new lectures (lecture-32+) unless Mesut explicitly asks. Run **`git pull`** before starting local work.
- **Stage 1 (from 28 Nov 2026): the project ladder** — `projects/ROADMAP.md`. 3-day projects of rising difficulty. Every morning at
  06:51 (TR) the cloud coach task reads `projects/<current>/progress.md` and the commits since yesterday, writes
  `projects/<current>/coach/day-N.md` (today's step-by-step instructions), and on the morning after day 3 grades the project
  (rubric in ROADMAP.md), closes it and opens the next one. The coach's instructions are in `tools/COACH_PROMPT.md`.
  **If today's `coach/day-N.md` does not exist when Mesut starts (the cloud task did not run), local Claude Code acts as the coach
  first — follow `tools/COACH_PROMPT.md`, write the note, then help him execute it.** Local Claude Code otherwise: follow today's
  `coach/day-N.md`, write to `progress.md`, commit `Project NN day N: ...`, push.
- After adding a new lecture/folder: `python tools/gen_sidebar.py`, add a card to the `home.md` grid, add a row to the `README.md` table.

## Doing projects locally (Claude Code)
- `git pull` → `cp -r projects/_template projects/NN-short-name` → fill in the README → code/results/images in the same folder → update the `projects/README.md` table → `python tools/gen_sidebar.py` → commit/push.
- Every experiment run adds one row to `results/runs.csv` (run_id, model, checkpoint, suite, task_id, lang, condition, n_episodes, success_rate). Open a card under `datasets/` for every dataset produced/used.
- Large files (checkpoints, HDF5, video) do not go into the repo (`.gitignore`); save results as small CSV/PNG.

## Working rules
- Language is English, technical terms as usual; address Mesut as "you". Turkish stays only for the research material (commands, segmentations, examples). Every claim is sourced (arXiv/GitHub link).
- A lecture is ≤ 1 hour; a figure is mandatory (Mermaid `.mmd` + PNG, matplotlib); anything runnable is actually run.
- Hardware: MacBook Pro M2 32 GB (no CUDA, MPS available) + Colab/Kaggle T4. Pins: Python 3.10, `numpy<2`, `mujoco==2.3.7`, `robosuite==1.4.1`.
- Copyrighted PDF/DOCX files do not go into the repo (`.gitignore`); links only.
- Commit: `Lecture NN: <topic>` or a short description; pushing directly to `main` is allowed (Mesut's permission).

## Thesis axis (candidate — not settled with the advisor)
Grounding of Turkish spatial commands in VLAs: OpenVLA (LLaMA-2 tokenizer) vs VLANeXt (Qwen3-VL) comparison,
LIBERO Spatial/Goal + LIBERO-Plus, success rate + empty-command baseline (correct/empty/wrong/paraphrase/Turkish conditions).

## Local site preview
`python3 -m http.server 8765` → http://localhost:8765
