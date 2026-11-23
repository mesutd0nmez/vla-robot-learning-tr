# Lecture writing guide (for the automated task and for Claude)

This file defines how `lectures/lecture-NN.md` files are written. Every new lecture follows these rules.

**Project language: English.** Lectures, notes, plans and code comments are written in English; Turkish is kept only for the research material itself (Turkish commands, morpheme segmentations, examples, Turkish paper titles, data files).

## Order and topic
- **Stage 0 ended at Lecture 31 (23 Nov 2026).** Do not write lecture-32 or later unless Mesut explicitly asks.
- New lecture number = largest `lecture-NN` under `lectures/` + 1.
- Topic: the row with the same "Day" number in the `plan/lecture-plan.md` table. After 30: follow the stages in `README.md` (literature → experiment infrastructure → experiments → thesis writing) and the notes/questions in `plan/daily-log.md`, using the topic promised in the previous lecture's "Next lecture" section.
- Stay faithful to the previous lecture's "Next lecture" section; if Mesut left a note/question in `plan/daily-log.md`, answer it in one paragraph at the start of the lecture.

## Format (same skeleton as lecture-01…11)
1. `# Lecture N — Title` (short name before the colon; the menu uses it)
2. Meta line: `**Phase X / Day N**` · `**Reading time:** ~40 min` · prerequisite · source/code path. The `**Phase X / Day N**` token must appear in the first 8 lines (`tools/gen_sidebar.py` parses it).
3. `> **Today in one sentence:** …` (one sentence, the key idea)
4. 5–8 numbered sections separated by `---`; at least one concrete example in each; total reading ≤ 40 min, ≤ 60 min total on hands-on days
   - Do NOT write a `## 0. Honesty note` section (what was run / what is hypothesis); sections start at 1. Mesut tries everything himself. If something could not be run, say so in one sentence in the relevant section.
5. `## N. Bridges for the software developer` (Mesut is a 10-year C#/.NET backend developer; analogies come from there)
6. `## N. Today's task (15 min)` — a single concrete task; do NOT add an "fill in the output" section
7. On hands-on days, an `## If you get stuck` table (Symptom | Fix)
8. `## Next lecture (Lecture N+1)` — one paragraph
9. `## References` — with arXiv/GitHub links; a note on how the figures were produced

Do NOT write question sections: `Test yourself`, `Multiple choice` and `Answers` sections were removed from all lectures (22 Nov 2026); they are not added to new lectures either.

## Figures (mandatory, at least 3)
- Folder: `lectures/images/lecture-NN/`; in the lecture `![alt](images/lecture-NN/01-name.png)`
- Diagrams with Mermaid (`mmdc -i x.mmd -o x.png -b white -w 1500 -s 2`), keep the `.mmd` source next to the PNG. In Mermaid sequenceDiagram do not use special characters (→, ≠, arrows inside parentheses).
- Charts with matplotlib; if real paper values are used, state the source in the title; if data is representative/synthetic, say so underneath.
- Color language: yellow `#ffd166` = model/policy/key idea, blue `#cfe8ff` = observation/input, green `#d4f5d4` = action/output, red `#ffd6d6` = problem.
- Actually run everything that can be run (MuJoCo, robosuite, scripts) and use the real output as a figure; if it cannot be run, say so explicitly in the lecture.

## Code
- On hands-on days, scripts go under `code/dayNN/`; commands in the lecture must be downloadable with `curl -O https://raw.githubusercontent.com/mesutd0nmez/vla-robot-learning-tr/main/code/dayNN/x.py`.
- Each `code/dayNN/` folder gets a README.md: for each script `## name.py` + a ```python block (the site displays it).
- Hardware: MacBook Pro M2 32 GB (no CUDA, MPS available) + free Colab/Kaggle T4. What does not work on the Mac (flash-attn, bitsandbytes, VLANeXt eval) goes to Colab; Colab notebooks live in `code/dayNN/*.ipynb`.
- Version pins: `numpy<2`, `mujoco==2.3.7`, `robosuite==1.4.1`, Python 3.10.

## Language and tone
- English; technical terms as is (policy, action, success rate). Address Mesut as "you". No marketing language. Every claim is sourced; anything uncertain is marked "hypothesis". Turkish text is used only where it is the research object (commands, segmentations, examples), with a short English gloss where helpful.
- Thesis axis (candidate, not settled with the advisor): grounding of Turkish spatial commands in VLAs; OpenVLA (LLaMA-2 tokenizer) vs VLANeXt (Qwen3-VL) comparison; LIBERO/LIBERO-Plus; empty-command baseline. Every lecture ties back to this axis in at least one sentence.

## When finishing
1. Add a `## Lecture N — Topic` entry at the end of `plan/daily-log.md` (Status / Lecture / Task / Notes).
2. `python tools/gen_sidebar.py`
3. Add a new card to the lecture grid in `home.md`; add a row to the lecture table in the README.
4. Commit message: `Lecture N: <topic>`; push.
