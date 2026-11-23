# Datasets

Two kinds of records are kept:

- **Data I produce myself** (Turkish command sets, EN–TR pairs, experiment result tables): the file itself is in the repo (CSV/JSON/JSONL under `data/`, UTF-8, with a header row). The thesis appendices and, later, a Hugging Face dataset are produced from these.
- **External data** (LIBERO HDF5, checkpoints, RLDS): the file does not enter the repo; only a **dataset card** (`dataset-card.md`) is kept — source, version, license, download command, schema, sha256, applied transformations.

Folder name: `NN-short-name`. Each folder must have a `dataset-card.md`; `data/` is only for small files (< 10 MB).

| # | Dataset | Type | Status | Summary |
|---|---|---|---|---|
| 01 | [LIBERO-Spatial TR mini](01-libero-spatial-tr-mini/dataset-card.md) | self-produced | v0.1 (translation verification pending) | 5 Spatial tasks × 6 command conditions (en/para/tr/tr2/distractor/empty), 30 rows + LLaMA-2 token counts — Lecture 15 |
| 02 | [LIBERO-Plus Spatial 0 rewrites + TR BDDL](02-libero-plus-spatial0-rewrites/dataset-card.md) | self-produced (labels over external text) | v0.1 (type labels are rule-based) | The 50 `_language_K` rewrites of task 0: benchmark membership, index, difficulty, R1/R2 type, preserved words; 10 Turkish `_language_trK.bddl` + task map rows; code table of the 7 perturbations — Lecture 17 |
| 03 | [LIBERO-Spatial 0 — 4-condition command experiment](03-libero-spatial0-4cond/dataset-card.md) | self-produced (design) | v0.2 (awaiting results) | Condition JSON clean/empty/wrong/tr for task 0 + 10 init × 4 condition episode schema; v0.2: real scene features of the 50 init states (`init_features_task0.csv`) + BDDL structure of the four suites (`libero_suite_structure.csv`) — Lectures 18–19 |
| 04 | [TR spatial command set](04-tr-spatial-commands/dataset-card.md) | self-produced | v0.1 (native-speaker reading pending) | 20 EN–TR pairs × 5 surfaces (en / tr_a / tr_b / tr_ascii / tr_bare) = 100 rows; 4 blocks (Spatial trained / Spatial new recipe / Goal trained / untrained placement), 3 tiers; suffix distinction, the bowl meant in three frames (from the 50 init states), 5 counterfactual BDDL — Lecture 20 |
| 05 | [Tokenization of the TR command set](05-tokenization-tr-commands/dataset-card.md) | self-produced (on top of datasets/04) | v0.1 (Qwen3-VL tokenizer identity to be verified from HF) | 100 commands × 2 tokenizers (LLaMA-2 Meta `tokenizer.model` / Qwen3 `tokenizer.json`): tokens/fertility/inflation, relation-word pieces + morpheme↔token boundary overlap, suffix shapes, OpenVLA action-range check, llama-tokenizer-js cross-check — Lecture 22 |
