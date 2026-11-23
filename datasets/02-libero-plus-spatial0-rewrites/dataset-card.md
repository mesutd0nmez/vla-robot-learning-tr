# LIBERO-Plus Spatial task 0 — types of the 50 rewrites + 10 Turkish BDDL variants

**Type:** self-produced (labels over external data) · **Version:** v0.1 · **Date:** Lecture 17 · **License:** MIT (labels, Turkish BDDLs); the rewrite texts are under LIBERO-Plus's MIT license · **Related project/lecture:** Lecture 17, `code/day17/`, the Lecture 18 experiment

## What it contains
1. `data/rewrites_task0.csv` — the **50** `_language_K.bddl` rewrites that LIBERO-Plus produced for Spatial base task 0 (*pick up the black bowl between the plate and the ramekin and place it on the plate*); each row has whether it is in the benchmark (46 yes, 4 no: 16, 17, 29, 35), the LIBERO-Plus task index (0-based; for `VLANEXT_TASK_IDS`), the difficulty level, a **rule-based type label** (R1 distraction 26 / R2 common sense full 16 / R2 common sense light 6 / R0 plain paraphrase 2), word count, whether the words *bowl* / *plate* / *between* are preserved, and the number of capital letters (`scan_tokens.lower()` changes them).
2. `data/tr_variants/` — **10 Turkish** `_language_trK.bddl` files produced with the LIBERO-Plus template (Spatial tasks 0–4 × {tr1 = `datasets/01` `tr`, tr2 = `tr2`}); they differ from the base BDDL only in the `:language` line. Alongside: `task_map_lines.txt` (lines to add to the task map) and `task_classification_entries.json`.
3. `data/perturbation_table_spatial.md` — a summary table of the 7 perturbations extracted from the code (carrier, real parameter range, code location, BDDL/asset requirement).

## Source and acquisition
- Texts: [sylvestf/LIBERO-plus](https://github.com/sylvestf/LIBERO-plus) commit `4976dc3` (21 Jan 2026), the `:language` field of `libero/libero/bddl_files/libero_spatial/*_language_K.bddl`; benchmark membership, index and difficulty from `libero/libero/benchmark/task_classification.json`.
- Type label: `code/day17/classify_rewrites.py` — **regex rules** following the definitions in the paper's Appendix A.6 (R1 Distraction, R2 Common Sense, R3 Reasoning Chain); the JSON has no type field, so this label is not the authors' label but our reading (hypothesis). R3 was not seen in Spatial since it is not expected in single-step tasks.
- Turkish variants: `code/day17/make_tr_variant.py --commands code/day17/tr_commands.csv`; the commands are from `datasets/01-libero-spatial-tr-mini` v0.1 (native-speaker verification is still pending). Each file was passed through the real `bddl==1.0.1` parser; the text that will go to the model is exactly the same as what was written (10/10).
- Perturbation table: `code/day17/perturbation_table.py` (by reading task names, BDDL files, `mounted_panda.py`, `problems/libero_tabletop_manipulation.py`).

## Schema (`data/rewrites_task0.csv`)
| Field | Type | Description | Example |
|---|---|---|---|
| k | int | `_language_K` number (1–50) | 5 |
| in_benchmark | bool | present in the task map / JSON | True |
| task_index | int/empty | LIBERO-Plus Spatial task index (0-based) | 988 |
| difficulty | int/empty | L1–L5 (how many of the 4 reference models solved it; Appendix C.3) | 1 |
| type | str | `R1 dikkat dağıtma` / `R2 sağduyu (tam)` / `R2 sağduyu (hafif)` / `R0 düz parafraz` (rule-based; values kept in Turkish as in the CSV: distraction / common sense (full) / common sense (light) / plain paraphrase) | `R1 dikkat dağıtma` |
| words | int | word count (original 17) | 22 |
| keeps_bowl / keeps_plate / keeps_between | bool | does the word appear in the text | True |
| relation_word | str | the expression carrying the relation (`between`, `amid`, `separating`, `midway`) | `between` |
| conversational / descriptive | bool | whether the R1 / R2 rule fired | True / False |
| lowercase_changes | int | number of capital letters (characters that `.lower()` will change) | 1 |
| text | str | raw `:language` text in the BDDL (no punctuation) | `Would you mind helping me by …` |

## Applied transformations
None; the texts are as in the BDDL (with capital letters). During LIBERO-Plus eval `bddl.parsing.scan_tokens` applies `.lower()` to the whole file; what the model sees is `text.lower()`. The Turkish variants were written in lowercase from the start (Lecture 16 rule), so the transformation has no effect on them.

## Known limitations
- The type label is rule-based; borderline cases (e.g. 11, 18, 21, 31, 46: `black bowl→dark bowl`, `plate→dish`) were counted as "R2 commonsense (light)" — second reading from Mesut.
- Only task 0; the 450 rewrites of the other 9 Spatial tasks were not labeled (the script produces them with `--base N`).
- The Turkish variants are 5 tasks × 2; the same scene as init state 0. Once added to the benchmark they have no difficulty levels (not measured with the 4 reference models).
- Rewrites numbered 20 and 44 use the expression "a plate / another plate" that is not in the scene (there is a single plate) — a small scene error of the LLM rewrite.

## Usage
```python
import pandas as pd
df = pd.read_csv("data/rewrites_task0.csv")
df.groupby("type")[["words", "keeps_between"]].mean()
df[df.in_benchmark & (df.difficulty == 1)][["k", "task_index", "type", "text"]]   # 988/994/995 from Lecture 16 are here
```

## Changelog
- v0.1 — first version (Lecture 17): 50 labeled rows, 10 Turkish BDDLs, perturbation table.
