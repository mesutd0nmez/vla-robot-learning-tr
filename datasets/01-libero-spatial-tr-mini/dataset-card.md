# LIBERO-Spatial TR mini — 5 tasks × 6 command conditions

**Type:** self-produced · **Version:** v0.1 · **Date:** Lecture 15 · **License:** MIT (command texts; the LIBERO commands are under LIBERO's MIT license) · **Related project/lecture:** Lecture 15, `code/day15/`, (project: `projects/03-tatar-mini/` — Mesut will open it)

## What it contains
30 command rows: 6 conditions for the first 5 tasks (0–4) of LIBERO-Spatial — `en` (LIBERO's own `task.language` text, the reference), `para` (an English paraphrase with the same meaning), `tr` (Turkish translation, LIBERO-style `-daki` participial construction, "kase"), `tr2` (a second Turkish surface form: "duran" construction, "kâse", "kaldır … yerleştir"), `distractor` (an English command describing the **other** black bowl in the same scene), `empty` (empty command). Each row marks the relation type (Lecture 6 calculi) and the suffixes that carry the spatial relation in Turkish. Also `token_counts.json`: the LLaMA-2 token count and the piece list of each row's OpenVLA prompt.

## Source and acquisition
- `en`: LIBERO `benchmark.get_benchmark_dict()["libero_spatial"]().get_task(i).language` (Lecture 10 output; LIBERO commit: main branch, Nov 2026 clone).
- `distractor`: determined by reading the `(:init …)` block of the LIBERO BDDL files — every Spatial scene has `akita_black_bowl_1` (target) and `akita_black_bowl_2` (distractor); the Spatial command describing the region of `bowl_2` was chosen (task 0→1, 1→6, 2→8, 3→9, 4→9).
- `para`, `tr`, `tr2`: written by Claude (consistent with the suffix scheme of Lecture 7 and the translation `tabak ile ramekinin arasındaki…` ("…between the plate and the ramekin…") of Lecture 14); **the second reading will be done by Mesut** (native-speaker verification pending — v0.2).
- Production script: `code/day15/make_commands.py`; token counts: `code/day15/count_tokens.mjs` (llama-tokenizer-js; to be verified with the HF tokenizer on the Mac).

## Schema (`data/commands.csv`)
| Field | Type | Description | Example |
|---|---|---|---|
| id | str | unique id `sp{task}_{condition}` | `sp0_tr` |
| suite | str | LIBERO suite | `libero_spatial` |
| task_id | int | task index (Lecture 10 order) | 0 |
| lang | str | `en` / `tr` / `none` | `tr` |
| condition | str | `en` / `para` / `tr` / `tr2` / `distractor` / `empty` | `tr` |
| instruction | str | command text (enters the OpenVLA prompt via `.lower()`) | `tabak ile ramekinin arasındaki siyah kaseyi al ve tabağın üstüne koy` |
| relation | str | spatial relation type | `arasında (2 referans)` |
| frame | str | viewpoint | `camera` |
| suffixes | str | suffixes carrying the relation in TR, separated by `;` | `-in;ara-sı-nda-ki;-ın;üst-ü-ne` |
| variant | str | surface form | `a` / `b` |
| note | str | note | `task.language (referans)` |

`data/token_counts.json`: `rows[]` (id, prompt_tokens_with_bos, instruction_tokens, instruction_pieces) and `phrases[]` (how spatial expressions and single words are split).

## Applied transformations
None (raw text). The OpenVLA prompt template is applied at call time: `In: What action should the robot take to {instruction.lower()}?\nOut:` — with an empty command `to ?` remains (deliberate; the same logic as LIBERO-Plus's empty-command condition).

## Known limitations
- 5 tasks, a single initial state (init 0): no statistical power (with n=5 the exact Wilcoxon p ≥ 0.0625). A pilot set.
- The translations have a single author (Claude); native-speaker check pending. Spelling differences such as `kase/kâse` and `üstüne/üzerine` were deliberately split between the two forms.
- `distractor` is the same command in tasks 3 and 4 (in both, `bowl_2` is on top of the cabinet).
- `frame=camera` has no effect since this set has no command containing a direction (right/left/front/behind).

## Usage
```python
import pandas as pd
df = pd.read_csv("data/commands.csv")
df[df.task_id == 0][["condition", "instruction"]]
```

## Changelog
- v0.1 — first version (Lecture 15): 30 rows, token counts.
