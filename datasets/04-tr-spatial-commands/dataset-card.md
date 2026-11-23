# TR spatial command set — 20 EN–TR pairs × 5 surfaces

**Type:** self-produced · **Version:** v0.2 (native-speaker reading pending; Lecture 24 parser outputs added) · **Date:** Lecture 20 · **License:** MIT (command texts; the English training commands and BDDL templates are under LIBERO's MIT license) · **Related project/lecture:** Lecture 20, `code/day20/`; continued in Lecture 21 (tokenization), Lectures 26–27 (experiment matrix)

## What it contains
20 EN–TR command pairs, in four blocks: **A** (6) the trained *pick* commands of LIBERO-Spatial (`…-ndeki siyah kaseyi al`, "pick up the black bowl that is …"), **B** (6) recipes in the same Spatial scenes that match no training sentence (left/right/front; near–far from the robot/plate), **C** (6) the trained *placement* commands of LIBERO-Goal (`…-nın üstüne koy`, "put it on top of …"), **D** (2) untrained placement (to the right/left; no predicate). Each pair comes with 5 surfaces → `commands.csv` has 100 rows: `en`, `tr_a` (the main form produced with the suffix generator), `tr_b` (a second, hand-written form: *duran / üzerine / alıp*, different word order, *kâse*), `tr_ascii` (without Turkish characters), `tr_bare` (a bag of words with the suffixes removed). In block B, **which bowl** the command means is computed in three frames (LIBERO naming / the image the model sees / robot) from the real coordinates of the 50 init states.

## Files
| File | Content |
|---|---|
| `data/pairs.csv` | 20 rows, one row per pair (design variables + all surfaces + reference labels) |
| `data/commands.csv` | 100 rows, one row per surface (compatible with the template schema; the file the experiment runner reads) |
| `data/referents.csv` | 10 Spatial scenes × 9 recipes (left/front × 3 frames, near the plate, near the robot, higher up): the bowl meant, the smallest margin over 50 scenes, whether it is above the threshold |
| `data/bddl_cf/*.bddl` + `index.csv` | 5 counterfactual BDDL: goal `(On akita_black_bowl_2 plate_1)`; scene and regions the same as the original |
| `data/human_check.csv` | Native-speaker reading template for Mesut to fill in (naturalness 1–5, correction, "which would you pick" for B07–B09) |
| `data/parse_roundtrip.csv` | Lecture 24: 80 Turkish surfaces passed through the rule-based parser (`code/day24/tr_parse.py`) and returned to an English template: frame (JSON), produced EN, gold EN, result (`ok` / `wrong_en` / `parse_fail`) — 79/80 (the parser was written looking at this set, an upper bound) |
| `data/heldout_tr.csv` | Lecture 24: 12 new Turkish sentences written after the parser was frozen + gold EN + note; the parser gets 5/12 (out-of-set behavior) |

## Source and acquisition
- `en` (A, C): LIBERO task names (`libero_suite_task_map.py`; the text that goes to the model is derived from the file name — Lecture 13). `en` (B, D): written by Claude; `make_tr_set.py --libero` verified that none of LIBERO's 130 task names contains them (`en_seen_libero = hayır`, i.e. "no").
- `tr_a`: `code/day20/make_tr_set.py` — dictionary + template + `tr_morph.py` (vowel harmony, buffer letters, consonant softening, pronominal n); segmentation in `tr_a_seg`. A01–A04 are identical to the `sp0/1/2/4_tr` rows of `datasets/01`.
- `tr_b`: written by hand by Claude. **The native-speaker reading will be done by Mesut** (`human_check.csv`, v0.2).
- `ref_*`: `code/day20/scene_frames.py` — LIBERO `.pruned_init` (50 init states) + agentview camera pose (`libero_tabletop_manipulation.py`); LIBERO commit: main branch, Nov 2026 `--depth 1` clone.
- `bddl_cf/`: `code/day20/make_cf_bddl.py`; each file was read back with `bddl==1.0.1` `scan_tokens` (language exact, goal bowl_2, `:init`/`:regions` unchanged).

## Schema (`data/pairs.csv`)
| Field | Type | Description | Example |
|---|---|---|---|
| pair_id | str | block letter + sequence | `B07` |
| block | str | `A` / `B` / `C` / `D` | `B` |
| tier | int | 1 = existing checkpoint + existing predicate; 2 = existing checkpoint + counterfactual BDDL; 3 = new predicate + fine-tune | 2 |
| suite, scene_task_id | str, int | the scene the command will run in (official task order) | `libero_spatial`, 1 |
| relation | str | relation type (Lecture 6 calculi) | `yön (sol)` |
| rel_noun | str | bare form of the relation noun (`—`: no relation noun) | `sol` |
| case | str | case of the relation word | `LOC+ki (iyeliksiz)` |
| frame_dependent | str | `evet` / `hayır` (yes / no) | `evet` |
| en, tr_a, tr_b, tr_ascii, tr_bare | str | the five surfaces | `soldaki siyah kaseyi al ve tabağın üstüne koy` |
| tr_a_seg | str | suffix segmentation of `tr_a` | `sol-da-ki siyah kase-yi al ve tabağ-ın üst-ü-ne koy` |
| rel_span, rel_seg | str | the word carrying the relation and its segmentation (Lecture 21 tokenizes it on its own) | `soldaki`, `sol-da-ki` |
| ref_span | str | reference noun (if any) | `ramekinin` |
| ref_libero, ref_image, ref_robot | str | the bowl meant in three frames (A: the target in training; C/D: `n/a`) | `bowl_2`, `bowl_1`, `bowl_1` |
| ref_min_margin | str | smallest margin over the 50 init states (cm / px) | `36.5 cm \| 111.6 px \| 36.5 cm` |
| en_seen_suite, en_seen_libero | str | is the EN command a task name of the same suite / of any LIBERO suite | `hayır` |
| goal_predicate | str | success predicate (BDDL `:goal`) | `(On akita_black_bowl_1 plate_1)` |
| note | str | design note | |

`data/commands.csv`: `id` (`B07_tr_a`), `pair_id`, `suite`, `task_id`, `lang`, `variant`, `instruction`, `relation`, `frame`, `suffixes`, `tier`.

## Frame definitions (`referents.csv`, `ref_*`)
| Frame | left | front | Basis |
|---|---|---|---|
| `libero` | small y (−y) | large x (+x) | LIBERO-90/10 BDDL region names: 5/5 left/right pairs, 2/2 front/back pairs (`scene_frames.py --conventions`) |
| `image` | small u | large v (lower in the image) | 256×256 frame with `obs[::-1, ::-1]` applied; projection from the camera quaternion, matches Mesut's project-2 render |
| `robot` | large y (+y) | near the robot base | base x = −0.66, facing +x (`mounted_panda.py`) |

The "not ambiguous" threshold: the same sign in all 50 scenes **and** a difference of at least 5 cm / 20 pixels.

## Applied transformations
None (raw text). All rows are BDDL-safe (lowercase, no `;` or parentheses, NFC) — checked by `validate_set.py`.

## Known limitations
- Single-author translation; no native-speaker reading (v0.1). The naturalness of `tr_b` and the human interpretation of the recipes in block B will come with `human_check.csv`.
- The 20 pairs are a **design skeleton**, not a statistical sample: each cell has 1–2 commands. For the thesis experiment they must be multiplied per block (the template + dictionary make this cheap).
- Block B uses only three scenes with two bowls (tasks 0, 1, 8); there is a single scene (task 1) for left/right.
- The counterfactual BDDLs were not loaded in a real LIBERO environment (no MuJoCo/LIBERO in this session); they only passed the parser.
- The `image` frame rests on the assumption that the model's training images were produced with the same 180° rotation (Lecture 10/12 code reading).
- Block D has no predicate; these rows exist today only for the language side (tokenization).
- `tr_bare` production leaves the compounds as they are (*şarap şişesi*, "wine bottle"; *üst çekmece*, "top drawer").

## Usage
```python
import pandas as pd
p = pd.read_csv("data/pairs.csv")
p[p.block == "B"][["pair_id", "tr_a", "ref_libero", "ref_image", "ref_robot"]]
c = pd.read_csv("data/commands.csv")
c[(c.pair_id == "C13")][["variant", "instruction"]]
```

## Changelog
- v0.1 — first version (Lecture 20): 20 pairs, 100 surfaces, 90 reference rows, 5 counterfactual BDDL.
- v0.2 — Lecture 24: `parse_roundtrip.csv` (80 rows) and `heldout_tr.csv` (12 rows) added; the command texts did not change.
