# LIBERO-Spatial task 0 — 4-condition command experiment (clean / empty / wrong / tr): condition definition + result schema

**Type:** self-produced (experiment design; results will come from Mesut's Colab run) · **Version:** v0.2 · **Date:** Lectures 18–19 · **License:** CC BY 4.0 (command texts), results the same · **Related project/lecture:** Lecture 18 (design), Lecture 19 (analysis), `projects/05-spatial0-4kosul/` (to be opened)

## What it contains
`data/conditions_task0.json`: four command conditions for LIBERO-Spatial task 0 (*pick up the black bowl between the plate and the ramekin and place it on the plate*) — `clean` (null = the benchmark's `task.language`), `empty` (`""`), `wrong` (an English command describing the **other** bowl in the same scene: *next to the ramekin* = `akita_black_bowl_2`; the `distractor` row of `datasets/01`), `tr` (the `sp0_tr` row of `datasets/01`). `data/episodes_schema.csv`: the episode-level schema of run results (2 example rows, **representative**); the real 40 rows will go to `projects/05-…/results/episodes.csv` and be copied here in card v0.2. Experiment: 10 init states (`initial_states[0..9]`) × 4 conditions = 40 episodes, the VLANeXt-B LIBERO-Spatial checkpoint, T4.

Added with v0.2 (Lecture 19): `data/init_features_task0.csv` — the **real scene features of the 50 init states** of task 0 (read from the original LIBERO `.pruned_init` file without installing torch/LIBERO: `code/day19/init_state_geometry.py`); it joins `episodes.csv` on the `init_state` key and is the data for the question "are the discordant pairs explained by the scene?". `data/libero_suite_structure.csv` — the structural summary of the four LIBERO suites extracted from BDDL (number of scenes, scene→goal determination, no-command upper bound; `code/day19/suite_structure.py`).

## Source and acquisition
- Commands: `datasets/01-libero-spatial-tr-mini/data/commands.csv` → `code/day18/make_conditions.py --task 0` (automatic; 6 conditions with `--extra para,tr2`).
- Init states: the first 10 of the original LIBERO `init_files/libero_spatial/<task>.pruned_init` file (VLANeXt `build_episode_specs`: `init_state = initial_states[episode_idx]`); exactly the same across conditions → paired design.
- Model: `VLANeXt_libero_spatial.pt` (16.4 GB) from the `DravenALG/VLANeXt` HF repo (Lecture 12 card/notebook). Eval code VLANeXt `d911e6b` + Lecture 12 + Lecture 18 patches (`code/day18/vlanext_cond_patch.py`).

## Schema (`data/conditions_task0.json`)
| Field | Type | Description | Example |
|---|---|---|---|
| key | str | condition name; `condition=` in the log, `<name>__` prefix in the video name | `tr` |
| value | str/null | command sent to the model; `null` → `task.language` | `tabak ile ramekinin arasındaki siyah kaseyi al ve tabağın üstüne koy` |

## Schema (`data/episodes_schema.csv` = `episodes.csv` output of `analyze_4cond.py`)
| Field | Type | Description | Example |
|---|---|---|---|
| run_id | str | run id (same as `runs.csv`) | `vlanext_sp0_4cond_01` |
| suite | str | LIBERO suite | `libero_spatial` |
| task_id | int | task index (0-based) | 0 |
| condition | str | `clean` / `empty` / `wrong` / `tr` (+ `para`, `tr2`) | `empty` |
| init_state | int | `trial` = init state index (0–9); the pairing key | 3 |
| success | int | 0/1 — was the BDDL goal `(On akita_black_bowl_1 plate_1)` satisfied (independent of the condition, always the original goal) | 1 |
| rollout_steps | int | number of actions produced (max 220; failures are usually 220) | 96 |
| episode_time_s | float | `episode_time` in the log (HH:MM:SS → seconds) | 91.0 |

## Schema (`data/init_features_task0.csv`, 50 rows — real)
| Field | Type | Description | Example |
|---|---|---|---|
| init_state | int | `.pruned_init` row index = `trial` | 0 |
| bowl1_x, bowl1_y | float | table coordinates (m) of the target bowl (`akita_black_bowl_1`) | −0.063, 0.202 |
| bowl2_x, bowl2_y | float | the other bowl (`akita_black_bowl_2`) | −0.189, 0.320 |
| plate_x, plate_y | float | plate | 0.053, 0.205 |
| d_bowl1_bowl2 | float | bowl–bowl distance (m); 0.144–0.211 over the 50 scenes | 0.172 |
| d_bowl1_plate | float | target bowl–plate (m); 0.097–0.137 | 0.116 |
| d_bowl2_plate | float | other bowl–plate (m) | 0.268 |
| robot_q_dev | float | norm of the deviation of the 7 arm joints from the 50-scene mean (rad); median 0.050 | 0.045 |
| robot_q1..q7 | float | arm joint angles (rad) | −0.177, … |

The object order (`bowl_1, bowl_2, cookies, ramekin, plate`) is the BDDL `:objects` order and was verified by each object being within ±1.5 cm of its BDDL init region (50/50, five objects). Bowl orientations (quaternion) are constant across the 50 scenes.

## Schema (`data/libero_suite_structure.csv`, 4 rows — real)
`suite, n_tasks, n_scenes, n_object_sets, max_tasks_per_scene, scene_determines_goal, blind_upper_bound` — scene signature = multiset of (object type, init region) + fixtures; `blind_upper_bound` = the number of tasks a policy that never reads the command but recognizes the scene perfectly could solve (Spatial 9/10, Object 10/10, Goal 1/10, Long 9/10).

`runs.csv` (1 row per condition) follows the repo convention: `run_id, model, checkpoint, suite, task_id, lang, condition, n_episodes, success_rate`; `lang`: clean/wrong → `en`, empty → `none`, tr → `tr`.

## Applied transformations
1. `make_conditions.py`: CSV → JSON; `;`/parenthesis warning (the `scan_tokens` rule if one goes through the BDDL path, Lecture 16).
2. Eval: the command is written to `EpisodeSpec.task_description` via `VLANEXT_CONDITIONS`; case is preserved (override path; not the BDDL path — Lecture 17 §6 comparison).
3. `analyze_4cond.py`: `eval.log` regex → episodes/runs/table; Wilson 95%; paired McNemar exact (discordant pairs), unpaired Fisher exact.

## Known limitations
- **No results yet** (v0.1): the run was not made because there was no GPU/HF access in this environment; the schema rows are representative.
- n = 10 gives direction, not significance: the Wilson interval of 9/10 is 60–98; even 5 discordant pairs give p = 0.0625 in the paired test (Lecture 18 figures 03–04). Going up to 20–50 init states is a one-line change with `num_trials_per_task`.
- In the `wrong` condition `success` only means "did it do the original task" (LIBERO-CF's *biased* measure); a video label is needed for "is it faithful to the wrong command" (*faithful*) (no column; `grasped_bowl` will be added in v0.2).
- One task, one model, one Turkish surface form; `tr2`/`para` are optional.
- The positions in `init_features_task0.csv` are **initial** positions read from the MuJoCo state (region sampling + physical settling); they change during the episode. The robot joint deviation is at the ~1° level, i.e. the 50 scenes are almost repeats (Lecture 19 §3).

## Usage
```python
import json, pandas as pd
conds = json.load(open("data/conditions_task0.json"))
df = pd.read_csv("data/episodes_schema.csv")     # real data: projects/05-spatial0-4kosul/results/episodes.csv
df.pivot(index="init_state", columns="condition", values="success")
```

## Changelog
- v0.2 — `init_features_task0.csv` (50 real scene features) + `libero_suite_structure.csv` (Lecture 19); still no run result
- v0.1 — condition definition + schema (Lecture 18); no results
