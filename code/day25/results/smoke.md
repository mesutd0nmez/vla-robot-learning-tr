# Phase 3 dry run — 28 OK / 0 FAILED / 2 SKIPPED (total 30)

| Lecture | Script | Dependency | Result | Time | Last line |
|---|---|---|---|---|---|
| 17 | `perturbation_table.py` | LIBERO-plus clone | OK | 0.1 | / Objects Layout / 385 / BDDL file: `_add_K` (new object + `add_object_region_K`) and `_levelL_sampleM` (regio |
| 17 | `camera_geometry.py` | LIBERO-plus clone | OK | 1.2 | written: /tmp/faz3smoke_jr9oicap/cam |
| 17 | `robot_init_spread.py` | LIBERO-plus clone | OK | 0.6 | written: /tmp/faz3smoke_jr9oicap/rob |
| 17 | `layout_shift.py` | LIBERO-plus clone | OK | 0.9 | written: /tmp/faz3smoke_jr9oicap/lay |
| 17 | `classify_rewrites.py` | LIBERO-plus clone | OK | 0.8 | written: /tmp/faz3smoke_jr9oicap/cls |
| 17 | `make_tr_variant.py` | LIBERO-plus clone + bddl | OK | 0.0 | # Then: as VLANEXT_TASK_IDS use the 0-based indices of the new tasks = old list length, +1, …  (Lecture 16: id |
| 18 | `make_conditions.py` | none | OK | 0.0 | Colab: os.environ["VLANEXT_CONDITIONS"] = open("cond.json").read() |
| 18 | `spec_order_demo.py` | none | OK | 0.0 | In the log: trial=<init>, condition=<name>; video name '<name>__<command>' — no need to infer the condition fr |
| 18 | `analyze_4cond.py` | none | OK | 0.7 | selftest OK (numbers are ILLUSTRATIVE; run with a real log) |
| 18 | `power_day18.py` | none | OK | 15.3 | McNemar smallest p, n=10, b discordant: b=3: 0.2500, b=4: 0.1250, b=5: 0.0625, b=6: 0.0312, b=10: 0.0020 |
| 19 | `read_patterns.py` | none | OK | 0.0 | selftest OK |
| 19 | `read_patterns.py --demo` | none | OK | 0.0 | demo OK: the four representative patterns were classified correctly |
| 19 | `suite_structure.py` | LIBERO clone | OK | 0.0 |   libero_10: LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket -> (And (In alph |
| 19 | `init_state_geometry.py` | LIBERO clone | OK | 0.1 |   bowl-bowl distance 17.6 cm (min 14.4, max 21.1); target bowl-plate 11.7 cm (min 9.7, max 13.7) |
| 20 | `tr_morph.py` | none | OK | 0.0 | tr_morph selftest: 42/42 correct |
| 20 | `make_tr_set.py` | none | OK | 0.0 |       kase-yi tabağ-ın sol-u-na koy |
| 20 | `validate_set.py` | none | OK | 0.0 | RESULT: CLEAN — no errors |
| 20 | `scene_frames.py` | LIBERO clone | OK | 0.1 |    higher      none    → bowl_1   min gap   22.2 cm |
| 20 | `make_cf_bddl.py` | LIBERO clone + bddl | OK | 0.0 | 5 counterfactual BDDL + index.csv → /tmp/faz3smoke_jr9oicap/cf |
| 21 | `check_webots_env.py` | Webots | SKIPPED |  |  |
| 21 | `ur10e_fk.py` | none | OK | 0.1 | 03_plate      0.220    0.021    1.000  True       0.921 |
| 22 | `tokenize_set.py` | sentencepiece+tokenizer | OK | 0.6 | - tokens falling in the action range over 100 commands: **0**; sample contents of the range: 忠, 错, 단, 면, ķ, 桥 |
| 22 | `tokens.csv == datasets/05?` | sentencepiece+tokenizer | OK | 0.0 | 201 vs 201 rows, rows differing in the first 7 columns: 0 -> IDENTICAL |
| 22 | `probe_edge_cases.py` | sentencepiece+tokenizer | OK | 0.5 |   [qwen3 ]  3  et / ▁al / . |
| 23 | `prediction_interval.py` | none | OK | 0.6 | selftest OK: PI formula (t·s·sqrt(1/n+1/n')), zero width, effect of n', single run = inf, 0.01 threshold |
| 23 | `prediction_interval.py --scores` | none | OK | 0.8 | n=3  x̄=0.867  PI95 = 0.867 ± 0.203  [0.664, 1.069]  width=0.406  -> add repeats (paper threshold 0.01) |
| 24 | `tr_parse.py --selftest` | none | OK | 0.0 | selftest: 4/4 |
| 24 | `tr_parse.py` | none | OK | 0.0 |   held-out: 5/12 |
| 24 | `qlora_budget.py` | none | OK | 0.0 | Warning: this is a calculation; the real number is the value you will measure on the T4 with `finetune.py --us |
| 24 | `mt_layer_mac.py` | HF+GPU | SKIPPED |  |  |
