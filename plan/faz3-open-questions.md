# Phase 3 open questions (Lecture 25)

The list that came out of the Phase 3 synthesis (Lectures 17–24) and feeds Phase 4 (Lecture 26, RQ formulation). The `Status` column is Mesut's; when a question closes, the row is not deleted — write the date + a one-line result. Lectures 26–27 read from here.

**Priority:** S1 is on the critical path — the decision tree of Lecture 24 (which solution becomes the thesis) depends on the pattern in Lecture 18. While S1 is open, RQs are *written*, not *ranked*.

## Waiting for Mesut's run (script ready)

| # | Question | Tool / input | Acceptance criterion | Feeds RQ | Status |
|---|---|---|---|---|---|
| S1 | Lecture 18 table: `clean / empty / wrong / tr` × 10 init states (VLANeXt-B, Spatial task 0) — which pattern? | `code/day18/colab_4cond_t4.ipynb` → `analyze_4cond.py` → `code/day19/read_patterns.py episodes.csv --features …` | Pattern letter (A/B/C/D) at the top of the `projects/05-spatial0-4kosul/` README; `results/runs.csv` rows (including `seed`, `date`) | RQ-a | waiting |
| S2 | MT round-trip: with Opus-MT tr-en, the 80 Turkish rows of `datasets/04` → exact match, count of "wrong but valid", median latency | `code/day24/mt_layer_mac.py --model Helsinki-NLP/opus-mt-tr-en` | Three numbers + `wrong_en` examples (Lecture 24 note) | RQ-c | waiting |
| S3 | Qwen3-VL tokenizer identity: is the HF `Qwen/Qwen3-VL-2B-Instruct` tokenizer byte-identical to the npm `@lenml/tokenizer-qwen3` `tokenizer.json`; does the OpenVLA HF tokenizer match SentencePiece exactly? | `code/day22/tokenize_set.py` twice (`--llama tokenizer.model` / `--llama openvla/openvla-7b`; `--qwen Qwen/Qwen3-VL-2B-Instruct`) → `diff` of the first 7 columns of `tokens.csv` | `diff` empty → both hypotheses close; otherwise `datasets/05` card v0.2 | RQ-a | waiting |
| S4 | Native-speaker reading: which bowl is "soldaki / sağdaki" ("the one on the left / right") in the task 1 render and "öndeki" ("the one in front") in task 8; `tr_a` / `tr_b` naturalness 1–5 + corrections | `datasets/04-tr-spatial-commands/data/human_check.csv` | 20 rows filled; one-sentence comparison of the `hangi_kase_mesut` ↔ `ref_*` columns | RQ-b | waiting |
| S5 | Does the Webots R2025a launcher work on the M2; first real render of `ders21_masa.wbt` | `code/day21/check_webots_env.py`, `ders21_kol.py --out out` | `view_01_bowl.jpg`, `joints.csv` `max_err_rad` → `projects/06-webots-ilk-sahne/` | S8 | waiting |
| S6 | Parser held-out: +5 sentences in your own Turkish; error type (no parse / wrong but valid) | `code/day24/heldout_tr.csv` + `tr_parse.py` | How many passed + error types in one line | RQ-c | waiting |

## For the advisor (decision)

| # | Question | Basis | Options | Feeds RQ | Status |
|---|---|---|---|---|---|
| S7 | Main suite: Spatial (CDC relations; no-command ceiling 9/10 → shortcut risk) or Goal (topology; ceiling 1/10; cleanest Turkish on the token side) — or both? | Lecture 19 §2–3, Lecture 23 §5 | Spatial 50 / Goal 50 / both as 10-episode pilots, then decide | RQ-b | not asked |
| S8 | Is Webots needed as a second environment (room scale, robot-relative direction, RoomSpace-style set); is there a ROS 2 / real-robot plan? | Lecture 21 §6 decision note + 3 questions | Stay in LIBERO / small Webots set / both | RQ-b (indirect) | not asked |
| S9 | Turkish LoRA / co-training budget: is 2–3 weeks on Kaggle at 30 h/week (estimate 65–98 T4 hours, 4-bit) acceptable, or should the thesis stay eval-only? | Lecture 24 §5, `code/day24/qlora_budget.py` | eval-only / 4-bit co-training / university GPU | RQ-c | not asked |
| S10 | Contribution criterion: weak reading (correct behavior from a Turkish command; MT/parser suffices) or strong reading (above `empty` on the B block + counterfactual BDDL; needs backbone/LoRA)? | Lecture 24 §1, §7 | weak is enough / strong required / weak floor + strong target | RQ-c | not asked |

## RQ seeds going into Lecture 26 (name level; question form in Lecture 26)

| Seed | Name | Inputs | Linked questions |
|---|---|---|---|
| RQ-a | **Is there a difference?** `tr − clean` and `tr − empty` in the same scene; two systems (OpenVLA-Spatial LoRA, VLANeXt-B); Wilson + PI | Lectures 18–19, 22 | S1, S3 |
| RQ-b | **Where does it break?** Relation family (CDC / topology / distance) × surface (`tr_a/tr_b/tr_ascii/tr_bare`) × frame; suite choice | Lectures 19–20, 23 | S4, S7, (S8) |
| RQ-c | **How does it close?** Parser/MT floor vs EN+TR co-training; criterion B block | Lecture 24 | S2, S6, S9, S10 |

## What remains hypothesis (to be closed in Phase 4 with a "pilot study")
- That the training images were also rotated with `[::-1, ::-1]` (Lecture 20 §3).
- VLANeXt's run-to-run variability (flow-matching `torch.randn`, seed; Lecture 23 §4) — ≥3 repetitions of S1 would measure it.
- That QLoRA takes ≈ 7.7 GB on the T4, and the duration (Lecture 24 §5).
- Whether Turkish falls on the Chinese or the Arabic side in Qwen3-VL (Chen Table 1 pattern; Lecture 24 §4) — the `tr` row of S1 is the first evidence.
- The space after `<|vision_end|>` in the VLANeXt template (Lecture 22 §4; `code/day13/show_prompt.py`).

## Lecture 25 task (Mesut fills in)
```
phase3_smoke.py: __ OK / __ FAIL / __ SKIPPED   (date: ______)
tokens.csv == datasets/05?: ______
Questions marked Priority 1: S__, S__, S__
```
