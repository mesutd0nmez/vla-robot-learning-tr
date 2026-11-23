# Candidate research questions (RQ) — draft v0.1 (Lecture 26)

Three candidate questions; each is written with the same six fields. This file is a **preregistration** draft: before the numbers arrive, the question, null hypothesis, measure and thresholds are written here and committed; after the results arrive the threshold is not moved, and if it is moved it is written with a date + justification. The ordering is **conditional**: once the Lecture 18 table (S1) enters the repo, the primary RQ is chosen according to the pattern in `plan/faz3-open-questions.md` (Lecture 26 §6 tree).

Abbreviations: `clean` = the original English command, `empty` = empty command, `wrong` = an English command describing another goal in the same scene, `tr_*` = the `datasets/04` surfaces (`tr_a` generated, `tr_b` by hand, `tr_ascii` without Turkish letters, `tr_bare` without suffixes), block A/B/C/D = `datasets/04/data/pairs.csv` (A: Spatial trained 6, B: an untrained recipe in the same scene 6, C: Goal trained 6, D: untrained placement 2). Systems: **S-OV** = `openvla/openvla-7b-finetuned-libero-spatial` (LLaMA-2 tokenizer, 256-bin action tokens), **S-VN** = VLANeXt-B (Qwen3-VL-2B backbone, flow-matching action head). The difference between the two systems is a **system difference**, not a "backbone difference" (backbone + action head + training data change together; Chen et al. 2606.15714).

Common statistics protocol (Lectures 18, 23): every cell is paired (same init states, same process); `n_runs × n_episodes` is written; within-run Wilson 95%, run-to-run prediction interval (PI, ≥3 repetitions); McNemar exact for the paired difference; TOST (90% CI, δ written in advance) for a "no difference" claim. Thresholds (computed with `code/day26/rq_power.py`): with n=50 paired init states and inverse noise r=0.05, the smallest difference caught with 80% power is ≈ 23 points; n≈100 for Δ=15; for equivalence n≈25 with δ=15 and n≈55 with δ=10 (q=0.20 discordant pairs).

---

## RQ-a — Is there a difference?

**Question:** In LIBERO, in the same initial scene, is there a closed-loop success-rate difference between the Turkish command (`tr_a`) and the original English command (`clean`); if so, where does this difference stand relative to the empty-command baseline (`empty`) — and is this pattern the same in S-OV and S-VN?

**Null hypothesis:** H0-a1: on the same init states P(success | tr_a) = P(success | clean) (paired difference 0; equivalence bound δ = 15 points). H0-a2: P(success | tr_a) = P(success | empty) (δ = 10 points; the "tr ≈ empty" threshold of Lecture 19). The two nulls are read together: if a1 is rejected and a2 cannot be rejected, "Turkish is not read" (pattern B); if both are rejected, "read but incomplete" (D); if a1 cannot be rejected, the `empty` row splits the result in two (A: language is ignored; D: Turkish works).

**Variables:** Independent: command condition {clean, empty, wrong, tr_a} (within scene, paired); system {S-OV, S-VN} (across scenes, same init state files). Dependent: episode success (0/1, LIBERO predicate). Covariates: number of tokens in the command and number of pieces of the relation suffix (`datasets/05/tokens.csv`), the no-command upper bound of the task (Lecture 19: Spatial 9/10, Goal 1/10). Held fixed: init state set, `num_steps_wait`, camera/resolution, checkpoint version, seed list.

**Criterion and threshold:** Primary: paired difference Δ(tr_a − clean) and Δ(tr_a − empty), in points; McNemar exact p; Wilson 95% (within run) + PI (≥3 repetitions). "There is a difference" decision: McNemar p < 0.05 **and** |Δ| ≥ 15. "No difference" decision: TOST 90% CI ⊂ (−δ, +δ). If neither, "inconclusive" (n is increased, the threshold is not moved). Secondary: the `grasped_bowl` label (which object was approached) — separates whether a failure is "wrong target" or "no movement at all".

**Suite / block / sample:** Spatial block A (6 tasks) + Goal block C (6 tasks). Pilot: Spatial task 0 × 10 init states × 4 conditions × ≥3 repetitions (= the Lecture 18 run + repetitions). Thesis: 50 init states per task (all 50 in `.pruned_init` for S-VN); if `empty` is near the ceiling in Spatial (≥ 80), the weight shifts to Goal (Lecture 19 rule, S7). Total episodes (thesis): 12 tasks × 50 init × 4 conditions × 2 systems = 4,800 (+ repetitions); the T4 budget is in Lecture 27.

**Systems:** S-OV (OpenVLA-7B Spatial LoRA; a separate checkpoint `openvla-7b-finetuned-libero-goal` for Goal), S-VN (VLANeXt-B; Spatial and Goal checkpoints). Each system is evaluated in the suite it was trained on; the comparison across systems is **descriptive** (not causal — system difference).

**If yes / if no, what the thesis says:** *Yes (a1 rejected, |Δ| ≥ 15):* "A Turkish command, in the same scene, lowers success by X / Y points in two open VLAs; the drop [approaches / does not approach] the `empty` baseline" → the main measurement contribution of the thesis; move on to RQ-b. *No (equivalent):* if `empty` ≈ `clean` (A) "language cannot be measured in these tasks; move to Goal / block B"; if `empty` ≪ `clean` (D) "Turkish is already read in this system" → the thesis reports where the Qwen3-VL backbone reads Turkish (RQ-b) and its difference from LLaMA-2. *Inconclusive:* n and repetitions are increased; the thesis is written in "direction" mode. No result makes the thesis void; every branch selects the next question.

---

## RQ-b — Where does it break?

**Question:** What feature of the command does the drop in Turkish (RQ-a's Δ) depend on: the relation family (CDC/direction, RCC/topology, distance, region), the surface (`tr_a / tr_b / tr_ascii / tr_bare`), the reference frame (B07–B09: LIBERO / image / robot) — and does an untrained Turkish recipe (block B) rise **above the empty command** in the model?

**Null hypothesis:** H0-b1: Δ(tr − clean) does not vary by relation family (no family × condition interaction). H0-b2: Δ does not vary by surface; specifically Δ(tr_ascii) = Δ(tr_a) in both systems (Lecture 22: the Turkish letter cost is 4.9 tokens in LLaMA-2 and 0.9 in Qwen3 → H0-b2 being rejected in S-OV and not rejected in S-VN would be the first evidence for the tokenization explanation). H0-b3 (strong reading): in block B P(success | tr_B) = P(success | empty) — the Turkish new recipe is no different from the empty command. H0-b4 (frame): in B07/B08 (*soldaki/sağdaki*, "the one on the left / right") the approached bowl is distributed as if chosen at random from the three frames.

**Variables:** Independent: relation family (4 levels; `pairs.csv` `relation`), surface (4 Turkish + en), block (A/B/C), frame (`ref_libero/ref_image/ref_robot` in B07–B09). Dependent: episode success; in block B also the **approached object** (`grasped_bowl` ∈ {bowl_1, bowl_2, none}) and the success measured with the counterfactual BDDL (`bddl_cf/`, target bowl_2). Covariates: number of tokens, number of pieces of the relation suffix, `ref_min_margin` (cm/px). Fixed: RQ-a's init states and protocol.

**Criterion and threshold:** Success + Wilson per cell; paired difference tables for family and surface (Δ and 95% CI); an interaction claim is written only if the cell CIs separate (small n: 6 pairs × 50 init, 1–3 tasks per family → **descriptive**, "direction" mode; Lecture 23). For H0-b3: tr_B − empty ≥ 15 and McNemar p < 0.05 → "evidence of strong reading"; within TOST δ=10 → "none". For H0-b4: which of the three frames the approached bowl agrees with, a count table + Wilson instead of chi-square (small n).

**Suite / block / sample:** Spatial block A + **B** (the EN column of B is also run: EN-B − clean separates the "new recipe" cost, TR-B − EN-B the "Turkish" cost; LangGap's "Spatial Description" cell is the external reference for EN-B: π0.5 11%). Goal block C (topology/EC, the cleanest Turkish on the token side). Block D (*sağına/soluna*, "to the right / left") is language side only (no predicate) → **not** in this RQ. Tier 2 rows (B07–B11) need the counterfactual BDDL (`datasets/04/data/bddl_cf/`, 5 files ready; loading in a real env is status: hypothesis). Sample: 50 inits per task; for the surface axis, a pilot first (10 init × 5 surfaces × 1 task / block). Suite weight is an advisor decision (S7).

**Systems:** S-OV and S-VN, the same checkpoints as RQ-a. The surface comparison (especially `tr_ascii`) is the only cell where **opposite directions** are expected in the two systems (Lecture 22 hypothesis) → running the two systems together is mandatory in this RQ.

**If yes / if no, what the thesis says:** *Family/surface effect exists:* "The Turkish drop is not uniform; it is concentrated in the [topology/CDC] family and the [tr_ascii] surface; the letter cost shows in S-OV and not in S-VN" → the tokenization and pre-training explanations separate, and the hypothesis of Tatar et al. is tested in closed loop. *None (uniform):* "The drop is independent of the form of the command, it comes from the language itself" → surface fixes (ASCII-fication, suffix-free writing) are eliminated on the solution side, RQ-c narrows to the backbone/training path. *H0-b3 rejected:* evidence of strong reading — the boldest sentence of the thesis. *H0-b3 could not be rejected:* "the Turkish new recipe is equivalent to the empty command" → RQ-c's criterion drops to weak reading (S10).

---

## RQ-c — How does it close?

**Question:** How much of the Turkish gap (RQ-a's Δ) is closed by two interventions that do not touch the model (MT translation layer, rule-based parser → English template) and one that does (EN+TR co-training LoRA, if the budget allows) — and does the closure stay at the level of **weak reading** (`tr ≈ clean` in blocks A/C) or reach **strong reading** (above `empty` in block B)?

**Null hypothesis:** H0-c1: for intervention M ∈ {MT, parser, LoRA} P(success | tr, M) = P(success | tr, no intervention). H0-c2 (ceiling): P(success | tr, M) = P(success | clean) — the intervention closes the gap completely (equivalence, δ = 15). H0-c3 (strong reading, LoRA only): in block B P(success | tr_B, LoRA) = P(success | empty). H0-c4 (side effect, LoRA only): P(success | clean, LoRA) = P(success | clean) — English is not degraded (Chen: EN −2.7 with E-FT/M-CT; δ = 5).

**Variables:** Independent: intervention {none, MT (Opus-MT tr-en), parser (`tr_parse.py` → EN template), LoRA (EN+TR mixed, 4-bit, T4)}; command condition {clean, tr_a, tr_B, empty}. Dependent: episode success; for MT/parser also the **intermediate output accuracy** (is the translation/parse exact, or "wrong but valid" — Lecture 24 H04) and latency (ms). Covariate: translation/parse error type. Fixed: RQ-a protocol; for LoRA the EN:TR training data ratio and the number of steps are written in advance.

**Criterion and threshold:** Primary: closure rate κ = [SR(tr, M) − SR(tr)] / [SR(clean) − SR(tr)] (0 = none, 1 = full), 95% bootstrap CI (over paired episodes). "It works": κ ≥ 0.5 and McNemar p < 0.05. "Closes fully": H0-c2 within TOST. Strong reading: rejection of H0-c3 (tr_B − empty ≥ 15). Side effect: reported if H0-c4 falls outside TOST δ=5. For MT/parser, error decomposition: how much of the gap comes from intermediate-layer error (translation wrong → command wrong) (the "wrong but valid" count of Lecture 24).

**Suite / block / sample:** Weak reading: Spatial A + Goal C (RQ-a cells are reused; the intervention only adds new rows). Strong reading: Spatial **B** (LoRA only; MT/parser reduce to English in B by definition, and take EN-B's result). Sample: RQ-a's 50 inits; the LoRA condition only on S-OV (OpenVLA `finetune.py --use_quantization`; Lecture 24 estimate 65–98 T4 hours, S9 advisor decision); MT/parser on both systems (seconds/ms on the Mac).

**Systems:** MT and parser: S-OV + S-VN (do not touch the model, applied to both). LoRA: S-OV (open training code; VLANeXt training does not fit the T4 — hypothesis, Lecture 24). The system-difference warning applies: a sentence like "MT closes less on S-VN" stays at the system level.

**If yes / if no, what the thesis says:** *MT/parser κ ≥ 0.5:* "The Turkish gap can largely be closed at the input level; the model does not understand Turkish, it works with a command reduced to English" (weak reading; depending on S10 the thesis ends here or this becomes the floor). *MT/parser κ < 0.5:* part of the gap is not from intermediate-layer error (it is low even when the translation is correct) → an in-model problem; the LoRA/backbone path is justified. *LoRA H0-c3 rejected:* "A Turkish spatial recipe, in the same scene, rose above the empty command" → strong reading, the main contribution of the thesis; reported together with the H0-c4 side effect. *LoRA H0-c3 not rejected:* "co-training taught the trained sentences, not the recipe" → the Turkish counterpart of LangGap's "data alone is not enough" finding; the thesis writes this as a negative result.

---

## Ordering rule (by S1 pattern; Lecture 26 §6)

| S1 pattern (Lecture 19) | Primary RQ | Rationale | Do immediately |
|---|---|---|---|
| A — language is ignored | RQ-b (suite axis) | No scale in Spatial task 0; move to Goal C and Spatial B | `conditions_goal8.json` run |
| B — Turkish is not read (tr ≈ empty) | RQ-c (MT/parser first) | The gap is large, the input-level intervention is cheap; LoRA for strong reading depends on S9 | `mt_layer_mac.py`, a `tr_mt` condition for Lecture 18 |
| C — wrong biased | RQ-b (frame + `grasped_bowl`) | The target is chosen from the scene; Turkish's share shows only in block B | Label the 10 `wrong` videos |
| D — Turkish is read | RQ-a (two systems) → RQ-b (surface) | If the difference is small, the equivalence test and the comparison with S-OV come forward | S-OV Spatial 4-condition run |
| inconclusive | RQ-a (increase n) | n=10 normal result | 50 init + ≥3 repetitions |

## Predictions written in advance (Mesut fills in — BEFORE the run, with a commit)

```
RQ-a: Δ(tr_a − clean) S-VN: ___ points   S-OV: ___ points   ; tr − empty: ___
RQ-b: most broken family: ______  surface: ______  ; tr_B > empty? yes / no
RQ-c: MT κ ≈ ___   parser κ ≈ ___   LoRA tr_B − empty ≈ ___
My primary RQ prediction: RQ-_   (one-sentence rationale: ______________________)
```

Version: v0.1 — Lecture 26 (automatic draft; ordering is conditional). Changes are added to this section with a date.
