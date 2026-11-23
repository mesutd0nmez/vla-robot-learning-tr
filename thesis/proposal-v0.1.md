<!-- Thesis proposal draft v0.1 — Lecture 29. One page: body ≤ 700 words (References excluded). Every numeric claim carries a tag: [Lecture NN], [arXiv:…], [proposal] or [calc]. Check: python code/day29/proposal_lint.py thesis/proposal-v0.1.md -->

# Thesis Proposal (draft v0.1)

**Title (TR):** Türkçe Mekânsal Komutların Görüntü-Dil-Eylem (VLA) Modellerinde Temellendirilmesi: Boş-Komut Tabanlı Kapalı-Döngü Bir Değerlendirme
**Title (EN):** Grounding Turkish Spatial Instructions in Vision-Language-Action Models: A Closed-Loop Evaluation Against an Empty-Instruction Baseline
**Student / Advisor:** Mesut Bahtiyar Dönmez / (advisor) · ALKU Graduate School, Computer Engineering (MSc with thesis)

## 1. Problem and significance

"Language is a beautiful compression of the world, but it is not the world" (P. Grady; the advisor's slide) [Lecture 1]. VLA models are trained to connect the command to the world — the camera image and the robot's action — but almost only with English data. Multilingual measurements report sharp drops: with a Korean command, LIBERO-Spatial success falls from 97.6 to 49.2 [arXiv:2606.11906]; in OpenVLA-OFT the loss for a foreign language (−27.8…−29.8) is the same size as giving no command at all (−27.5) [arXiv:2606.15714]. For Turkish, the only study measures open-loop, on a single frame [Lecture 5]. We cannot say that a model really reads the Turkish "kâsenin solundaki" ("the one to the left of the bowl") without separating it from the empty command.

## 2. Aim

To measure, against the empty-command baseline, how well Turkish spatial commands are grounded in closed loop in two open VLA systems; to find where the drop is concentrated; and to show how much of the gap a cheap input-level fix closes.

## 3. Research questions

- **RQ-a (is there a difference):** In the same initial scene, is there a success difference between the Turkish command and the English command; where is the difference relative to the empty-command baseline? [Lecture 26]
- **RQ-b (where):** Does the drop depend on the relation family, the surface (Turkish letters, suffixes) and the reference frame; does an unseen Turkish recipe rise above the empty command? [Lecture 26]
- **RQ-c (how it closes):** How much of the gap do machine translation and a rule-based parser close (closure rate κ)? [Lecture 26]

## 4. Method

**Benchmark:** LIBERO-Spatial and LIBERO-Goal; 50 initial states per task [Lecture 27]. **Data:** 20 paired EN–TR commands, four blocks (A trained Spatial 6, B new recipe 6, C Goal 6, D 2) and four Turkish surfaces [Lecture 20]. **Systems:** S-OV = the OpenVLA-7B LIBERO checkpoint (LLaMA-2 tokenizer; 4-bit on the T4) and S-VN = VLANeXt (Qwen3-VL backbone) [Lecture 27]; the comparison is at system level, not causal [arXiv:2606.15714]. **Conditions:** `clean`, `empty`, `wrong`, `tr` — paired on the same initial state [Lecture 18]. **Measure:** paired McNemar and TOST equivalence; for "there is a difference", p < 0.05 and at least 15 points, with thresholds written before the run (preregistration) [Lecture 26]. Since the smallest visible difference with 50 states per task is 23 points, the main claim is built at the block level: 6 tasks × 20 states give 13 points [Lecture 27]. The baseline and ablation plan is in `thesis/ablation-plan.md` [Lecture 28].

## 5. Expected contributions

1. A Turkish spatial command set: paired, labeled by block and surface, openly licensed [Lecture 20].
2. The first closed-loop, empty-command-based VLA measurement for Turkish; the behavioral counterpart of two tokenizers (Turkish letter cost 4.9 and 0.9 tokens) [Lecture 22].
3. A preregistered, low-budget (free T4) reproducible evaluation protocol and its code [Lecture 27].

## 6. Schedule

The unit of account is Kaggle T4 hours; lean matrix ≈ 108 hours, full core ≈ 258 hours [calc: Lecture 27]; weekly quota ≈ 30 hours (user report) [Lecture 27]. Work packages (week 1 = 30 November 2026) [proposal]: WP1 pilot + calibration (weeks 1–2); WP2 RQ-a (weeks 2–5); WP3 RQ-b (weeks 5–7); WP4 RQ-c (weeks 7–8); WP5 analysis (weeks 8–10); WP6 writing (weeks 9–16). Midterm: RQ-a table in week 5; final: thesis draft in week 16 [proposal].

## 7. Risks and plan Bs

| Risk | Plan B |
|---|---|
| Model calls slower than estimated (pessimistic 497 hours) [Lecture 27] | lean matrix; only Spatial + Goal blocks A/C |
| 4-bit S-OV does not reach the published 84.7 [Lecture 27] | label it "OpenVLA (4-bit)"; paired bf16 subset on the Mac |
| Empty command near the ceiling in Spatial (no-command 9/10) [Lecture 19] | shift weight to Goal (no-command 1/10) [Lecture 19] |
| Counterfactual BDDL does not load | block B stays descriptive, with EN-B as an external reference [arXiv:2603.00592] |
| LoRA training is not approved | RQ-c stays at input level (MT + parser) |

## References (short)

Kim et al. 2024, OpenVLA, arXiv:2406.09246 · Liu et al. 2023, LIBERO, arXiv:2306.03310 · Dong et al. 2026, arXiv:2606.11906 (ACL 2026) · Chen et al. 2026, arXiv:2606.15714 · Hou & Zhao 2026, LangGap, arXiv:2603.00592 · Tatar et al. 2026, SIU (IEEE) · VLANeXt: github.com/DravenALG/VLANeXt · Details: `lectures/lecture-05…28`, `thesis/rq-candidates.md`, `thesis/experiment-matrix.csv`.

Version: v0.1 — Lecture 29 (automatic draft; to be reviewed with the advisor). Changes are added here with a date.
