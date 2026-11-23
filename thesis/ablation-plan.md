# Baseline + ablation plan — v0.1 (Lecture 28)

This file is the bridge between `rq-candidates.md` (the RQs) and `experiment-matrix.csv` (the cells): it writes down in advance **what** we will read the result of each RQ against (baseline) and which difference belongs to **which single factor** (ablation). Details: `lectures/lecture-28.md`. Code: `code/day28/`.

Rule: a comparison counts as an **ablation** only if a single factor changes between the two conditions; if several factors change together, it is written as a **descriptive comparison** and no causal sentence is made.

## Baselines

| baseline | what it says | from | cost |
|---|---|---|---|
| No-command ceiling / 1/k | the highest success reachable without language (Spatial 9/10, Goal 1/10) | Lecture 19, analytic | 0 |
| `empty` | the true success of the language-free policy; reference for the claim "Turkish is not read" (H0-a2) | A1–A4, B5 | in the matrix |
| `wrong` | does the model follow the language (faithful) or choose the goal from the scene (biased) | A1–A4 | in the matrix |
| `clean` | upper reference; Δ(tr − clean) | A1–A4 | in the matrix |
| EN-B | English recipe not seen in training; TR-B − EN-B = the "Turkish" cost, EN-B − clean = the "new recipe" cost; external reference LangGap Spatial Description 11% (π0.5) | B3/B4 | in the matrix |
| Published numbers | calibration of the systems (S-OV Spatial 84.7 ± 0.9) | K1, K2 | ~4 h |

## Ablations

| id | question (single factor) | comparison | type | cell | status |
|---|---|---|---|---|---|
| ABL-letter | cost of the Turkish letters | tr_a − tr_ascii (with suffixes) and tr_bare − tr_bare_ascii (without suffixes) | ablation (only the letters change in 20/20 pairs) | A1–A4, B1–B2 (+ AB1) | ready |
| ABL-suffix | cost of the suffixes | **tr_ascii − tr_bare_ascii** (at the ASCII level) | ablation — tr_a − tr_bare is NOT pure: it also deletes 66 of the 97 Turkish letters | AB1 (Goal C, core), AB2 (Spatial A, extension) | ready |
| ABL-interaction | does the letter cost grow in suffixed words (LLaMA-2 tokens: +4.3) | 2×2 interaction | ablation, but **descriptive in power** (~16× the sample for an interaction; Gelman) | AB1 | ready |
| ABL-object | cost of the object names being Turkish | tr_cs − tr_a (Turkish skeleton, English object names; Chen et al. Table 3) | ablation | AB3 (extension) | blocked: the `tr_cs` surface has not been written (datasets/04 v0.3) |
| ABL-MT-oracle | how much of the MT gap is translation error | tr_mt − clean (oracle = human translation = `clean`, because tr_a was produced from English) | re-reading (no new episodes); S-OV greedy → exact episode-level decomposition | C1 + A1–A4 | C1 blocked (MT output) |
| ABL-parser-oracle | parser gap | tr_parse − clean | re-reading; **a tautology on A/C** (the parser was written looking at the set, 79/80 exact) → the real evidence is held-out | C2 | ready |
| ABL-quantization | effect of 4-bit | S-OV bf16 (Mac) − S-OV 4-bit (T4), same 25 inits | **two factors** (precision + device/kernel) → written as "S-OV-4bit-T4 vs S-OV-bf16-MPS" | K3 | ready |
| — system | backbone, action head, or data | S-OV − S-VN | **descriptive** (three factors together); splitting it needs a StarVLA layout (same head, two backbones; Chen et al.) — out of budget | A1–A4 | out of scope |

## Budget (Lecture 27 model, `code/day28/price_ablations.py`)

Matrix v0.1 → v0.2: AB1 (core), AB2, AB3 (extension) added; B6 reduced to `tr_b` only (`tr_bare` moved to AB1/AB2).

| matrix | inits per task | pilot + core T4 hours optimistic / medium / pessimistic | Kaggle weeks (medium) | extension (medium) |
|---|---|---|---|---|
| v0.1 | 50 | 151 / 258 / 504 | 8.6 | 225 |
| v0.1 | 20 | 63 / 108 / 211 | 3.6 | 225 |
| v0.2 | 50 | 178 / 303 / 594 | 10.1 | 254 |
| v0.2 | 20 | 74 / 126 / 247 | 4.2 | 254 |

Reading rule written in advance: if ABL-letter comes out ≥ 15 points on S-OV and |Δ| < 15 on S-VN (TOST δ = 15) → "the letter cost is consistent with tokenization" (closed-loop evidence for the Lecture 22 hypothesis); ≥ 15 on both systems → "tokenization is not the only explanation"; < 15 on both → "letters do not matter, the drop is elsewhere" (suffix, word, language). ABL-suffix with the same rule. The interaction is reported in direction mode only.

Version: v0.1 — Lecture 28 (automatic draft). Changes are added here with a date.
