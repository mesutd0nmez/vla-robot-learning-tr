# Lecture 28 — surface ablation check (20 pairs)

- tr_a_seg with hyphens removed equals tr_a: 20/20
- letter step (tr_a → tr_ascii) changes only letters: 20/20
- suffix step (tr_a → tr_bare) strips only suffixes (stem, with consonant hardening undone where needed): 20/20
- word count tr_a = tr_bare: 20/20
- suffix count tr_a: total 111, per pair 5.5 (0 in tr_bare)
- Turkish letters: tr_a 97, tr_bare 31, tr_ascii 0, tr_bare_ascii 0
- datasets/05 tokens.csv cross-check (4 surfaces × 20 × 2 tokenizers): mismatches 0

## 2×2 token table (mean tokens per pair; min–max in parentheses)

| tokenizer | en | tr_a (TR, suffix) | tr_ascii (ASCII, suffix) | tr_bare (TR, no suffix) | tr_bare_ascii (ASCII, no suffix) | letter effect | suffix effect | interaction |
|---|---|---|---|---|---|---|---|---|
| llama2 | 13.6 (7–21) | 22.4 (12–36) | 17.6 (7–30) | 14.9 (7–25) | 14.3 (6–24) | +2.7 | +5.4 | +4.3 |
| qwen3 | 12.3 (6–19) | 16.9 (8–28) | 16.0 (8–26) | 12.2 (6–20) | 12.2 (6–20) | +0.4 | +4.2 | +0.9 |

letter effect = TR letters − ASCII (mean over the two suffix levels); suffix effect = suffixed − bare; interaction = difference of the letter effect between suffixed and bare.
