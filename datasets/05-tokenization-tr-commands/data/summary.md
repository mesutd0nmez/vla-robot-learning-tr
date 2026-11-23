# Lecture 22 — tokenization summary (real numbers; `tokenize_set.py`)

Tokenizers: llama2: tokenizer.model sha256=OK (Meta LLaMA/LLaMA-2) (vocab 32000); qwen3: tokenizer.json (vocab 151643)

## Tokens per command (surface form × tokenizer; mean over 20 pairs, min–max)

| surface | LLaMA-2 | Qwen3 | ratio L/Q | fertility L | fertility Q |
|---|---|---|---|---|---|
| en | 13.6 (7–21) | 12.3 (6–19) | 1.11 | 1.14 | 1.01 |
| tr_a | 22.4 (12–36) | 16.9 (8–28) | 1.33 | 3.09 | 2.36 |
| tr_b | 21.9 (13–37) | 16.1 (8–26) | 1.37 | 3.35 | 2.44 |
| tr_ascii | 17.6 (7–30) | 16.0 (8–26) | 1.10 | 2.35 | 2.20 |
| tr_bare | 14.9 (7–25) | 12.2 (6–20) | 1.22 | 2.02 | 1.68 |

## Turkish/English token ratio (same pair, tr_a / en)

- llama2: tr_a/en mean **1.63** (min 1.20, max 2.00)
- qwen3: tr_a/en mean **1.41** (min 0.89, max 2.00)

## Mean tokens per block (tr_a)

| block | LLaMA-2 | Qwen3 | en |
|---|---|---|---|
| A | 30.7 | 22.8 | 17.7 |
| B | 26.0 | 19.2 | 15.5 |
| C | 13.8 | 10.8 | 8.8 |
| D | 13.0 | 10.5 | 10.0 |

## Byte fragments / standalone whitespace / action range (total over TR rows)

- llama2: byte fragments 0, standalone whitespace (▁) 14, ids falling in the action range 0  (80 rows)
- qwen3: byte fragments 0, standalone whitespace (▁) 0, ids falling in the action range 0  (80 rows)

## Total tokens inside the template

**llama2** — OpenVLA prompt, incl. BOS:
- en: mean 27.6 (min 21, max 35)
- tr_a: mean 36.5 (min 26, max 50)
- tr_b: mean 36.0 (min 27, max 51)
- tr_ascii: mean 31.6 (min 21, max 44)
- tr_bare: mean 28.9 (min 21, max 39)
**qwen3** — text part of the Qwen chat template (+1 image placeholder; a real image is 64 tokens/frame):
- en: mean 23.3 (min 17, max 30)
- tr_a: mean 27.9 (min 19, max 39)
- tr_b: mean 27.1 (min 19, max 37)
- tr_ascii: mean 27.0 (min 19, max 37)
- tr_bare: mean 23.2 (min 17, max 31)

## Pieces of the relation word (`rel_span`) inside the sentence

| pair | word | segmentation | LLaMA-2 | Qwen3 | last suffix separate (L/Q) | boundary recall (L/Q) |
|---|---|---|---|---|---|---|
| A01 | arasındaki | ara-sı-nda-ki | `▁ar|as|ı|nd|aki` (5) | `▁arasındaki` (1) | hayır / hayır | 0.33 / 0.00 |
| A02 | yanındaki | yan-ı-nda-ki | `▁y|an|ı|nd|aki` (5) | `▁yan|ındaki` (2) | hayır / hayır | 0.67 / 0.33 |
| A03 | ortasındaki | orta-sı-nda-ki | `▁ort|as|ı|nd|aki` (5) | `▁ort|as|ındaki` (3) | hayır / hayır | 0.33 / 0.00 |
| A04 | üst çekmecesindeki | üst çekmece-si-nde-ki | `▁ü|st|▁|ç|ek|m|ec|es|inde|ki` (10) | `▁üst|▁çek|m|ec|es|inde|ki` (7) | evet / evet | 0.33 / 0.33 |
| A05 | üstündeki | üst-ü-nde-ki | `▁ü|st|ünd|ek|i` (5) | `▁üst|ünd|eki` (3) | hayır / hayır | 0.33 / 0.33 |
| A06 | üstündeki | üst-ü-nde-ki | `▁ü|st|ünd|ek|i` (5) | `▁üst|ünd|eki` (3) | hayır / hayır | 0.33 / 0.33 |
| B07 | soldaki | sol-da-ki | `▁sold|aki` (2) | `sold|aki` (2) | hayır / hayır | 0.00 / 0.00 |
| B08 | sağdaki | sağ-da-ki | `▁sa|ğ|d|aki` (4) | `sa|ğ|d|aki` (4) | hayır / hayır | 0.50 / 0.50 |
| B09 | öndeki | ön-de-ki | `▁ö|nd|ek|i` (4) | `ö|nd|eki` (3) | hayır / hayır | 0.00 / 0.00 |
| B10 | yakın | yakın | `▁y|ak|ı|n` (4) | `▁yakın` (1) | — / — |  /  |
| B11 | uzak | uzak | `▁uz|ak` (2) | `▁uz|ak` (2) | — / — |  /  |
| B12 | yakın | yakın | `▁y|ak|ı|n` (4) | `▁yakın` (1) | — / — |  /  |
| C13 | üstüne | üst-ü-ne | `▁ü|st|ü|ne` (4) | `▁üst|ü|ne` (3) | evet / evet | 1.00 / 1.00 |
| C14 | üstüne | üst-ü-ne | `▁ü|st|ü|ne` (4) | `▁üst|ü|ne` (3) | evet / evet | 1.00 / 1.00 |
| C15 | üstüne | üst-ü-ne | `▁ü|st|ü|ne` (4) | `▁üst|ü|ne` (3) | evet / evet | 1.00 / 1.00 |
| C16 | içine | iç-i-ne | `▁i|ç|ine` (3) | `▁iç|ine` (2) | hayır / hayır | 0.50 / 0.50 |
| C17 | rafa | raf-a | `▁ra|fa` (2) | `▁r|afa` (2) | hayır / hayır | 0.00 / 0.00 |
| C18 | önüne | ön-ü-ne | `▁|ön|ü|ne` (4) | `▁önüne` (1) | evet / hayır | 1.00 / 0.00 |
| D19 | sağına | sağ-ı-na | `▁sa|ğ|ı|na` (4) | `▁sa|ğı|na` (3) | evet / evet | 1.00 / 0.50 |
| D20 | soluna | sol-u-na | `▁sol|una` (2) | `▁sol|una` (2) | hayır / hayır | 0.50 / 0.50 |

## Morpheme boundary <-> token boundary (all tr_a commands, total over 20 pairs)

- llama2: 87 of the 111 morpheme boundaries coincide with a token boundary (recall 0.78); 87 of the 296 within-word token boundaries are morpheme boundaries (precision 0.29)
- qwen3: 78 of the 111 morpheme boundaries coincide with a token boundary (recall 0.70); 78 of the 189 within-word token boundaries are morpheme boundaries (precision 0.41)

## Token forms carrying the relation suffix (how many distinct forms?)

| tokenizer | suffix | token forms (count) |
|---|---|---|
| llama2 | -ki | `aki`×5, `i`×3, `ki`×1 -> **3 forms** |
| llama2 | -ne/na/a (DAT) | `ne`×4, `ine`×1, `fa`×1, `na`×1, `una`×1 -> **5 forms** |
| qwen3 | -ki | `eki`×3, `ındaki`×2, `aki`×2, `▁arasındaki`×1, `ki`×1 -> **5 forms** |
| qwen3 | -ne/na/a (DAT) | `ne`×3, `ine`×1, `afa`×1, `▁önüne`×1, `na`×1, `una`×1 -> **6 forms** |

## Selected words (sentence-medial form)

| word | LLaMA-2 | Qwen3 |
|---|---|---|
| sağına | `▁sa|ğ|ı|na` (4) | `▁sa|ğı|na` (3) |
| soluna | `▁sol|una` (2) | `▁sol|una` (2) |
| sağdaki | `▁sa|ğ|d|aki` (4) | `▁sağ|d|aki` (3) |
| soldaki | `▁sold|aki` (2) | `▁sold|aki` (2) |
| öndeki | `▁ö|nd|ek|i` (4) | `▁ö|nd|eki` (3) |
| üstündeki | `▁ü|st|ünd|ek|i` (5) | `▁üst|ünd|eki` (3) |
| üstüne | `▁ü|st|ü|ne` (4) | `▁üst|ü|ne` (3) |
| ustune | `▁u|st|une` (3) | `▁ust|une` (2) |
| ustundeki | `▁u|st|und|ek|i` (5) | `▁ust|und|eki` (3) |
| içine | `▁i|ç|ine` (3) | `▁iç|ine` (2) |
| icine | `▁ic|ine` (2) | `▁ic|ine` (2) |
| arasındaki | `▁ar|as|ı|nd|aki` (5) | `▁arasındaki` (1) |
| arasindaki | `▁ar|as|ind|aki` (4) | `▁aras|ind|aki` (3) |
| yanındaki | `▁y|an|ı|nd|aki` (5) | `▁yan|ındaki` (2) |
| ortasındaki | `▁ort|as|ı|nd|aki` (5) | `▁ort|as|ındaki` (3) |
| kase | `▁k|ase` (2) | `▁k|ase` (2) |
| kaseyi | `▁k|ase|yi` (3) | `▁k|ase|yi` (3) |
| kâseyi | `▁k|â|sey|i` (4) | `▁k|â|sey|i` (4) |
| koy | `▁ko|y` (2) | `▁k|oy` (2) |
| al | `▁al` (1) | `▁al` (1) |
| it | `▁it` (1) | `▁it` (1) |
| ramekinin | `▁r|ame|kin|in` (4) | `▁r|ame|kin|in` (4) |
| tabağın | `▁tab|a|ğ|ı|n` (5) | `▁tab|ağ|ın` (3) |
| tabagin | `▁tab|agin` (2) | `▁tab|ag|in` (3) |
| ocağın | `▁o|ca|ğ|ı|n` (5) | `▁o|ca|ğ|ın` (4) |
| dolabın | `▁dol|ab|ı|n` (4) | `▁dol|ab|ın` (3) |
| çekmecesindeki | `▁|ç|ek|m|ec|es|inde|ki` (8) | `▁çek|m|ec|es|inde|ki` (6) |
| cekmecesindeki | `▁c|ek|m|ec|es|inde|ki` (7) | `▁c|ek|m|ec|es|inde|ki` (7) |
| rafa | `▁ra|fa` (2) | `▁r|afa` (2) |
| robota | `▁rob|ota` (2) | `▁rob|ota` (2) |
| duran | `▁dur|an` (2) | `▁dur|an` (2) |
| alıp | `▁al|ı|p` (3) | `▁al|ıp` (2) |

## Qwen3: sentence-initial <-> sentence-medial difference (TR words whose piece count changes)

- 31 words; examples: ahsap: initial `ah|sap` / medial `▁a|hs|ap`; arasında: initial `aras|ında` / medial `▁arasında`; arasındaki: initial `aras|ındaki` / medial `▁arasındaki`; cekmecesindeki: initial `ce|km|ec|es|inde|ki` / medial `▁c|ek|m|ec|es|inde|ki`; daha: initial `d|aha` / medial `▁daha`; dolabin: initial `d|ol|abin` / medial `▁dol|abin`; dolabın: initial `d|ol|ab|ın` / medial `▁dol|ab|ın`; dolap: initial `d|ol|ap` / medial `▁dol|ap`

## Overlap with the action range (LLaMA-2 ids 31744–31999)

- tokens falling in the action range over 100 commands: **0**; sample contents of the range: 忠, 错, 단, 면, ķ, 桥

## llama-tokenizer-js cross-check
- 100/100 commands have an id sequence identical to SentencePiece; differences: 0
