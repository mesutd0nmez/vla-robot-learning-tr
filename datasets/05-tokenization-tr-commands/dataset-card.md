# Tokenization of the TR command set — 100 commands × 2 tokenizers (LLaMA-2 / Qwen3)

**Type:** self-produced (derived on top of datasets/04) · **Version:** v0.1 (Qwen3-VL tokenizer identity awaits verification from HF) · **Date:** Lecture 22 · **License:** MIT (tables); tokenizer files do not enter the repo (Meta LLaMA 2 Community License / Apache 2.0) · **Related project/lecture:** Lecture 22, `code/day22/`; input `datasets/04-tr-spatial-commands/` v0.1; continuation of Lecture 15's `token_counts.json`

## What it contains
The 100 commands of `datasets/04` (20 EN–TR pairs × 5 surfaces) were passed through the tokenizers of two VLAs: **LLaMA-2** (OpenVLA; SentencePiece BPE, 32,000 pieces, Meta `tokenizer.model`, sha256 verified) and **Qwen3** (the Qwen3-VL-2B backbone of VLANeXt; byte-level BPE, 151,643 + 26 special). Per row: token/word/byte counts, pieces and ids; the pieces of the relation word (`rel_span`) and the morpheme–token boundary overlap; the token shapes carrying the relation suffix; a check of tokens falling in OpenVLA's action range (ids 31744–31999) in LLaMA-2; a cross-check of the llama-tokenizer-js count from Lecture 15 against SentencePiece (100/100 the same). No model was run; all numbers are tokenizer output.

## Files
| File | Content |
|---|---|
| `data/tokens.csv` | 200 rows (100 commands × 2 tokenizers): `n_tokens`, `n_words`, `fertility`, `bytes_per_token`, `n_space_alone` (lone ▁), `n_byte_fragments`, `n_in_action_range`, `prompt_tokens` (LLaMA: the OpenVLA prompt including BOS; Qwen: chat template text + 1 image placeholder), `pieces`, `ids` |
| `data/rel_span.csv` | 40 rows (20 pairs × 2): the pieces of the relation word within the sentence, `recall`/`precision` (morpheme boundary ↔ token boundary), `last_suffix_separate` |
| `data/seg_align.csv` | 40 rows: overlap counts for all morpheme boundaries in `tr_a_seg` |
| `data/words.csv` | 274 rows: every distinct word in the set × 2 tokenizers, sentence-initial and sentence-medial pieces |
| `data/suffix_forms.csv` | token shapes carrying the `-ki` and dative (`-nA/-A`) suffixes and their counts |
| `data/action_collision.csv` | list of tokens falling in the action range (empty) + sample contents of the range |
| `data/crosscheck.md`, `data/js_ids.json` | llama-tokenizer-js id sequences and the comparison with SentencePiece |
| `data/probe_edge_cases.txt` | output of `probe_edge_cases.py`: `lower()`/İ, Turkish-letter starts, the Qwen sentence-initial difference, action range (tr/ko/ja/zh/ru/ar), `Kal|dır`, false friends |
| `data/summary.md` | summary tables (the source of the numbers in the lecture text) |

## Source and acquisition
- LLaMA-2 tokenizer: `tokenizer.model` (499,723 bytes), `curl -L -o tokenizer.model https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.model`; `sha256 = 9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347` (the same as `tokenizer_checklist.chk` in Meta's LLaMA / LLaMA 2 distribution). OpenVLA uses the `LlamaTokenizerFast` derived from this file (`openvla/openvla-7b`; the added `<PAD>` id 32000 does not affect the text). `sentencepiece==0.2.3`, `add_dummy_prefix` (▁ at the start of every text).
- Qwen3 tokenizer: `tokenizer.json` (byte-level BPE, 151,387 merges; `normalizer: NFC`; 26 added special tokens: `<|im_start|>`, `<|vision_start|>`, `<|image_pad|>`, `<think>`…), taken in this session from the npm package `@lenml/tokenizer-qwen3@3.7.2` (there was no HF access); `"Hello world" → [9707, 1879]` was verified against the known Qwen2/2.5/3 ids. **Hypothesis:** byte-identical to the `tokenizer.json` of `Qwen/Qwen3-VL-2B-Instruct` (same `Qwen2Tokenizer` class, same vocab) — Mesut will compare the `ids` column in `tokens.csv` with HF on the Mac (v0.2).
- Warning: the `tokenizer.json` of the npm package `@lenml/tokenizer-llama2` is **not** Meta's vocab (id 29871 is 'ข' instead of ▁; a different merge order) — it was not used.
- Commands: `datasets/04-tr-spatial-commands/data/commands.csv` v0.1 (the version before the native-speaker reading).
- Production: `code/day22/tokenize_set.py --llama tokenizer.model --qwen tokenizer.json --jsids js_ids.json`; `count_js_ids.mjs` (llama-tokenizer-js 1.2.2); `probe_edge_cases.py`.

## Schema (`data/tokens.csv`)
| Field | Type | Description | Example |
|---|---|---|---|
| id, pair_id, block, variant, lang | str | ids from datasets/04 | `B07_tr_a`, `B07`, `B`, `tr_a`, `tr` |
| tokenizer | str | `llama2` / `qwen3` | `llama2` |
| n_tokens | int | command text only (no BOS/template) | 21 |
| n_words, n_chars, n_bytes | int | whitespace-separated words, characters, UTF-8 bytes | 8, 45, 51 |
| fertility | float | n_tokens / n_words | 2.625 |
| bytes_per_token | float | n_bytes / n_tokens | 2.43 |
| n_space_alone | int | lone ▁ tokens (in LLaMA `▁ç`, `▁ş`, `▁ğ`, `▁ı` are not in the vocab) | 0 |
| n_byte_fragments | int | pieces that are not valid UTF-8 on their own | 0 |
| n_in_action_range | int | LLaMA-2 id ≥ 31744 (OpenVLA action bin) | 0 |
| prompt_tokens | int | LLaMA: `In: What action should the robot take to {command.lower()}?\nOut:` including BOS; Qwen: chat template text + 1 image placeholder | 35 |
| pieces | str | pieces separated by `\|`; ▁ = word-initial space | `▁sold\|aki\|▁si\|y\|ah\|…` |
| ids | str | token ids separated by spaces | `5239 9940 1354 …` |

`rel_span.csv`: `pair_id, rel_span, rel_seg, n_morphemes, tokenizer, n_pieces, pieces, ids, morph_cuts, token_cuts, cuts_hit, recall, precision, last_suffix, last_suffix_separate`. `words.csv`: `lang, word, count, tokenizer, n_pieces_mid, pieces_mid, ids_mid, n_pieces_initial, pieces_initial, occurs_initial`.

## Applied transformations
- The command text was not changed (NFC, lowercase — datasets/04 already is). In the LLaMA prompt count `.lower()` was applied as OpenVLA does (no effect on the rows in the set).
- In Qwen, word pieces depend on position: the sentence-initial word was tokenized without a space, the others as `' ' + word` (in the VLANeXt template the command comes right after `<|vision_end|>` — hypothesis, Lecture 13).
- Morpheme boundaries from the `tr_a_seg` / `rel_seg` columns of `datasets/04` (segmentation by the rule-based `tr_morph.py`); token boundaries from character offsets.

## Known limitations
- Qwen3-VL's tokenizer comes from the npm package, not from HF (v0.1); instead of OpenVLA's HF tokenizer, Meta `tokenizer.model` + sentencepiece (should be the same as HF `legacy=True` behavior; edge cases such as special-token adjacency were not tested).
- No `tr_b` segmentation (it is hand-written, so `seg_align` covers `tr_a` only).
- Token count ≠ understanding: this set measures only the form of the input; whether its effect shows up in success rate depends on the Lecture 18 run.

## Usage
```python
import pandas as pd
t = pd.read_csv("data/tokens.csv")
t.pivot_table(index="variant", columns="tokenizer", values="n_tokens", aggfunc="mean")
```

## Changelog
- v0.1 — Lecture 22: first version (on top of datasets/04 v0.1; Qwen3 tokenizer from npm, LLaMA-2 Meta tokenizer.model sha256 verified)
