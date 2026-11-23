# Dataset name

**Type:** self-produced / external · **Version:** v0.1 · **Date:** — · **License:** — · **Related project/lecture:** —

## What it contains
One paragraph: how many examples, which languages, which tasks.

## Source and acquisition
- External data: source link, version/commit, download command, file size, `sha256sum` output.
- My own data: how it was produced (hand-written / translation / script), who produced it, verification (second reading, tokenizer check).

## Schema
| Field | Type | Description | Example |
|---|---|---|---|
| id | str | unique id | `spatial_03_tr_a` |
| suite | str | LIBERO suite | `libero_spatial` |
| task_id | int | task index | 3 |
| lang | str | `en` / `tr` | `tr` |
| instruction | str | command text | `siyah kâseyi tabağın üstüne koy` ("put the black bowl on the plate") |
| relation | str | spatial relation (EC/NTPP/1-cross/distance/between) | `EC` |
| frame | str | viewpoint: `camera` / `robot` / `n/a` | `n/a` |
| suffixes | str | Turkish suffixes, separated by `;` | `-ın;üst-ü-ne` |
| variant | str | paraphrase variant | `a` |

## Applied transformations
Image rotation, gripper sign, normalization, filtering — in order, with the script name.

## Known limitations
Number of examples, viewpoint ambiguity, single translator, etc.

## Usage
```python
import pandas as pd
df = pd.read_csv("data/commands.csv")
```

## Changelog
- v0.1 — first version
