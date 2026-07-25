# Data

Because we study a **pretrained** model, the "dataset" is a set of hand-authored **prompt suites**
(minimal pairs) plus a small set of **edit targets**. There is no large corpus to clean.

## Files

| File | Rows | Purpose |
|---|---|---|
| [`prompts/subject_verb_agreement.jsonl`](prompts/subject_verb_agreement.jsonl) | 20 | Agreement behavior; each item has a distractor noun of opposite number. |
| [`prompts/factual_recall.jsonl`](prompts/factual_recall.jsonl) | 12 | Factual recall (capitals + landmarks). |
| [`prompts/induction.jsonl`](prompts/induction.jsonl) | 16 | Induction: continue a pattern shown once earlier in the prompt. |
| [`edit_targets.json`](edit_targets.json) | 3 | ROME edit targets with paraphrase (generalization) and neighborhood (specificity) prompt sets. |

## Record schema (`.jsonl`)

```json
{"id": "agr_000_s", "type": "agreement",
 "prompt": "The key on the table", "correct": " is", "incorrect": " are",
 "contrast_id": "agr_000_p"}
```

Every item is a **minimal pair**: `prompt` should be continued by `correct`, not `incorrect`, and
`contrast_id` points at the matched item that flips the answer. This lets each behavior be scored as
a *difference* of probabilities rather than a single number.

## Edit-target schema (`edit_targets.json`)

```json
{"subject": "The Eiffel Tower",
 "prompt": "The Eiffel Tower is located in the city of",
 "true": "Paris", "new": "Rome",
 "paraphrases":  ["You can find the Eiffel Tower in the heart of", "..."],
 "neighborhood": [["Big Ben is located in the city of", "London"], ["...", "..."]]}
```

## Processed dataset (`processed/`)

The pipeline never reads the raw files directly. [`src/data_loader.py`](../src/data_loader.py)
validates and normalizes them (whitespace collapsed, answers stripped, missing fields rejected),
assigns a deterministic train/val/test split with contrast pairs kept together, and writes:

| File | Contents |
|---|---|
| `processed/prompts.jsonl` | 48 records, one per prompt, with `suite`, `split`, and length fields added. |
| `processed/edit_targets.json` | The 3 validated edit targets (subject must occur in the prompt). |
| `processed/manifest.json` | Counts per suite and split, SHA-256 of every raw source file, and the split seed. |

The build records no timestamps, so it is byte-for-byte reproducible and CI checks it with
`python src/data_loader.py --check`.

## Regenerating / extending

Both layers are the deterministic output of their generators — edit and re-run:

```bash
python data/build_suites.py     # raw suites   (prompts/*.jsonl, edit_targets.json)
python src/data_loader.py       # processed dataset (data/processed/)
```

## Loading and preprocessing

```python
# pipeline layer — what src/model_runner.py uses
from src.data_loader import load_processed_prompts, load_processed_edit_targets, select_inference_samples

records = load_processed_prompts(suite="factual_recall", split="test")
batch   = select_inference_samples(load_processed_prompts(), n_samples=10, per_suite=4, seed=0)

# raw layer — for exploration and custom experiments
from data import load_behavior_suite, load_edit_targets, explore_suite
from data.preprocess import verify_single_token_answers, make_splits

recs = load_behavior_suite("factual_recall")   # list[dict]
explore_suite("agreement")                      # quick EDA
splits = make_splits(recs)                       # train/val/test, contrast pairs kept together
```

`verify_single_token_answers(model, recs)` drops any item whose answer is not a single BPE token,
which keeps the logit-difference comparisons well defined. It needs a loaded model, so it is applied
at inference time rather than in the offline build.
