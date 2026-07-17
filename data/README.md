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

## Regenerating / extending

The files are the deterministic output of the generator — edit it and re-run:

```bash
python data/build_suites.py
```

## Loading and preprocessing

```python
from data import load_behavior_suite, load_edit_targets, explore_suite
from data.preprocess import verify_single_token_answers, make_splits

recs = load_behavior_suite("factual_recall")   # list[dict]
explore_suite("agreement")                      # quick EDA
splits = make_splits(recs)                       # train/val/test, contrast pairs kept together
```

`verify_single_token_answers(model, recs)` drops any item whose answer is not a single BPE token,
which keeps the logit-difference comparisons well defined.
