# Generated samples

Everything in this directory is produced by one command from the repository root:

```bash
python src/model_runner.py
```

The committed copies come from the run recorded in [`run_metadata.json`](run_metadata.json):
**pythia-160m**, CPU, greedy decoding, 30 new tokens per prompt, seed 0, ~60 seconds end to end.

## What was generated

The pipeline produces two kinds of sample.

**Part A — 10 behavioral prompts** (4 subject–verb agreement, 3 factual recall, 3 induction),
drawn from `data/processed/prompts.jsonl` by a seed-deterministic, behavior-balanced
selector. For each prompt: the top-5 next tokens with log-probabilities, the log-probability
of the correct answer against its matched incorrect one, and a greedy 30-token continuation.

**Part B — 3 factual edits.** For each target the pipeline locates the MLP layer storing the
fact by causal tracing, fits a rank-one update to that layer's output matrix so the subject
maps to a new object, and regenerates. Every prompt is generated twice — before and after —
and the two runs differ by exactly one edited weight matrix.

## Preliminary results

The model behaves as the literature predicts on the behavioral suites: it prefers the
correct continuation on **9 of 10** prompts, mean logit difference **+3.14**. Agreement is the
strongest behavior (+4.20 mean, 4/4 correct), including the attractor cases where a
plural distractor sits between subject and verb (*"The senators from the state → are"*).
Factual recall is correct but often not top-1 — for *"The capital of Spain is"* the model
ranks ` Madrid` behind generic continuations like ` the`, so it knows the fact without
committing to it in one token. The single failure is an induction prompt
(*"riverdesert river"*), where the model prefers a newline over continuing the pattern.

The edits work on 2 of 3 targets: *Mount Everest → Canada* and *The Mona Lisa → Rembrandt*
both flip the generated text ("efficacy" 1.00) while leaving neighbouring facts' top-1
predictions untouched ("top-1 unchanged on neighbours" 1.00). The *Eiffel Tower → Rome* edit
fails at the layer causal tracing selects (6) although the same edit succeeds when layer 5
is forced, which is the main open issue below. Generalization to paraphrases is **0.00**
across all three targets: the rewritten fact does not survive a change of wording. Fluency
is essentially unchanged (4.42 → 4.52 mean 2/3-gram entropy), so the edits are not
destroying the model's text.

| | efficacy | generalization | specificity | top-1 unchanged | fluency before → after |
|---|---|---|---|---|---|
| mean over 3 targets | 0.67 | 0.00 | 0.33 | 0.83 | 4.42 → 4.52 |

Two caveats on reading these numbers. `specificity` counts whether a neighbouring fact's
original answer still appears in the generated text — it scores 0 even when the base model
never knew that fact (pythia-160m places Big Ben in New York *before* any edit), so
`specificity_pred_preserved`, which compares the top-1 prediction before and after, is the
fairer measure of collateral damage. And with 3 targets and 10 prompts these are
illustrative, not statistically meaningful: the milestone's purpose is a working end-to-end
system, and the full suites (48 prompts) are what the final evaluation will use.

## Files

| File | Contents |
|---|---|
| `samples.txt` | Human-readable report — every generation, both parts. Start here. |
| `samples_behavioral.csv` | One row per behavioral prompt: top-k, log-probs, logit difference, generation, fluency. |
| `samples_edit.csv` | One row per (edit target × evaluation prompt × before/after): the generation and its scores. |
| `edit_summary.csv` | One row per edit target: efficacy, generalization, specificity, fluency, edited layer. |
| `run_metadata.json` | Model, config, package versions, dataset counts, and aggregate results for this run. |
| `images/logit_diff_by_suite.png` | Mean logit difference per behavior. |
| `images/edit_quality.png` | Per-target edit scores. |
| `images/edit_fit_loss.png` | Loss curve of the v\* optimization (written by `src/train.py`). |
| `edits/edit_<subject>.json` | Fitted-edit record: layer, update norm, first/final loss. |
| `edits/edit_fit_losses.csv` | Loss per optimization step for every target. |

Naming convention: `samples_*` are per-sample tables, `*_summary` are per-target aggregates,
`images/` holds figures, `edits/` holds fitted-edit artifacts. The binary edit tensors
(`edits/*.pt`) and `run.log` are git-ignored because they are regenerated on every run.

## Reproducing

```bash
python src/model_runner.py                 # regenerate everything above (~1 min on CPU)
python src/model_runner.py --skip-edit     # Part A only (~15 s)
python src/train.py                        # refit the edits and the loss curve
```

Results are deterministic given the same seed, model, and package versions (recorded under
`versions` in `run_metadata.json`) — decoding is greedy and every random draw is seeded.

The same run inside the Docker container (Linux, Python 3.11, identical package versions)
reproduced every generation, top-1 prediction, and fluency score in Part A, with log-probabilities
agreeing to ~1e-3. Two binary Part B scores flipped on that float noise, because the edit is fitted
by 25 Adam steps — see limitation 6 in the top-level README.
