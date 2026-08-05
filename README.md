# Opening the Black Box

[![CI](https://github.com/Jesus-D-Jimenez/IE7374_Project/actions/workflows/ci.yml/badge.svg)](https://github.com/Jesus-D-Jimenez/IE7374_Project/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Mechanistic interpretability + ROME-style factual editing of small generative language models.**

Group 25 · Jesus D. Jimenez Ballestas · IE7374 (Generative AI)

## What this project does

This project looks *inside* a small open language model (EleutherAI **Pythia-160m / 410m**, with
**GPT-2 small** as a control), locates the internal components that drive specific behaviors, and
then **uses those components to edit the model and generate text under the edit** — so the work
does not just analyze a model, it produces and evaluates model outputs.

**Objectives**

1. Measure three behaviors — subject–verb agreement, factual recall, and induction — as the
   log-probability gap between a correct and a matched incorrect continuation.
2. Localize *which* attention heads and MLP layers cause each behavior (ablation, activation
   patching, causal tracing, logit lens).
3. **Generative core:** use the localized components to apply a rank-one edit to one stored fact,
   generate text under that edit, and score it on efficacy, generalization, specificity, and
   fluency (Meng et al. 2022).

Nothing is trained from scratch and no weights are fine-tuned: the pretrained model is used
exactly as released, and the single optimization in the project is the fit of the rank-one edit
([`src/train.py`](src/train.py)).

## Quickstart

```bash
git clone <this-repo-url> && cd black-box-lm
pip install -r requirements.txt
python src/model_runner.py          # data -> model -> outputs/  (~1 min on CPU)
```

That one command loads the preprocessed dataset, loads `pythia-160m`, runs inference on 10
behavioral prompts, applies and evaluates 3 factual edits, and writes every generation to
[`outputs/`](outputs). Read [`outputs/samples.txt`](outputs/samples.txt) first.

## Setup

Python ≥ 3.10, CPU only (a GPU merely speeds things up). The first run downloads `pythia-160m`
(~380 MB) from Hugging Face.

```bash
# option A — pip
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                                       # optional: makes the packages importable anywhere

# option B — conda
conda env create -f environment.yml
conda activate black-box-lm
pip install -e .

# option C — Docker (no local Python setup at all)
docker build -t black-box-lm .
docker run --rm -v "$(pwd)/outputs:/app/outputs" black-box-lm
```

The image is verified: it builds from a clean context and runs the full pipeline (2.9 GB on disk,
~2 min to build with a warm base image, ~70 s to run). Mounting `outputs/` writes the generated
samples back to the host; add `-v hf-cache:/app/.cache/huggingface` to keep the model download
between runs.

> **Version pin that matters:** `transformers` must stay below 5.x. Version 5 renamed the GPT-NeoX
> unembedding (`embed_out` → `lm_head`), which breaks TransformerLens's Pythia weight conversion
> with `AttributeError: 'GPTNeoXForCausalLM' object has no attribute 'embed_out'`. `requirements.txt`
> pins `transformers>=4.40,<5` and `transformer_lens>=2.0,<3` accordingly.

## Usage

```bash
# --- the pipeline -----------------------------------------------------------
python src/model_runner.py                        # full run: 10 samples + 3 edits
python src/model_runner.py --skip-edit            # behavioral samples only (~15 s)
python src/model_runner.py --n-samples 12 --max-new-tokens 60
python src/model_runner.py --model gpt2           # the replication control model
python src/model_runner.py --edit-layer 5         # force the edited layer, skip causal tracing
python src/model_runner.py --help                 # every flag

# --- dataset ----------------------------------------------------------------
python data/build_suites.py                       # regenerate the raw prompt suites
python src/data_loader.py                         # rebuild data/processed/
python src/data_loader.py --check                 # verify the committed dataset is current

# --- fitting the edit -------------------------------------------------------
python src/train.py                               # fit + save every edit, plot the loss curve
python src/train.py --target 0 --steps 50

# --- the original experiment scripts ----------------------------------------
python scripts/run_baseline.py                    # Experiment 1 over the full suites
python scripts/run_ablation.py --prompt "The capital of Japan is" --correct " Tokyo" --incorrect " Beijing"
python scripts/run_edit.py                        # Experiment 4 over all targets

# --- the extended study (writes outputs/study/) ------------------------------
python scripts/run_logit_lens.py                  # Exp 2:  where the answer forms, by depth
python scripts/run_scaling.py                     # Exp 5:  gpt2 / 160m / 410m on all 48 prompts
python scripts/run_layer_sweep.py                 # Exp 6:  edit every layer; test the tracing rule
python scripts/run_layer_sweep.py --plot-only     #         redraw its figure from the saved CSVs
python scripts/run_selectivity.py                 # Exp 7:  per-behavior ablation + head selectivity
python scripts/run_checkpoints.py                 # Exp 8:  behavior vs. pretraining step
python scripts/run_edit_models.py                 # Exp 9:  the same edit in all three models
python scripts/run_edit_penalty.py                # Exp 10: why those edit settings do not transfer
```

As a library:

```python
from models import load_config, load_model, evaluate_edit
from src.data_loader import load_processed_edit_targets

cfg = load_config()
model = load_model(cfg)
df, scores = evaluate_edit(model, load_processed_edit_targets(cfg)[0], cfg)
print(scores)   # {'efficacy': ..., 'generalization': ..., 'specificity': ..., 'fluency_before': ...}
```

Notebooks: [`notebooks/demo_pipeline.ipynb`](notebooks/demo_pipeline.ipynb) walks the pipeline one
stage at a time; [`experiments/`](experiments) holds the per-experiment notebooks.

## Reproducing the committed outputs

```bash
python src/data_loader.py     # rebuild data/processed/ (deterministic — should produce no diff)
python src/model_runner.py    # rewrite outputs/
python src/train.py           # rewrite outputs/edits/ and the loss-curve figure
pytest -q                     # 30 tests; the model-dependent ones skip without torch
```

Runs are deterministic given the same seed, model, and package versions: decoding is greedy,
every random draw is seeded through `utils.helpers.set_seed`, and the exact versions used for the
committed run are recorded under `versions` in
[`outputs/run_metadata.json`](outputs/run_metadata.json). Re-running on the same machine
reproduces the committed numbers exactly.

**Across machines**, verified by running the same pipeline natively (Windows, Python 3.13) and in
the container (Linux, Python 3.11, same package versions): all 10 generations, all top-1
predictions, and every fluency score are identical, and log-probabilities agree to ~1e-3. That
float-level noise is invisible in the behavioral pass, but the gradient-fitted edit amplifies it —
two of the binary edit metrics flipped between the two environments (see limitation 6 below). Use
the container when byte-identical edit results matter.

## Results at a glance

Six further experiments were added for the final milestone; every table and figure they produce is
committed under [`outputs/study/`](outputs/study) with the numbers written up in
[`outputs/study/README.md`](outputs/study/README.md).

| Question | Answer | Evidence |
|---|---|---|
| Does the behavior hold across models and scales? | Agreement is saturated everywhere; **factual recall is what scales** (+2.07 → +4.41, 0.83 → 1.00 accuracy from 160m to 410m). GPT-2 beats pythia-160m overall. | Exp 5 |
| Does the edit layer have to be the traced one? | No — a fact can be rewritten from **most early layers** (Everest: 9 of 12), never from the last three. The multi-layer tracing window, proposed here as the fix, makes selection **worse** (0.93 → 0.73 efficacy), so it is implemented but not the default. | Exp 6 |
| Is each behavior carried by its own components? | Agreement has a **dedicated head, L6.H4 (selectivity 1.00)**; induction (0.75) and factual recall (0.52) are diffuse. MLP 0 dominates all three. | Exp 7 |
| When do the behaviors appear during pretraining? | Different times, different shapes: agreement switches on at step ~1 000, induction climbs from ~1 000, factual recall is *actively wrong* at steps 128–512 first. **The final checkpoint is not the best one** for two of three behaviors. | Exp 8 |
| Why did the edit fail on GPT-2? | Not a GPT-2 property — a **hyperparameter–scale mismatch**. Its residual norm at the edited layer is 75 vs. Pythia's 13–19, so the absolute `edit_kl_weight` penalty crushes δ to 2.76 instead of the ~100 needed. At `edit_kl_weight=0.01` GPT-2 edits succeed. | Exp 10 |
| Is "generalization = 0.00" a property of the method? | No — it was suppressed by that same penalty. Pooled generalization rises 0.167 → 0.611 as the penalty falls, while top-1 preservation falls 0.944 → 0.000: a real trade-off the single committed configuration had hidden. | Exp 10 |

## Preliminary results

From the committed run (pythia-160m, greedy, seed 0 — full detail in
[`outputs/README.md`](outputs/README.md)):

* **Behavioral prompts.** The model prefers the correct continuation on **9 of 10** prompts,
  mean logit difference **+3.14**. By behavior: agreement **+4.20** (4/4), factual recall
  **+2.90** (3/3), induction **+1.96** (2/3). Agreement holds up even with a plural attractor
  between subject and verb. Factual recall is correct but rarely top-1 — the model ranks
  ` Madrid` behind generic continuations for *"The capital of Spain is"*.
  Extending to all 48 prompts with `python scripts/run_baseline.py` gives the same picture at
  scale — **43/48** correct: agreement 20/20 (+4.62), factual recall 10/12 (+2.07), induction
  13/16 (+1.73). That output lands in the git-ignored `results/`, so the committed 10-sample
  batch stays the reviewable artifact.
* **Edits.** 2 of 3 rank-one edits change the generated text as intended (*Mount Everest → Canada*,
  *Mona Lisa → Rembrandt*, efficacy 1.00) while leaving neighbouring facts' top-1 predictions
  unchanged (0.83 mean). **Generalization to paraphrases is 0.00 across all targets** — the
  rewritten fact does not survive rewording. Fluency is essentially unaffected (4.42 → 4.52).
* **Text quality.** Generations are locally fluent but repetitive at 160M parameters; several
  continuations loop the prompt back on itself, which the fluency (n-gram entropy) score picks up.

## Known issues and limitations

1. **Causal tracing picks an unreliable edit layer — but the layer barely matters.** For *The
   Eiffel Tower* the traced peak (layer 6) yields efficacy 0.00 while layer 5 succeeds. Sweeping
   every layer (Exp 6) shows this is an isolated dead layer, not a depth effect: 8 of 12 layers
   work for that target. The multi-layer tracing window planned as the fix was implemented
   (`best_edit_layer(..., window=w)`) and measured — it selects *worse* layers than the raw
   argmax (efficacy 0.93 → 0.73), so `edit_layer_window` stays 1. Over 5 tracing seeds the raw
   rule lands on a working layer 14 times out of 15.
2. **Generalization was suppressed by a hyperparameter, not by the method.** The committed run's
   0.00 generalization comes from `edit_kl_weight = 0.0625`; pooled over three models it rises to
   0.611 as that penalty goes to zero, at the cost of all neighbourhood preservation (Exp 10).
   The identity-covariance simplification (used in place of ROME's corpus-estimated second-moment
   matrix C, see [`models/editing.py`](models/editing.py)) remains a real limitation on top of
   that, but it is no longer the leading explanation.
3. **`specificity` under-reports.** It checks whether a neighbouring fact's original answer still
   appears in the generation, so it scores 0 when the base model never knew that fact.
   `specificity_pred_preserved` (top-1 unchanged before vs. after) is the fairer companion metric
   and is reported alongside it.
4. **Induction answers and the leading space.** Answers are tokenized as `" answer"`, but in
   induction prompts like `applebanana apple` the true continuation is `ban`/`banana` with no
   leading space. The comparison stays fair (both alternatives are scored the same way) but the
   absolute induction numbers are pessimistic.
5. **Small scale.** 48 prompts and 3 edit targets — against the 300–500 prompts and 20–40 targets
   the proposal projected. Enough to demonstrate the system end to end and to support the
   qualitative claims above, not enough for statistical ones: a single edit flipping changes a
   3-target mean by 0.33. Pythia-410m, the GPT-2 control, and the checkpoint analysis (RQ4) *have*
   now been run (Exp 5, 8, 9); scaling the prompt set is the remaining gap.
6. **Edit metrics are sensitive to the floating-point environment.** Running the identical
   pipeline in the Docker container reproduced every generation and top-1 prediction of the native
   Windows run, but two binary edit scores flipped (*Mona Lisa* generalization 0.00 → 0.50,
   *Eiffel Tower* top-1-preserved 0.50 → 0.00). Log-probabilities differ by ~1e-3 between the two
   platforms, and 25 Adam steps on that starting point can land either side of a threshold.
   Reporting rates over 3 targets makes single flips look large; more targets and averaging over
   seeds are the fix.

## Repository structure

```
black-box-lm/
├── src/                    # pipeline entry points
│   ├── data_loader.py      #   raw suites -> data/processed/ (+ loaders for everything downstream)
│   ├── model_runner.py     #   data/processed/ -> outputs/    ← single-command run
│   └── train.py            #   fit + save the rank-one edit (the only optimization here)
├── models/                 # reusable research package
│   ├── config.py           #   typed config + YAML loader
│   ├── loader.py           #   load HookedTransformer (no training)
│   ├── interpret.py        #   logit diff, ablation, causal tracing, logit lens   (Exp 1–3)
│   ├── editing.py          #   v* optimization + rank-one W_out edit               (Exp 4)
│   ├── generation.py       #   reproducible decoding
│   ├── metrics.py          #   efficacy / generalization / specificity / fluency
│   └── utils.py            #   token & probability helpers
├── utils/                  # shared plumbing
│   ├── helpers.py          #   paths, logging, seeding, JSON/JSONL I/O, formatting
│   └── visualization.py    #   headless-safe chart helpers
├── data/
│   ├── prompts/            #   raw *.jsonl minimal-pair suites
│   ├── edit_targets.json   #   raw ROME edit targets
│   ├── processed/          #   cleaned + split dataset with a manifest (committed)
│   └── loaders.py  preprocess.py  build_suites.py
├── configs/model_config.yaml   # every hyperparameter, one file
├── outputs/                # generated samples, tables, figures, run record (committed)
│   └── study/              #   the six extended experiments behind the report (committed)
├── notebooks/              # demo_pipeline.ipynb — the pipeline stage by stage
├── experiments/            # per-experiment notebooks (01_baseline, 04_editing_and_generation)
├── scripts/                # headless per-experiment entry points
├── docs/                   # proposal, literature review, benchmarking, model docs
│   └── report/             #   technical report source + one-command DOCX/PDF build
├── tests/                  # pytest — fast suite runs without torch
├── Dockerfile  .dockerignore
├── requirements.txt  environment.yml  pyproject.toml
```

## Configuration

Every hyperparameter lives in [`configs/model_config.yaml`](configs/model_config.yaml) and is
loaded into the typed `ProjectConfig` dataclass in [`models/config.py`](models/config.py). Any
field can be overridden on the command line (`--model`, `--device`, `--seed`, `--n-samples`,
`--max-new-tokens`, `--edit-layer`). Key knobs: `model_name`, `n_samples`, `edit_lr`, `edit_steps`,
`edit_kl_weight`, `noise_scale`, `gen_max_new_tokens`.

## Tests and CI

```bash
pytest -q
```

30 tests. The fast suite — config, dataset build/validation determinism, batch selection, report
rendering, utility I/O — needs only `numpy pandas pyyaml pytest`, so
[CI](.github/workflows/ci.yml) runs it on every push (Python 3.10 and 3.11) and also verifies that
both the raw suites and `data/processed/` are byte-for-byte reproducible. The model-dependent
tests skip automatically when torch is absent.

## How this maps to the milestone

| Required component | Where |
|---|---|
| Single-command pipeline (`python src/model_runner.py`) | [`src/model_runner.py`](src/model_runner.py) |
| Modular source code | [`src/`](src) (entry points) · [`models/`](models) (research code) |
| Data loading & preprocessing | [`src/data_loader.py`](src/data_loader.py) · [`data/`](data) · [`data/processed/`](data/processed) |
| Utility functions | [`utils/helpers.py`](utils/helpers.py) · [`utils/visualization.py`](utils/visualization.py) |
| Configuration files | [`configs/model_config.yaml`](configs/model_config.yaml) |
| Generated output samples | [`outputs/`](outputs) — `samples.txt`, CSV tables, figures, `run_metadata.json` |
| Training / fine-tuning (here: fitting the edit) | [`src/train.py`](src/train.py) · [`models/editing.py`](models/editing.py) |
| Evaluation & metrics | [`models/metrics.py`](models/metrics.py) |
| Documentation & reproducibility | this README · [`outputs/README.md`](outputs/README.md) · [`docs/`](docs) · Dockerfile · tests |
| Container | [`Dockerfile`](Dockerfile) |
| Technical report (8–10 pages) | [`docs/report/technical_report.md`](docs/report/technical_report.md) → `Group25_TechnicalReport.pdf` |
| Presentation slides | [`docs/report/presentation.md`](docs/report/presentation.md) → `Group25_Presentation.pptx` |
| Presentation video (≤10 min) | recorded from the deck → `Group25_Presentation.mpeg` (`--video`) |
| Submission archive | `python scripts/package_submission.py` → `Group25_FinalProject.zip` |

## Documentation

- [Technical report](docs/report/technical_report.md) and
  [presentation](docs/report/presentation.md) — the written study and the talk; build the
  submitted PDF/PPTX with `python docs/report/build_report.py` and `build_slides.py`
  (see [docs/report/README.md](docs/report/README.md))
- [Extended study results](outputs/study/README.md) — every number the report cites
- [Generated samples and results](outputs/README.md)
- [Proposal (summary)](docs/proposal.md) · full DOCX in [`docs/`](docs)
- [Methods & literature review](docs/methods_and_literature_review.md)
- [Benchmarking](docs/benchmarking.md)
- [Model documentation](docs/model_documentation.md)
- [Dataset](data/README.md) · [Experiments](experiments/README.md)

## References

Vaswani et al. (2017); Radford et al. (2019); Elhage et al. (2021); Olsson et al. (2022); Wang et
al. (2022); Meng et al. (2022, ROME; 2023, MEMIT); Biderman et al. (2023, Pythia); Nanda & Bloom
(2022, TransformerLens). Full list in [`docs/methods_and_literature_review.md`](docs/methods_and_literature_review.md).

## License

MIT — see [LICENSE](LICENSE).
