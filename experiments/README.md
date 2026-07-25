# Experiments

Notebooks that demonstrate each stage of the project end to end. They import the reusable
code in [`../models`](../models) and [`../data`](../data); the same logic is available as
headless scripts in [`../scripts`](../scripts).

For the end-to-end pipeline itself — processed data → model → saved samples — see
[`../notebooks/demo_pipeline.ipynb`](../notebooks/demo_pipeline.ipynb) or run
`python src/model_runner.py`.

| Notebook | Maps to | What it shows |
|---|---|---|
| [`01_baseline.ipynb`](01_baseline.ipynb) | Experiments 1–3 | Baseline logit differences, the logit lens, and causal tracing to localize a fact. |
| [`04_editing_and_generation.ipynb`](04_editing_and_generation.ipynb) | Experiment 4 | The generative component: ROME-style rank-one edit → generate → evaluate (efficacy, generalization, specificity, fluency). This notebook is self-contained and can run without installing the package. |

## Running

```bash
# from the repo root, with the environment installed (see ../README.md)
jupyter lab            # then open experiments/01_baseline.ipynb
```

The first cell of each notebook installs `torch` / `transformer_lens` if missing and downloads
`pythia-160m` (~380 MB). Everything runs on CPU; a GPU only makes it faster.

Outputs (figures, CSVs) are written to [`../results`](../results), which is git-ignored except
for its placeholder.
