# Opening the Black Box

[![CI](https://github.com/Jesus-D-Jimenez/IE7374_Project/actions/workflows/ci.yml/badge.svg)](https://github.com/Jesus-D-Jimenez/IE7374_Project/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Mechanistic interpretability + ROME-style factual editing of small generative language models.**

Group 25 · Jesus D. Jimenez Ballestas · IE7374 (Generative AI)

This project looks *inside* a small open language model (EleutherAI **Pythia-160m / 410m**, with
**GPT-2 small** as a control), locates the internal components that drive specific behaviors, and
then **uses those components to edit the model and generate text under the edit** — so the work does
not just analyze the model, it produces and evaluates model outputs.

| Behavior studied | Generative payoff |
|---|---|
| subject–verb agreement · factual recall · induction | A rank-one **edit** of a localized fact, then **generation** scored on efficacy, generalization, specificity, and fluency (Meng et al. 2022). |

---

## Repository structure

```
black-box-lm/
├── config/            # default.yaml — all hyperparameters
├── data/              # prompt suites + edit targets, loaders, preprocessing, generator
│   ├── prompts/       # *.jsonl minimal-pair suites
│   ├── edit_targets.json
│   ├── loaders.py  preprocess.py  build_suites.py
├── models/            # reusable package
│   ├── config.py      # typed config + YAML loader
│   ├── loader.py      # load HookedTransformer (no training)
│   ├── interpret.py   # logit diff, ablation, causal tracing, logit lens   (Exp 1–3)
│   ├── editing.py     # v* optimization + rank-one W_out edit               (Exp 4)
│   ├── generation.py  # reproducible decoding
│   ├── metrics.py     # efficacy / generalization / specificity / fluency
│   └── utils.py       # token & probability helpers
├── scripts/           # headless entry points
│   ├── run_baseline.py  run_ablation.py  run_edit.py
├── experiments/       # notebooks (01_baseline, 04_editing_and_generation)
├── docs/              # proposal, literature review, benchmarking, model docs
├── tests/             # pytest smoke tests (fast suite runs without torch)
├── results/           # generated CSVs/figures (git-ignored)
├── requirements.txt  environment.yml  pyproject.toml
```

Maps to the milestone's recommended layout: `data/`, `models/`, `experiments/`, `docs/`,
dependency files.

## Setup

```bash
git clone <this-repo-url>
cd black-box-lm

# option A: pip
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                                          # makes `models` / `data` importable

# option B: conda
conda env create -f environment.yml
conda activate black-box-lm
pip install -e .
```

Python ≥ 3.10. Everything runs on **CPU**; a GPU only speeds things up. The first run downloads
`pythia-160m` (~380 MB) from Hugging Face.

## Quickstart

```bash
# 1. (re)generate the datasets — already committed, but reproducible
python data/build_suites.py

# 2. Experiment 1 — baseline behavioral logit differences
python scripts/run_baseline.py

# 3. Experiment 3a — attention-head ablation scan for one prompt
python scripts/run_ablation.py --prompt "The capital of Japan is" --correct " Tokyo" --incorrect " Beijing"

# 4. Experiment 4 — ROME-style edit -> generate -> evaluate (the generative component)
python scripts/run_edit.py                     # writes results/edit_eval_*.csv and a summary figure
```

Or open the notebooks:

```bash
jupyter lab   # experiments/01_baseline.ipynb  and  experiments/04_editing_and_generation.ipynb
```

### Library use

```python
from models import load_config, load_model, evaluate_edit
from data import load_edit_targets

cfg = load_config(model_name="pythia-160m")
model = load_model(cfg)
df, scores = evaluate_edit(model, load_edit_targets()[0], cfg)
print(scores)   # {'efficacy':..., 'generalization':..., 'specificity':..., 'fluency_before':..., 'fluency_after':...}
```

## Configuration

All hyperparameters live in [`config/default.yaml`](config/default.yaml) and can be overridden on
the command line (`--model`, `--device`, `--layer`). Key knobs: `model_name`, `edit_lr`,
`edit_steps`, `edit_kl_weight`, `noise_scale`, `gen_max_new_tokens`.

## Tests

```bash
pytest -q          # fast data/config tests run anywhere; model tests run once torch is installed
```

## How this maps to the milestone

| Required component | Where |
|---|---|
| Research & selection of methods | [`docs/methods_and_literature_review.md`](docs/methods_and_literature_review.md), [`docs/benchmarking.md`](docs/benchmarking.md) |
| Dataset preparation | [`data/`](data) — loaders, preprocessing, splits, generator, [`data/README.md`](data/README.md) |
| Model implementation | [`models/`](models) — modular, documented, config-driven |
| Training / fine-tuning (here: fitting the edit) | [`models/editing.py`](models/editing.py), [`docs/model_documentation.md`](docs/model_documentation.md) |
| Evaluation & metrics | [`models/metrics.py`](models/metrics.py), [`scripts/run_edit.py`](scripts/run_edit.py) |
| GitHub repo + docs + reproducibility | this README, `docs/`, `requirements.txt` / `environment.yml`, tests |

## Documentation

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
