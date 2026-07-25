"""Pipeline entry points.

`src/` holds the runnable stages of the milestone pipeline; the reusable research code
lives in `models/` (model, interpretability, editing, metrics), `data/` (raw suites), and
`utils/` (shared plumbing).

    src/data_loader.py    raw prompt suites  -> data/processed/
    src/model_runner.py   data/processed/    -> outputs/            (single-command run)
    src/train.py          fit + save the rank-one edit for one factual target
"""
