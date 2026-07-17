"""Smoke tests.

The lightweight tests (config, data loading, splits) run anywhere. The model-dependent
tests are skipped automatically when torch / transformer_lens are not installed, so CI can
run the fast suite without downloading a model.
"""
import pytest

from models import load_config, ProjectConfig
from data import load_behavior_suite, load_edit_targets, SUITES
from data.preprocess import make_splits


# --------------------------------------------------------------------------- config
def test_config_defaults():
    cfg = load_config()
    assert isinstance(cfg, ProjectConfig)
    assert cfg.model_name == "pythia-160m"
    assert cfg.resolved_device() in {"cpu", "cuda"}


def test_config_override():
    cfg = load_config(model_name="gpt2", edit_steps=3)
    assert cfg.model_name == "gpt2"
    assert cfg.edit_steps == 3


# --------------------------------------------------------------------------- data
@pytest.mark.parametrize("name", list(SUITES))
def test_suites_load_and_have_answers(name):
    recs = load_behavior_suite(name)
    assert len(recs) > 0
    for r in recs:
        assert r["prompt"] and r["correct"]


def test_contrast_pairs_are_symmetric():
    recs = load_behavior_suite("agreement")
    by_id = {r["id"]: r for r in recs}
    for r in recs:
        cid = r.get("contrast_id")
        if cid:
            assert cid in by_id
            assert by_id[cid]["contrast_id"] == r["id"]      # symmetric
            assert by_id[cid]["correct"] == r["incorrect"]   # answer flips


def test_edit_targets_shape():
    targets = load_edit_targets()
    assert len(targets) >= 1
    for t in targets:
        for key in ("subject", "prompt", "true", "new", "paraphrases", "neighborhood"):
            assert key in t
        assert t["subject"] in t["prompt"]


# --------------------------------------------------------------------------- splits
def test_splits_no_pair_leakage_and_cover():
    recs = load_behavior_suite("agreement")
    splits = make_splits(recs, seed=0)
    # every record appears exactly once across splits
    ids = [r["id"] for part in splits.values() for r in part]
    assert sorted(ids) == sorted(r["id"] for r in recs)
    # a contrast pair never straddles two splits
    where = {r["id"]: name for name, part in splits.items() for r in part}
    for r in recs:
        cid = r.get("contrast_id")
        if cid:
            assert where[r["id"]] == where[cid]


# --------------------------------------------------------------------------- model (optional)
# These are skipped automatically when torch / transformer_lens are absent, so the fast
# suite above still runs everywhere.
@pytest.fixture(scope="module")
def model():
    pytest.importorskip("torch", reason="torch not installed")
    pytest.importorskip("transformer_lens", reason="transformer_lens not installed")
    from models import load_model
    return load_model(load_config(model_name="pythia-160m", device="cpu"))


def test_logit_diff_prefers_correct(model):
    from models import logit_diff
    # a fact the model reliably knows
    ld = logit_diff(model, "The capital of France is", " Paris", " London")
    assert ld > 0


def test_edit_changes_generation(model):
    from models import evaluate_edit
    cfg = load_config(model_name="pythia-160m", device="cpu", edit_steps=20)
    df, scores = evaluate_edit(model, load_edit_targets()[0], cfg)
    assert {"efficacy", "generalization", "specificity"} <= set(scores)
    assert 0.0 <= scores["efficacy"] <= 1.0
