"""Tests for the Phase 2 pipeline layer: utils, the data loader, and the runner's plumbing.

Everything here runs without torch — the pieces that need a model are exercised by the
optional model tests in `test_smoke.py`. Keeping the fast suite model-free is what lets
CI check the pipeline on every push.
"""
from __future__ import annotations

import json
import os

import pytest

from models.config import load_config
from src.data_loader import (
    build_processed_dataset, load_manifest, load_processed_edit_targets,
    load_processed_prompts, normalize_prompt_record, processed_paths,
    select_inference_samples, validate_edit_target,
)
from utils.helpers import (
    banner, one_line, read_json, read_jsonl, repo_path, set_seed,
    write_json, write_jsonl, write_text,
)


# --------------------------------------------------------------------------- utils
def test_repo_path_points_at_the_repository():
    assert os.path.isfile(repo_path("configs", "model_config.yaml"))
    assert os.path.isdir(repo_path("src"))


def test_json_roundtrip(tmp_path):
    path = str(tmp_path / "nested" / "obj.json")     # parent directory is created for us
    write_json(path, {"a": [1, 2], "b": "café"})
    assert read_json(path) == {"a": [1, 2], "b": "café"}


def test_jsonl_roundtrip_uses_lf_newlines(tmp_path):
    path = str(tmp_path / "rows.jsonl")
    rows = [{"id": "a"}, {"id": "b"}]
    write_jsonl(path, rows)
    assert read_jsonl(path) == rows
    with open(path, "rb") as f:                      # committed files must not gain CRLF
        assert b"\r\n" not in f.read()


def test_write_text_and_formatting_helpers(tmp_path):
    path = str(tmp_path / "report.txt")
    write_text(path, "hello\n")
    assert open(path, encoding="utf-8").read() == "hello\n"

    assert one_line("a\n  b\tc") == "a b c"
    assert one_line("abcdef", limit=4).endswith("…") and len(one_line("abcdef", limit=4)) == 4
    assert banner("T", width=5, char="-").splitlines()[0] == "-----"


def test_set_seed_is_reproducible():
    import random
    set_seed(7)
    first = [random.random() for _ in range(3)]
    set_seed(7)
    assert [random.random() for _ in range(3)] == first


# --------------------------------------------------------------------------- config
def test_config_exposes_pipeline_paths():
    cfg = load_config()
    assert cfg.processed_dir == "data/processed"
    assert cfg.outputs_dir == "outputs"
    assert cfg.n_samples >= 5                        # milestone asks for 5-10 samples
    assert abs(sum(cfg.split_ratios) - 1.0) < 1e-9


# --------------------------------------------------------------------------- normalization
def test_normalize_collapses_whitespace_and_strips_answers():
    row = normalize_prompt_record(
        {"id": "x1", "type": "factual_recall", "prompt": "The  capital of\tFrance is ",
         "correct": " Paris", "incorrect": " London", "contrast_id": ""},
        suite="factual_recall")
    assert row["prompt"] == "The capital of France is"
    assert (row["correct"], row["incorrect"]) == ("Paris", "London")
    assert row["contrast_id"] is None                # "" normalizes to None
    assert row["suite"] == "factual_recall" and row["n_chars"] == len(row["prompt"])


@pytest.mark.parametrize("bad", [
    {"id": "", "prompt": "p", "correct": "c"},
    {"id": "x", "prompt": "", "correct": "c"},
    {"id": "x", "prompt": "p", "correct": ""},
])
def test_normalize_rejects_incomplete_records(bad):
    with pytest.raises(ValueError):
        normalize_prompt_record(bad, suite="agreement")


def test_validate_edit_target_requires_subject_in_prompt():
    good = {"subject": "Mount Everest", "prompt": "Mount Everest is located in the country of",
            "true": "Nepal", "new": "Canada", "paraphrases": [], "neighborhood": []}
    assert validate_edit_target(good, 0) is good

    with pytest.raises(ValueError):
        validate_edit_target({**good, "subject": "K2"}, 0)
    with pytest.raises(ValueError):
        validate_edit_target({**good, "new": "nepal"}, 0)      # identical to `true`
    with pytest.raises(ValueError):
        validate_edit_target({k: v for k, v in good.items() if k != "paraphrases"}, 0)


# --------------------------------------------------------------------------- processed dataset
def test_processed_dataset_is_committed_and_complete():
    manifest = load_manifest(auto_build=False)
    records = load_processed_prompts(auto_build=False)
    assert len(records) == manifest["n_prompts"] > 0
    assert set(manifest["by_suite"]) == {"agreement", "factual_recall", "induction"}
    for r in records:
        assert r["prompt"] and r["correct"] and r["split"] in {"train", "val", "test"}
        assert r["correct"] == r["correct"].strip()


def test_processed_targets_match_manifest():
    targets = load_processed_edit_targets(auto_build=False)
    assert len(targets) == load_manifest(auto_build=False)["n_edit_targets"]
    for i, t in enumerate(targets):
        validate_edit_target(t, i)


def test_rebuild_is_deterministic(tmp_path):
    """Rebuilding into a scratch directory reproduces the committed files byte-for-byte."""
    cfg = load_config(processed_dir=str(tmp_path))
    build_processed_dataset(cfg)
    fresh = processed_paths(cfg)
    committed = processed_paths(load_config())

    assert read_jsonl(fresh["prompts"]) == read_jsonl(committed["prompts"])
    assert read_json(fresh["edit_targets"]) == read_json(committed["edit_targets"])

    # Everything in the manifest matches except `files`, which records where this
    # particular build wrote its output.
    fresh_manifest, committed_manifest = read_json(fresh["manifest"]), read_json(committed["manifest"])
    assert fresh_manifest.pop("files") != committed_manifest.pop("files")
    assert fresh_manifest == committed_manifest


def test_filters_and_unknown_suite():
    factual = load_processed_prompts(suite="factual_recall", auto_build=False)
    assert factual and all(r["suite"] == "factual_recall" for r in factual)
    train = load_processed_prompts(split="train", auto_build=False)
    assert train and all(r["split"] == "train" for r in train)
    with pytest.raises(KeyError):
        load_processed_prompts(suite="not_a_suite", auto_build=False)


def test_missing_dataset_raises_a_helpful_error(tmp_path):
    cfg = load_config(processed_dir=str(tmp_path / "absent"))
    with pytest.raises(FileNotFoundError, match="python src/data_loader.py"):
        load_processed_prompts(cfg, auto_build=False)


# --------------------------------------------------------------------------- batch selection
def test_inference_batch_is_balanced_and_deterministic():
    records = load_processed_prompts(auto_build=False)
    batch = select_inference_samples(records, n_samples=10, per_suite=4, seed=0)

    assert len(batch) == 10
    assert len({r["id"] for r in batch}) == 10                    # no duplicates
    assert len({r["suite"] for r in batch}) == 3                  # every behavior covered
    assert batch == select_inference_samples(records, 10, 4, seed=0)
    assert batch != select_inference_samples(records, 10, 4, seed=1)


def test_inference_batch_rejects_impossible_requests():
    records = load_processed_prompts(auto_build=False)
    with pytest.raises(ValueError):
        select_inference_samples(records, n_samples=0)
    with pytest.raises(ValueError):
        select_inference_samples(records, n_samples=99, per_suite=1)


# --------------------------------------------------------------------------- report rendering
def test_samples_report_renders_without_a_model():
    """render_samples_text is pure formatting, so it is testable with fabricated rows."""
    import pandas as pd
    from src.model_runner import render_samples_text

    behavioral = pd.DataFrame([{
        "sample_id": "S01", "prompt_id": "fact_000", "suite": "factual_recall",
        "split": "test", "prompt": "The capital of France is", "correct": "Paris",
        "incorrect": "London", "top1_token": "Paris", "top1_logprob": -1.0,
        "top_k": json.dumps([["Paris", -1.0]]), "correct_logprob": -1.0,
        "incorrect_logprob": -4.0, "logit_diff": 3.0, "prefers_correct": True,
        "generation": "The capital of France is Paris.", "continuation": " Paris.",
        "fluency": 3.5, "status": "ok",
    }])
    meta = {
        "model": {"name": "pythia-160m", "n_params": 162334848, "n_layers": 12,
                  "device": "cpu"},
        "config": {"gen_do_sample": False, "gen_max_new_tokens": 30, "seed": 0},
        "run": {"utc": "2026-01-01 00:00:00", "command": "python src/model_runner.py"},
    }
    report = render_samples_text(behavioral, pd.DataFrame(), pd.DataFrame(), meta)

    assert "PART A" in report and "[S01]" in report
    assert "The capital of France is" in report
    assert "prefers the correct answer" in report
    assert "PART B" not in report                    # no edits were passed in


def test_committed_outputs_look_like_a_real_run():
    """The samples committed for review must match what the runner claims it produced."""
    outputs = repo_path("outputs")
    meta_path = os.path.join(outputs, "run_metadata.json")
    if not os.path.exists(meta_path):
        pytest.skip("outputs/ not generated yet — run: python src/model_runner.py")

    meta = read_json(meta_path)
    assert meta["results"]["n_behavioral_samples"] >= 5          # milestone minimum
    assert meta["model"]["n_params"] > 0
    samples = open(os.path.join(outputs, "samples.txt"), encoding="utf-8").read()
    assert samples.count("[S") >= meta["results"]["n_behavioral_samples"]
