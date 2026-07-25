"""Dataset stage of the pipeline: raw prompt suites -> cleaned, split, versioned dataset.

Responsibilities
----------------
1. **Build** `data/processed/` from the raw suites in `data/prompts/` and
   `data/edit_targets.json` (validate, normalize, split, and write a manifest).
2. **Load** that processed dataset for the inference stage (`src/model_runner.py`),
   the notebooks, and the tests.

The build is deterministic — same inputs and seed produce byte-identical files — so the
processed dataset is committed and CI can verify it is reproducible.

Run directly to (re)build the dataset:

    python src/data_loader.py            # build (or rebuild) data/processed/
    python src/data_loader.py --check    # verify the committed dataset is up to date
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

# Make the repository root importable when this file is run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loaders import SUITES, load_behavior_suite, load_edit_targets   # noqa: E402
from data.preprocess import make_splits                                   # noqa: E402
from models.config import ProjectConfig, load_config                      # noqa: E402
from utils.helpers import (                                               # noqa: E402
    file_sha256, get_logger, read_json, read_jsonl, relative_to_repo,
    write_json, write_jsonl,
)

# File names inside `processed_dir`. Kept as constants so every consumer agrees.
PROMPTS_FILE = "prompts.jsonl"
EDIT_TARGETS_FILE = "edit_targets.json"
MANIFEST_FILE = "manifest.json"

# A processed prompt record always carries these fields.
PROMPT_FIELDS = ("id", "suite", "type", "prompt", "correct", "incorrect",
                 "contrast_id", "split", "n_chars", "n_tokens_est")

# An edit target always carries these fields.
EDIT_TARGET_FIELDS = ("subject", "prompt", "true", "new", "paraphrases", "neighborhood")


# --------------------------------------------------------------------------- #
# Normalization / validation
# --------------------------------------------------------------------------- #
def normalize_prompt_record(record: dict, suite: str) -> dict:
    """Clean one raw prompt record and return it in the canonical processed schema.

    Cleaning rules:
      * collapse stray whitespace in the prompt and strip trailing spaces;
      * store answers without surrounding whitespace (the tokenizer helpers in
        `models/utils.py` re-attach the leading space the BPE vocabulary expects);
      * keep `contrast_id` as None rather than an empty string.

    Raises ValueError if a required field is missing, so a malformed suite fails loudly
    at build time instead of silently producing bad inference samples.
    """
    for key in ("id", "prompt", "correct"):
        if not record.get(key):
            raise ValueError(f"[{suite}] record is missing '{key}': {record!r}")

    prompt = " ".join(str(record["prompt"]).split())
    correct = str(record["correct"]).strip()
    incorrect = str(record.get("incorrect") or "").strip() or None

    return {
        "id": record["id"],
        "suite": suite,
        "type": record.get("type", suite),
        "prompt": prompt,
        "correct": correct,
        "incorrect": incorrect,
        "contrast_id": record.get("contrast_id") or None,
        "split": None,                       # filled in by build_processed_dataset
        "n_chars": len(prompt),
        "n_tokens_est": len(prompt.split()),  # whitespace estimate; exact counts need the tokenizer
    }


def validate_edit_target(target: dict, index: int) -> dict:
    """Check one ROME edit target and return it unchanged.

    The subject must literally appear in the prompt, otherwise `models.utils.subject_span`
    cannot locate the tokens to edit.
    """
    for key in EDIT_TARGET_FIELDS:
        if key not in target:
            raise ValueError(f"edit target #{index} is missing '{key}': {target!r}")
    if target["subject"] not in target["prompt"]:
        raise ValueError(
            f"edit target #{index}: subject {target['subject']!r} does not occur in "
            f"prompt {target['prompt']!r}"
        )
    if target["true"].strip().lower() == target["new"].strip().lower():
        raise ValueError(f"edit target #{index}: 'true' and 'new' objects are identical")
    return target


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build_processed_dataset(config: Optional[ProjectConfig] = None, logger=None) -> dict:
    """Build `data/processed/` from the raw suites and return the manifest dict."""
    cfg = config or load_config()
    log = logger or get_logger("data_loader")

    raw_dir = cfg.abs_path(cfg.data_dir)
    out_dir = cfg.abs_path(cfg.processed_dir)

    records: list[dict] = []
    sources: list[dict] = []
    by_suite: dict[str, int] = {}

    for suite, filename in SUITES.items():
        raw = load_behavior_suite(suite, raw_dir)
        cleaned = [normalize_prompt_record(r, suite) for r in raw]

        # Deterministic, pair-aware split (contrast pairs stay together).
        splits = make_splits(cleaned, ratios=tuple(cfg.split_ratios), seed=cfg.seed)
        split_of = {r["id"]: name for name, part in splits.items() for r in part}
        for r in cleaned:
            r["split"] = split_of[r["id"]]

        records.extend(cleaned)
        by_suite[suite] = len(cleaned)
        src_path = os.path.join(raw_dir, "prompts", filename)
        sources.append({
            "path": relative_to_repo(src_path),
            "sha256": file_sha256(src_path),
            "records": len(cleaned),
        })
        log.info("%-16s %3d prompts -> %s", suite, len(cleaned),
                 {k: len(v) for k, v in splits.items()})

    duplicates = _duplicate_ids(records)
    if duplicates:
        raise ValueError(f"duplicate prompt ids across suites: {sorted(duplicates)}")

    targets_path = os.path.join(raw_dir, "edit_targets.json")
    targets = [validate_edit_target(t, i) for i, t in enumerate(load_edit_targets(raw_dir))]
    sources.append({
        "path": relative_to_repo(targets_path),
        "sha256": file_sha256(targets_path),
        "records": len(targets),
    })

    # Sort by id so the written file never depends on dictionary or filesystem order.
    records.sort(key=lambda r: (r["suite"], r["id"]))

    prompts_out = write_jsonl(os.path.join(out_dir, PROMPTS_FILE), records)
    targets_out = write_json(os.path.join(out_dir, EDIT_TARGETS_FILE), targets)

    by_split: dict[str, int] = {}
    for r in records:
        by_split[r["split"]] = by_split.get(r["split"], 0) + 1

    manifest = {
        "generated_by": "src/data_loader.py",
        "description": (
            "Behavioral minimal-pair prompt suites and ROME edit targets, normalized and "
            "split deterministically. No timestamps are recorded so the build is "
            "byte-for-byte reproducible."
        ),
        "config": {"seed": cfg.seed, "split_ratios": list(cfg.split_ratios)},
        "sources": sources,
        "n_prompts": len(records),
        "n_edit_targets": len(targets),
        "by_suite": by_suite,
        "by_split": by_split,
        "files": {
            "prompts": relative_to_repo(prompts_out),
            "edit_targets": relative_to_repo(targets_out),
        },
        "schema": {"prompts": list(PROMPT_FIELDS), "edit_targets": list(EDIT_TARGET_FIELDS)},
    }
    manifest_out = write_json(os.path.join(out_dir, MANIFEST_FILE), manifest)

    log.info("wrote %d prompts + %d edit targets -> %s",
             len(records), len(targets), relative_to_repo(out_dir))
    log.info("manifest -> %s", relative_to_repo(manifest_out))
    return manifest


def _duplicate_ids(records: list[dict]) -> set[str]:
    seen, dupes = set(), set()
    for r in records:
        (dupes if r["id"] in seen else seen).add(r["id"])
    return dupes


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def processed_paths(config: Optional[ProjectConfig] = None) -> dict[str, str]:
    """Absolute paths of the three processed-dataset files."""
    cfg = config or load_config()
    out_dir = cfg.abs_path(cfg.processed_dir)
    return {
        "prompts": os.path.join(out_dir, PROMPTS_FILE),
        "edit_targets": os.path.join(out_dir, EDIT_TARGETS_FILE),
        "manifest": os.path.join(out_dir, MANIFEST_FILE),
    }


def _ensure_built(cfg: ProjectConfig, auto_build: bool, logger=None) -> dict[str, str]:
    paths = processed_paths(cfg)
    if all(os.path.exists(p) for p in paths.values()):
        return paths
    if not auto_build:
        missing = [relative_to_repo(p) for p in paths.values() if not os.path.exists(p)]
        raise FileNotFoundError(
            f"processed dataset is incomplete (missing: {missing}). "
            f"Build it with: python src/data_loader.py"
        )
    (logger or get_logger("data_loader")).info("processed dataset missing — building it now")
    build_processed_dataset(cfg, logger)
    return paths


def load_processed_prompts(config: Optional[ProjectConfig] = None, suite: Optional[str] = None,
                           split: Optional[str] = None, auto_build: bool = True,
                           logger=None) -> list[dict]:
    """Return processed prompt records, optionally filtered by suite and/or split."""
    cfg = config or load_config()
    paths = _ensure_built(cfg, auto_build, logger)

    if suite is not None and suite not in SUITES:
        raise KeyError(f"unknown suite {suite!r}; choose from {list(SUITES)}")

    records = read_jsonl(paths["prompts"])
    if suite:
        records = [r for r in records if r["suite"] == suite]
    if split:
        records = [r for r in records if r["split"] == split]
    if not records:
        raise ValueError(f"no records match suite={suite!r}, split={split!r}")
    return records


def load_processed_edit_targets(config: Optional[ProjectConfig] = None,
                                auto_build: bool = True, logger=None) -> list[dict]:
    """Return the validated ROME edit targets from the processed dataset."""
    cfg = config or load_config()
    paths = _ensure_built(cfg, auto_build, logger)
    return read_json(paths["edit_targets"])


def load_manifest(config: Optional[ProjectConfig] = None, auto_build: bool = True,
                  logger=None) -> dict:
    """Return the processed-dataset manifest (counts, checksums, split sizes)."""
    cfg = config or load_config()
    paths = _ensure_built(cfg, auto_build, logger)
    return read_json(paths["manifest"])


# --------------------------------------------------------------------------- #
# Inference-batch selection
# --------------------------------------------------------------------------- #
def select_inference_samples(records: list[dict], n_samples: int = 10,
                             per_suite: Optional[int] = None, seed: int = 0) -> list[dict]:
    """Pick a small, behavior-balanced, deterministic batch of prompts for inference.

    Round-robins across the behaviors so a 10-prompt run always covers agreement,
    factual recall, and induction rather than 10 near-identical items. Selection depends
    only on `seed`, so reruns of the pipeline produce the same batch.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    import random
    rng = random.Random(seed)

    by_suite: dict[str, list[dict]] = {}
    for r in records:
        by_suite.setdefault(r["suite"], []).append(r)

    # Shuffle within each behavior, then take a bounded slice of each.
    pools = {}
    for suite in sorted(by_suite):
        pool = sorted(by_suite[suite], key=lambda r: r["id"])
        rng.shuffle(pool)
        pools[suite] = pool[:per_suite] if per_suite else pool

    chosen: list[dict] = []
    while len(chosen) < n_samples and any(pools.values()):
        for suite in sorted(pools):
            if pools[suite] and len(chosen) < n_samples:
                chosen.append(pools[suite].pop(0))

    if len(chosen) < n_samples:
        raise ValueError(
            f"requested {n_samples} samples but only {len(chosen)} available "
            f"(per_suite={per_suite})"
        )
    return chosen


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build or verify the processed dataset.")
    ap.add_argument("--config", default=None, help="path to a YAML config file")
    ap.add_argument("--seed", type=int, default=None, help="override the split seed")
    ap.add_argument("--check", action="store_true",
                    help="rebuild into memory and fail if the committed files differ")
    args = ap.parse_args(argv)

    log = get_logger("data_loader")
    cfg = load_config(args.config, seed=args.seed)

    if args.check:
        paths = processed_paths(cfg)
        before = {k: (read_jsonl(p) if p.endswith(".jsonl") else read_json(p))
                  for k, p in paths.items() if os.path.exists(p)}
        if len(before) != len(paths):
            log.error("processed dataset is missing files — run: python src/data_loader.py")
            return 1
        build_processed_dataset(cfg, log)
        after = {k: (read_jsonl(p) if p.endswith(".jsonl") else read_json(p))
                 for k, p in paths.items()}
        drifted = [k for k in paths if before[k] != after[k]]
        if drifted:
            log.error("processed dataset is stale: %s changed on rebuild", drifted)
            return 1
        log.info("processed dataset is up to date (%d prompts)", len(after["prompts"]))
        return 0

    manifest = build_processed_dataset(cfg, log)
    log.info("by suite: %s | by split: %s", manifest["by_suite"], manifest["by_split"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
