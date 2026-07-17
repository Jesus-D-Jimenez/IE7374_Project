"""Load and explore the behavioral prompt suites and edit targets."""
from __future__ import annotations
import json
import os
from typing import Optional

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(HERE, "prompts")

# Friendly names -> files on disk.
SUITES = {
    "agreement": "subject_verb_agreement.jsonl",
    "factual_recall": "factual_recall.jsonl",
    "induction": "induction.jsonl",
}


def load_behavior_suite(name: str, data_dir: Optional[str] = None) -> list[dict]:
    """Return the list of prompt records for a behavior suite (see SUITES)."""
    if name not in SUITES:
        raise KeyError(f"Unknown suite '{name}'. Choose from {list(SUITES)}.")
    prompts_dir = os.path.join(data_dir, "prompts") if data_dir else PROMPTS_DIR
    path = os.path.join(prompts_dir, SUITES[name])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Generate the suites first: python data/build_suites.py"
        )
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_all_suites(data_dir: Optional[str] = None) -> dict[str, list[dict]]:
    return {name: load_behavior_suite(name, data_dir) for name in SUITES}


def load_edit_targets(data_dir: Optional[str] = None) -> list[dict]:
    """Return the ROME edit targets (subject/prompt/true/new + eval prompt sets)."""
    base = data_dir if data_dir else HERE
    path = os.path.join(base, "edit_targets.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Generate it first: python data/build_suites.py"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def explore_suite(name_or_records) -> pd.DataFrame:
    """Quick EDA: return a DataFrame and print a short summary of a suite."""
    records = (
        load_behavior_suite(name_or_records)
        if isinstance(name_or_records, str)
        else name_or_records
    )
    df = pd.DataFrame(records)
    print(f"{len(df)} prompts | types: {df['type'].value_counts().to_dict()}")
    print(f"mean prompt length (chars): {df['prompt'].str.len().mean():.1f}")
    n_paired = df["contrast_id"].notna().sum() if "contrast_id" in df else 0
    print(f"prompts with a matched contrast: {n_paired}")
    return df


if __name__ == "__main__":
    for suite in SUITES:
        print(f"\n=== {suite} ===")
        explore_suite(suite)
    print(f"\n=== edit targets: {len(load_edit_targets())} ===")
