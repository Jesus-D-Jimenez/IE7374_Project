"""Experiment 1 - baseline behavioral logit differences for every prompt suite.

Usage:
    python scripts/run_baseline.py                      # pythia-160m, all suites
    python scripts/run_baseline.py --model pythia-410m
    python scripts/run_baseline.py --suite factual_recall
"""
from __future__ import annotations
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import load_config, load_model, suite_logit_diffs   # noqa: E402
from data import load_behavior_suite, SUITES                    # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Baseline logit differences.")
    ap.add_argument("--model", default=None, help="override model_name")
    ap.add_argument("--device", default=None, help="cpu | cuda")
    ap.add_argument("--suite", default="all", choices=["all", *SUITES])
    args = ap.parse_args()

    cfg = load_config(model_name=args.model, device=args.device)
    model = load_model(cfg)

    names = list(SUITES) if args.suite == "all" else [args.suite]
    frames = []
    for name in names:
        recs = load_behavior_suite(name, cfg.abs_path(cfg.data_dir))
        df = pd.DataFrame(suite_logit_diffs(model, recs))
        df["suite"] = name
        frames.append(df)
        print(f"{name:16s} n={len(df):3d}  mean logit diff = {df['logit_diff'].mean():+.3f}  "
              f"(accuracy = {(df['logit_diff'] > 0).mean():.2%})")

    out = pd.concat(frames, ignore_index=True)
    results_dir = cfg.abs_path(cfg.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "baseline_logit_diffs.csv")
    out.to_csv(path, index=False)
    print("saved ->", path)


if __name__ == "__main__":
    main()
