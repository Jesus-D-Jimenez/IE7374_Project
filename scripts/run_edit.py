"""Experiment 4 - ROME-style edit -> generate -> evaluate, over all edit targets.

For each target this localizes the fact (causal tracing), applies a rank-one edit, generates
under the edit, and scores efficacy / generalization / specificity / fluency. Saves a
per-prompt table, a summary table, and a bar chart.

Usage:
    python scripts/run_edit.py
    python scripts/run_edit.py --model pythia-410m
    python scripts/run_edit.py --layer 5          # skip tracing, force an edit layer
"""
from __future__ import annotations
import argparse
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import load_config, load_model, evaluate_edit   # noqa: E402
from data import load_edit_targets                          # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="ROME-style edit and generation evaluation.")
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--layer", type=int, default=None, help="force edit layer (skip tracing)")
    args = ap.parse_args()

    cfg = load_config(model_name=args.model, device=args.device)
    model = load_model(cfg)
    targets = load_edit_targets(cfg.abs_path(cfg.data_dir))

    all_rows, summaries = [], []
    for t in targets:
        print(f"\n=== editing '{t['subject']}': {t['true']} -> {t['new']} ===")
        df, scores = evaluate_edit(model, t, cfg, layer=args.layer)
        df["subject"] = t["subject"]
        all_rows.append(df)
        summaries.append({"subject": t["subject"], **scores})
        print(f"  layer={scores['layer']}  efficacy={scores['efficacy']}  "
              f"generalization={scores['generalization']}  specificity={scores['specificity']}  "
              f"fluency {scores['fluency_before']} -> {scores['fluency_after']}")

    results_dir = cfg.abs_path(cfg.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    per_prompt = pd.concat(all_rows, ignore_index=True)
    per_prompt.to_csv(os.path.join(results_dir, "edit_eval_per_prompt.csv"), index=False)

    summary = pd.DataFrame(summaries)
    summary.to_csv(os.path.join(results_dir, "edit_eval_summary.csv"), index=False)
    print("\n=== summary ===")
    print(summary.to_string(index=False))

    # aggregate bar chart
    means = summary[["efficacy", "generalization", "specificity"]].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(means.index, means.values, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_ylim(0, 1); ax.set_ylabel("rate")
    ax.set_title(f"ROME-style edit quality ({cfg.model_name}) - mean over {len(targets)} targets")
    for i, v in enumerate(means.values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    plt.tight_layout()
    fig_path = os.path.join(results_dir, "edit_eval_summary.png")
    plt.savefig(fig_path, dpi=120)
    print("saved ->", fig_path)


if __name__ == "__main__":
    main()
