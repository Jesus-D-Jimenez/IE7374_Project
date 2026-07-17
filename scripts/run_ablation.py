"""Experiment 3a - attention-head ablation scan for one prompt.

Ablates every head and reports the drop in logit difference, i.e. how much each head
contributes to the correct prediction. Saves a heatmap and a ranked table.

Usage:
    python scripts/run_ablation.py
    python scripts/run_ablation.py --prompt "The capital of Japan is" --correct " Tokyo" --incorrect " Beijing"
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import load_config, load_model, head_ablation_scan, logit_diff   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Head-ablation scan.")
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--correct", default=" Paris")
    ap.add_argument("--incorrect", default=" London")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    cfg = load_config(model_name=args.model, device=args.device)
    model = load_model(cfg)

    base = logit_diff(model, args.prompt, args.correct, args.incorrect)
    print(f"baseline logit diff = {base:+.3f}  |  prompt: {args.prompt!r}")

    grid = head_ablation_scan(model, args.prompt, args.correct, args.incorrect)

    results_dir = cfg.abs_path(cfg.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # ranked table
    rows = [dict(layer=l, head=h, logit_diff_drop=float(grid[l, h]))
            for l in range(grid.shape[0]) for h in range(grid.shape[1])]
    tbl = pd.DataFrame(rows).sort_values("logit_diff_drop", ascending=False)
    tbl.to_csv(os.path.join(results_dir, "head_ablation_scan.csv"), index=False)
    print("\nTop heads by contribution (drop in logit diff when ablated):")
    print(tbl.head(args.top).to_string(index=False))

    # heatmap
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(grid, aspect="auto", cmap="RdBu_r",
                   vmin=-np.abs(grid).max(), vmax=np.abs(grid).max())
    ax.set_xlabel("head"); ax.set_ylabel("layer")
    ax.set_title(f"Head ablation - drop in logit diff\n{args.model or cfg.model_name}: {args.prompt!r}")
    plt.colorbar(im, label="logit-diff drop when ablated")
    plt.tight_layout()
    fig_path = os.path.join(results_dir, "head_ablation_scan.png")
    plt.savefig(fig_path, dpi=120)
    print("saved ->", fig_path)


if __name__ == "__main__":
    main()
