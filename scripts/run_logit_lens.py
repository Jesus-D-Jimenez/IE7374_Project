"""Experiment 2 - where in the stack does the answer become the answer?

The logit lens decodes every layer's residual stream through the unembedding, so the
log-probability of the correct answer can be read off layer by layer. Averaged over a suite
it shows *when* each behavior is resolved: a behavior that is settled at layer 3 is being
computed by different machinery than one that only separates in the last two layers.

Usage:
    python scripts/run_logit_lens.py
    python scripts/run_logit_lens.py --model pythia-410m --per-suite 8
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import load_config, load_model, logit_lens                   # noqa: E402
from data import load_behavior_suite, SUITES                             # noqa: E402
from utils.helpers import ensure_dir, get_logger, relative_to_repo, set_seed  # noqa: E402
from utils.visualization import PALETTE                                  # noqa: E402


def plot_lens(df: pd.DataFrame, path: str, model_name: str) -> str:
    """Correct-minus-incorrect log-prob per layer, one line per behavior."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for i, suite in enumerate(SUITES):
        s = df[df.suite == suite].sort_values("layer")
        color = PALETTE[i % len(PALETTE)]
        ax1.plot(s.layer, s.correct_logprob, "-o", markersize=4, color=color, label=suite)
        ax2.plot(s.layer, s.margin, "-o", markersize=4, color=color, label=suite)
    ax1.set_ylabel("mean log-prob of the correct answer")
    ax2.set_ylabel("mean log-prob margin (correct - incorrect)")
    for ax, title in ((ax1, "Answer confidence by depth"), (ax2, "Correct-vs-incorrect margin")):
        ax.set_xlabel("layer (residual stream after block L)")
        ax.set_title(title, fontsize=10); ax.grid(alpha=0.3)
        ax.axhline(0, color="grey", linewidth=0.8)
    ax1.legend(fontsize=8)
    fig.suptitle(f"Logit lens by behavior ({model_name})", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Logit-lens trajectories per behavior.")
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--per-suite", type=int, default=8, dest="per_suite")
    ap.add_argument("--output-dir", default="outputs/study", dest="output_dir")
    args = ap.parse_args()

    cfg = load_config(model_name=args.model, device=args.device)
    out_dir = ensure_dir(cfg.abs_path(args.output_dir))
    log = get_logger("logit_lens", log_file=os.path.join(out_dir, "run_logit_lens.log"))
    set_seed(cfg.seed)
    model = load_model(cfg)

    rows = []
    for suite in SUITES:
        records = load_behavior_suite(suite, cfg.abs_path(cfg.data_dir))[: args.per_suite]
        correct = np.mean([logit_lens(model, r["prompt"], r["correct"]) for r in records], axis=0)
        incorrect = np.mean([logit_lens(model, r["prompt"], r["incorrect"]) for r in records],
                            axis=0)
        for layer, (c, w) in enumerate(zip(correct, incorrect)):
            rows.append({"model": cfg.model_name, "suite": suite, "layer": layer,
                         "correct_logprob": round(float(c), 4),
                         "incorrect_logprob": round(float(w), 4),
                         "margin": round(float(c - w), 4)})
        crossing = next((l for l, m in enumerate(correct - incorrect) if m > 0), None)
        log.info("%-16s n=%d  margin turns positive at layer %s  final margin %+.3f",
                 suite, len(records), crossing, (correct - incorrect)[-1])

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "logit_lens.csv"), index=False)
    fig = plot_lens(df, os.path.join(out_dir, "logit_lens.png"), cfg.model_name)
    log.info("saved -> %s", relative_to_repo(fig))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
