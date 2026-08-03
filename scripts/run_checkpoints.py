"""Experiment 8 - when during pretraining does each behavior appear?

Pythia ships intermediate checkpoints of the *same* model, so the three behaviors can be
measured as a function of training step rather than only at the end of training. This is the
one experiment in the project that looks at how the circuitry got there: agreement is
expected to be present very early, factual recall to accumulate slowly, and induction to
appear abruptly (the induction-head phase change reported by Olsson et al. 2022).

Each checkpoint is downloaded (~350 MB), scored on all 48 prompts, and released before the
next one loads.

Usage:
    python scripts/run_checkpoints.py
    python scripts/run_checkpoints.py --steps 0 512 4000 143000
"""
from __future__ import annotations
import argparse
import gc
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import load_config, suite_logit_diffs                        # noqa: E402
from data import load_behavior_suite, SUITES                             # noqa: E402
from utils.helpers import ensure_dir, get_logger, relative_to_repo, set_seed, write_json  # noqa: E402
from utils.visualization import PALETTE                                  # noqa: E402

# Log-spaced over Pythia's 143k steps: dense where the induction phase change is reported.
DEFAULT_STEPS = [0, 16, 128, 512, 1000, 2000, 4000, 8000, 16000, 64000, 143000]


def load_checkpoint(model_name: str, step: int, device: str):
    """Load one intermediate Pythia checkpoint as a HookedTransformer."""
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(model_name, checkpoint_value=step, device=device)
    model.eval()
    return model


def score_checkpoint(model, cfg) -> list[dict]:
    """Per-suite mean logit difference and accuracy at one checkpoint."""
    out = []
    for suite in SUITES:
        records = load_behavior_suite(suite, cfg.abs_path(cfg.data_dir))
        df = pd.DataFrame(suite_logit_diffs(model, records))
        out.append({"suite": suite, "n": int(len(df)),
                    "mean_logit_diff": round(float(df["logit_diff"].mean()), 3),
                    "accuracy": round(float((df["logit_diff"] > 0).mean()), 3)})
    return out


def plot_curves(summary: pd.DataFrame, path: str, model_name: str) -> str:
    """Logit difference and accuracy vs. training step, one line per behavior."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for i, suite in enumerate(SUITES):
        s = summary[summary.suite == suite].sort_values("step")
        # step 0 cannot be drawn on a log axis; shift it to 0.5 and label the tick.
        x = s["step"].clip(lower=0.5)
        ax1.plot(x, s["mean_logit_diff"], "-o", markersize=4,
                 color=PALETTE[i % len(PALETTE)], label=suite)
        ax2.plot(x, s["accuracy"], "-o", markersize=4,
                 color=PALETTE[i % len(PALETTE)], label=suite)
    for ax, ylabel, title in ((ax1, "mean logit difference", "Behavior strength"),
                              (ax2, "share of prompts correct", "Behavior accuracy")):
        ax.set_xscale("log"); ax.set_xlabel("training step (0 plotted at 0.5)")
        ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3); ax.axhline(0, color="grey", linewidth=0.8)
    ax2.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=8)
    fig.suptitle(f"Behavior emergence during pretraining ({model_name})", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Behavior vs. pretraining checkpoint.")
    ap.add_argument("--model", default=None, help="a Pythia model (checkpoints are Pythia-only)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--steps", nargs="+", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--output-dir", default="outputs/study", dest="output_dir")
    args = ap.parse_args()

    cfg = load_config(model_name=args.model, device=args.device)
    out_dir = ensure_dir(cfg.abs_path(args.output_dir))
    log = get_logger("checkpoints", log_file=os.path.join(out_dir, "run_checkpoints.log"))
    set_seed(cfg.seed)

    rows, failed = [], []
    for step in args.steps:
        log.info("=== step %d ===", step)
        try:
            model = load_checkpoint(cfg.model_name, step, cfg.resolved_device())
        except Exception as exc:                       # a missing revision must not kill the run
            log.warning("step %d unavailable: %s", step, exc)
            failed.append({"step": step, "error": str(exc)})
            continue
        for record in score_checkpoint(model, cfg):
            rows.append({"model": cfg.model_name, "step": step, **record})
            log.info("  %-16s mean logit diff %+.3f  accuracy %.2f",
                     record["suite"], record["mean_logit_diff"], record["accuracy"])
        del model
        gc.collect()

    summary = pd.DataFrame(rows)
    if summary.empty:
        log.error("no checkpoint could be scored")
        return 1

    summary.to_csv(os.path.join(out_dir, "checkpoint_summary.csv"), index=False)
    fig = plot_curves(summary, os.path.join(out_dir, "checkpoint_emergence.png"), cfg.model_name)
    write_json(os.path.join(out_dir, "checkpoint_meta.json"),
               {"model": cfg.model_name, "steps": args.steps, "failed": failed})

    log.info("\n%s", summary.pivot(index="step", columns="suite",
                                   values="mean_logit_diff").to_string())
    log.info("saved -> %s", relative_to_repo(fig))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
