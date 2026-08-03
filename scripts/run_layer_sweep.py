"""Experiment 6 - which layer should the rank-one edit go into?

The committed pipeline picks the edit layer by causal tracing (the argmax of the
restoration curve at the subject's last token) and that choice sometimes produces an edit
with efficacy 0.00. This script measures the ground truth it should be compared against:
for every target, it applies the *same* edit at *every* layer and scores each one.

It then reports, for each layer-selection rule, whether that rule lands on a layer that
actually works:

  * raw argmax              - `best_edit_layer(recovery, pos, window=1)`
  * windowed argmax (w=3,5) - the small-model analogue of ROME's multi-layer window
  * seed stability          - the layer each rule picks under 5 different tracing noise seeds

Usage:
    python scripts/run_layer_sweep.py
    python scripts/run_layer_sweep.py --model gpt2 --seeds 0 1 2
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import (                                                     # noqa: E402
    apply_rank_one_edit, best_edit_layer, causal_trace, load_config, load_model,
    optimize_v_star, restore_weights, summarize_scores,
)
from models.metrics import snapshot_generations                          # noqa: E402
from data import load_edit_targets                                       # noqa: E402
from utils.helpers import ensure_dir, get_logger, relative_to_repo, set_seed, write_json  # noqa: E402
from utils.visualization import PALETTE                                  # noqa: E402

WINDOWS = (1, 3, 5)


def sweep_target(model, target: dict, cfg, log) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Score an edit of `target` at every layer. Returns (rows, recovery grid, subj pos)."""
    recovery, subj_last, _ = causal_trace(model, target["prompt"], target["subject"],
                                          target["true"], cfg)

    restore_weights(model)
    before = snapshot_generations(model, target, cfg)
    before["phase"] = "before"

    rows = []
    for layer in range(model.cfg.n_layers):
        edit = optimize_v_star(model, target["prompt"], target["subject"], target["new"],
                               layer, cfg)
        apply_rank_one_edit(model, edit)
        after = snapshot_generations(model, target, cfg)
        after["phase"] = "after"
        restore_weights(model)

        scores = summarize_scores(pd.concat([before, after], ignore_index=True))
        rows.append({"subject": target["subject"], "layer": layer,
                     "trace_recovery": round(float(recovery[layer, subj_last]), 4),
                     "final_fit_loss": round(float(edit.losses[-1]), 4), **scores})
        log.info("  layer %2d  efficacy=%.2f generalization=%.2f specificity=%.2f "
                 "(top-1 preserved %.2f)  fluency %.2f -> %.2f  recovery=%.3f",
                 layer, scores["efficacy"], scores["generalization"], scores["specificity"],
                 scores["specificity_pred_preserved"], scores["fluency_before"],
                 scores["fluency_after"], recovery[layer, subj_last])
    return pd.DataFrame(rows), recovery, subj_last


def selection_rules(model, target: dict, cfg, seeds: list[int]) -> list[dict]:
    """Which layer each (window, seed) rule selects — the tracing side of the comparison."""
    picks = []
    for seed in seeds:
        seed_cfg = load_config(model_name=cfg.model_name, device=cfg.device, seed=seed)
        recovery, subj_last, _ = causal_trace(model, target["prompt"], target["subject"],
                                              target["true"], seed_cfg)
        for window in WINDOWS:
            picks.append({"subject": target["subject"], "seed": seed, "window": window,
                          "layer": best_edit_layer(recovery, subj_last, window=window)})
    return picks


def plot_sweep(sweep: pd.DataFrame, picks: pd.DataFrame, path: str, model_name: str) -> str:
    """Efficacy vs. layer for every target, shading the layers causal tracing selects.

    Only the default rule (window = 1) is shaded: it is the rule the pipeline actually uses,
    and drawing all three windows over five seeds buries the curves the figure is about.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subjects = list(sweep["subject"].unique())
    fig, axes = plt.subplots(1, len(subjects), figsize=(4.2 * len(subjects), 3.9), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, subject in zip(axes, subjects):
        s = sweep[sweep.subject == subject].sort_values("layer")

        traced = sorted(picks[(picks.subject == subject) & (picks.window == 1)]["layer"].unique())
        for j, layer in enumerate(traced):
            ax.axvspan(layer - 0.35, layer + 0.35, color=PALETTE[4], alpha=0.45, linewidth=0,
                       label="traced layer (5 seeds)" if j == 0 else None)

        ax.plot(s.layer, s.efficacy, "-o", color=PALETTE[0], label="efficacy", markersize=5)
        ax.plot(s.layer, s.specificity_pred_preserved, "-s", color=PALETTE[1],
                label="top-1 preserved", markersize=4, alpha=0.85)
        recovery = s.trace_recovery / (s.trace_recovery.abs().max() or 1.0)
        ax.plot(s.layer, recovery, "--", color=PALETTE[2], alpha=0.9,
                label="trace recovery (scaled)")

        ax.set_title(subject, fontsize=10)
        ax.set_xlabel("edited MLP layer")
        ax.set_xticks(range(0, int(s.layer.max()) + 1, 2))
        ax.set_ylim(-0.35, 1.25)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("score")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, fontsize=8, loc="lower center", frameon=False)
    fig.suptitle(f"Edit quality at every layer ({model_name})", fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep the edit over every layer.")
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                    help="tracing-noise seeds used for the layer-selection stability check")
    ap.add_argument("--output-dir", default="outputs/study", dest="output_dir")
    ap.add_argument("--plot-only", action="store_true", dest="plot_only",
                    help="redraw the figure from the saved CSVs (no model, no re-fitting)")
    args = ap.parse_args()

    cfg = load_config(model_name=args.model, device=args.device)
    out_dir = ensure_dir(cfg.abs_path(args.output_dir))
    log = get_logger("layer_sweep", log_file=os.path.join(out_dir, "run_layer_sweep.log"))
    set_seed(cfg.seed)

    if args.plot_only:
        sweep = pd.read_csv(os.path.join(out_dir, "layer_sweep.csv"))
        picks = pd.read_csv(os.path.join(out_dir, "layer_selection_picks.csv"))
        fig = plot_sweep(sweep, picks, os.path.join(out_dir, "layer_sweep.png"), cfg.model_name)
        log.info("redrawn -> %s", relative_to_repo(fig))
        return 0

    model = load_model(cfg)
    targets = load_edit_targets(cfg.abs_path(cfg.data_dir))

    sweeps, all_picks, traces = [], [], {}
    for target in targets:
        log.info("=== %s: %s -> %s ===", target["subject"], target["true"], target["new"])
        rows, recovery, subj_last = sweep_target(model, target, cfg, log)
        sweeps.append(rows)
        traces[target["subject"]] = {
            "subject_last_pos": int(subj_last),
            "recovery_at_subject": [round(float(v), 4) for v in recovery[:, subj_last]],
        }
        all_picks += selection_rules(model, target, cfg, args.seeds)

    sweep = pd.concat(sweeps, ignore_index=True)
    picks = pd.DataFrame(all_picks)

    # Does the layer a rule picks actually work? Join each pick to its swept scores.
    scored = picks.merge(sweep[["subject", "layer", "efficacy", "generalization",
                                "specificity_pred_preserved"]],
                         on=["subject", "layer"], how="left")
    rule_summary = (scored.groupby("window")
                    .agg(mean_efficacy=("efficacy", "mean"),
                         mean_generalization=("generalization", "mean"),
                         mean_top1_preserved=("specificity_pred_preserved", "mean"),
                         n_distinct_layers=("layer", "nunique"))
                    .round(3).reset_index())

    sweep.to_csv(os.path.join(out_dir, "layer_sweep.csv"), index=False)
    scored.to_csv(os.path.join(out_dir, "layer_selection_picks.csv"), index=False)
    rule_summary.to_csv(os.path.join(out_dir, "layer_selection_rules.csv"), index=False)
    write_json(os.path.join(out_dir, "trace_profiles.json"),
               {"model": cfg.model_name, "targets": traces})
    fig = plot_sweep(sweep, picks, os.path.join(out_dir, "layer_sweep.png"), cfg.model_name)

    best = (sweep.sort_values(["subject", "efficacy", "specificity_pred_preserved"],
                              ascending=[True, False, False])
            .groupby("subject").head(1))
    log.info("\nbest layer per target by efficacy:\n%s",
             best[["subject", "layer", "efficacy", "generalization",
                   "specificity_pred_preserved"]].to_string(index=False))
    log.info("\nlayers that work (efficacy = 1.0), per target:\n%s",
             sweep[sweep.efficacy == 1.0].groupby("subject")["layer"].apply(list).to_string())
    log.info("\nlayer-selection rule quality (averaged over %d seeds x %d targets):\n%s",
             len(args.seeds), len(targets), rule_summary.to_string(index=False))
    log.info("saved -> %s", relative_to_repo(fig))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
