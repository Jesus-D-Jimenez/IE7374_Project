"""Experiment 10 - why the same edit hyperparameters do not transfer between models.

`scripts/run_edit_models.py` reports efficacy 0.00 for GPT-2 on every target while
pythia-160m succeeds on two of three. The cause is not that GPT-2 stores facts differently:
it is that the edit objective

    loss = -log p(new object) + edit_kl_weight * ||delta||^2

carries an *absolute* penalty on the injected vector, while the size of a vector that
actually changes the prediction depends on the norm of the residual stream it is added to.
GPT-2's residual stream at the edited layer is roughly an order of magnitude larger than
pythia-160m's, so the delta needed there is penalized ~100x harder and the optimizer drives
it back to zero.

This script measures that: for each model it traces the edit layer once, then re-fits and
re-scores the edit at several penalty weights, recording the norm of the delta each time.

Usage:
    python scripts/run_edit_penalty.py
    python scripts/run_edit_penalty.py --models gpt2 --weights 0.0625 0.0
"""
from __future__ import annotations
import argparse
import gc
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import (                                                     # noqa: E402
    apply_rank_one_edit, best_edit_layer, causal_trace, load_config, load_model,
    optimize_v_star, restore_weights, summarize_scores,
)
from models.loader import _cached_load                                   # noqa: E402
from models.metrics import snapshot_generations                          # noqa: E402
from data import load_edit_targets                                       # noqa: E402
from utils.helpers import ensure_dir, get_logger, relative_to_repo, set_seed, write_json  # noqa: E402
from utils.visualization import PALETTE                                  # noqa: E402

DEFAULT_MODELS = ["gpt2", "pythia-160m", "pythia-410m"]
DEFAULT_WEIGHTS = [0.0625, 0.01, 0.001, 0.0]


def residual_norms(model, prompt: str) -> list[float]:
    """Norm of the residual stream at the last position, per layer — the scale that matters."""
    _, cache = model.run_with_cache(model.to_tokens(prompt))
    return [round(float(cache[f"blocks.{l}.hook_resid_post"][0, -1].norm()), 2)
            for l in range(model.cfg.n_layers)]


def plot_penalty(df: pd.DataFrame, path: str) -> str:
    """Left: the trade-off the penalty controls. Right: the delta size it permits, per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.9))

    # 0 has no place on a log axis; plot it a decade below the smallest non-zero weight.
    nonzero = df[df.edit_kl_weight > 0]["edit_kl_weight"]
    floor = (nonzero.min() / 10) if len(nonzero) else 1e-4
    df = df.assign(x=df["edit_kl_weight"].replace(0.0, floor))

    pooled = df.groupby("x", sort=True)[["efficacy", "generalization",
                                         "specificity_pred_preserved"]].mean().reset_index()
    for i, (col, label) in enumerate((("efficacy", "efficacy"),
                                      ("generalization", "generalization"),
                                      ("specificity_pred_preserved", "top-1 preserved"))):
        ax1.plot(pooled.x, pooled[col], "-o", color=PALETTE[i], markersize=5, label=label)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_ylabel("mean score (3 models x 3 targets)")
    ax1.set_title("What the penalty trades away", fontsize=10)
    ax1.legend(fontsize=8, loc="center left")

    for i, (model, g) in enumerate(df.groupby("model", sort=False)):
        g = g.groupby("x", sort=True)["delta_norm"].mean().reset_index()
        ax2.plot(g.x, g.delta_norm, "-o", color=PALETTE[i % len(PALETTE)], markersize=5,
                 label=model)
    ax2.set_yscale("log")
    ax2.set_ylabel("mean ||delta|| injected")
    ax2.set_title("How large an edit the penalty permits", fontsize=10)
    ax2.legend(fontsize=8)

    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xlabel(f"edit_kl_weight (0 plotted at {floor:g})")
        ax.grid(alpha=0.3)
        ax.axvline(0.0625, color="grey", linestyle=":", linewidth=1.4)
        ax.annotate("default", xy=(0.0625, ax.get_ylim()[1]), fontsize=7, color="grey",
                    ha="right", va="top", rotation=90)

    fig.suptitle("The edit penalty is absolute, but the scale it must overcome is not",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Edit-penalty sensitivity across models.")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--weights", nargs="+", type=float, default=DEFAULT_WEIGHTS)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output-dir", default="outputs/study", dest="output_dir")
    ap.add_argument("--plot-only", action="store_true", dest="plot_only",
                    help="redraw the figure from the saved CSV (no model, no re-fitting)")
    args = ap.parse_args()

    base = load_config(device=args.device)
    out_dir = ensure_dir(base.abs_path(args.output_dir))
    log = get_logger("edit_penalty", log_file=os.path.join(out_dir, "run_edit_penalty.log"))
    set_seed(base.seed)

    if args.plot_only:
        df = pd.read_csv(os.path.join(out_dir, "edit_penalty.csv"))
        fig = plot_penalty(df, os.path.join(out_dir, "edit_penalty.png"))
        log.info("redrawn -> %s", relative_to_repo(fig))
        return 0

    targets = load_edit_targets(base.abs_path(base.data_dir))

    rows, norms = [], {}
    for name in args.models:
        cfg = load_config(model_name=name, device=args.device, seed=base.seed)
        model = load_model(cfg)
        norms[name] = residual_norms(model, targets[0]["prompt"])
        log.info("=== %s === residual norms by layer: %s", name, norms[name])

        for target in targets:
            recovery, subj_last, _ = causal_trace(model, target["prompt"], target["subject"],
                                                  target["true"], cfg)
            layer = best_edit_layer(recovery, subj_last, window=cfg.edit_layer_window)

            restore_weights(model)
            before = snapshot_generations(model, target, cfg)
            before["phase"] = "before"

            for weight in args.weights:
                run_cfg = load_config(model_name=name, device=args.device, seed=base.seed,
                                      edit_kl_weight=weight)
                edit = optimize_v_star(model, target["prompt"], target["subject"],
                                       target["new"], layer, run_cfg)
                apply_rank_one_edit(model, edit)
                after = snapshot_generations(model, target, run_cfg)
                after["phase"] = "after"
                restore_weights(model)

                scores = summarize_scores(pd.concat([before, after], ignore_index=True))
                delta_norm = float((edit.v_star - edit.v_orig).norm())
                rows.append({"model": name, "subject": target["subject"], "layer": layer,
                             "edit_kl_weight": weight,
                             "delta_norm": round(delta_norm, 3),
                             "resid_norm_at_layer": norms[name][layer],
                             "final_loss": round(float(edit.losses[-1]), 4), **scores})
                log.info("  %-18s kl=%-7g |delta|=%8.2f  efficacy=%.2f generalization=%.2f "
                         "specificity=%.2f (top-1 preserved %.2f) fluency %.2f -> %.2f",
                         target["subject"], weight, delta_norm, scores["efficacy"],
                         scores["generalization"], scores["specificity"],
                         scores["specificity_pred_preserved"], scores["fluency_before"],
                         scores["fluency_after"])
        _cached_load.cache_clear()
        del model
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "edit_penalty.csv"), index=False)
    write_json(os.path.join(out_dir, "residual_norms.json"), norms)
    fig = plot_penalty(df, os.path.join(out_dir, "edit_penalty.png"))

    pivot = df.pivot_table(index="edit_kl_weight", columns="model",
                           values=["efficacy", "generalization", "delta_norm"]).round(3)
    log.info("\n%s", pivot.to_string())
    log.info("saved -> %s", relative_to_repo(fig))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
