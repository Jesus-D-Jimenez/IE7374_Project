"""Experiment 9 - does the rank-one edit behave the same way in every model?

Runs the Experiment-4 edit-and-generate evaluation unchanged on each model in `--models`
and puts the four ROME scores side by side. The question is whether the identity-covariance
simplification's failure mode (edits that stick to the edited prompt but do not survive a
paraphrase) is a property of *this* small model or of the method as implemented here.

Usage:
    python scripts/run_edit_models.py
    python scripts/run_edit_models.py --models pythia-160m gpt2
"""
from __future__ import annotations
import argparse
import gc
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import evaluate_edit, load_config, load_model                # noqa: E402
from models.loader import _cached_load                                   # noqa: E402
from data import load_edit_targets                                       # noqa: E402
from utils.helpers import ensure_dir, get_logger, one_line, relative_to_repo, set_seed  # noqa: E402
from utils.visualization import grouped_bar_chart                        # noqa: E402

DEFAULT_MODELS = ["gpt2", "pythia-160m", "pythia-410m"]
SCORES = ["efficacy", "generalization", "specificity", "specificity_pred_preserved"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Cross-model edit evaluation.")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--device", default=None)
    ap.add_argument("--layer", type=int, default=None, help="force the edit layer for every model")
    ap.add_argument("--output-dir", default="outputs/study", dest="output_dir")
    args = ap.parse_args()

    base = load_config(device=args.device)
    out_dir = ensure_dir(base.abs_path(args.output_dir))
    log = get_logger("edit_models", log_file=os.path.join(out_dir, "run_edit_models.log"))
    set_seed(base.seed)
    targets = load_edit_targets(base.abs_path(base.data_dir))

    summaries, generations = [], []
    for name in args.models:
        cfg = load_config(model_name=name, device=args.device, seed=base.seed)
        model = load_model(cfg)
        log.info("=== %s (%d layers) ===", name, model.cfg.n_layers)
        for target in targets:
            df, scores = evaluate_edit(model, target, cfg, layer=args.layer)
            summaries.append({"model": name, "subject": target["subject"],
                              "true": target["true"], "new": target["new"], **scores})
            df.insert(0, "model", name)
            df.insert(1, "subject", target["subject"])
            generations.append(df)
            log.info("  %-18s layer=%2d efficacy=%.2f generalization=%.2f specificity=%.2f "
                     "(top-1 preserved %.2f) fluency %.2f -> %.2f",
                     target["subject"], scores["layer"], scores["efficacy"],
                     scores["generalization"], scores["specificity"],
                     scores["specificity_pred_preserved"], scores["fluency_before"],
                     scores["fluency_after"])
            after = df[(df.phase == "after") & (df.kind == "efficacy")]["generation"]
            if len(after):
                log.info("    edited generation: %s", one_line(after.iloc[0], 110))
        _cached_load.cache_clear()
        del model
        gc.collect()

    summary = pd.DataFrame(summaries)
    gens = pd.concat(generations, ignore_index=True)
    summary.to_csv(os.path.join(out_dir, "edit_by_model.csv"), index=False)
    gens.to_csv(os.path.join(out_dir, "edit_by_model_generations.csv"), index=False)

    means = summary.groupby("model", sort=False)[SCORES].mean().round(3)
    groups = {m: {s: float(row[s]) for s in SCORES} for m, row in means.iterrows()}
    fig = grouped_bar_chart(groups, os.path.join(out_dir, "edit_by_model.png"),
                            title="Rank-one edit quality by model (mean over 3 targets)",
                            ylabel="rate", ylim=(0, 1.05))

    log.info("\n%s", means.to_string())
    log.info("saved -> %s", relative_to_repo(fig))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
