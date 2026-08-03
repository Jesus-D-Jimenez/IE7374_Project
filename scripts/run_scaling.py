"""Experiment 5 - does the behavior hold across model families and scales?

Runs the full 48-prompt suite through every model in `--models` and reports, per behavior,
the mean logit difference and the share of prompts on which the model prefers the correct
continuation. GPT-2 small is the cross-family control (different tokenizer, different
training corpus); pythia-410m is the scale step inside the same family.

Usage:
    python scripts/run_scaling.py
    python scripts/run_scaling.py --models gpt2 pythia-160m
    python scripts/run_scaling.py --output-dir outputs/study
"""
from __future__ import annotations
import argparse
import gc
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import load_config, load_model, suite_logit_diffs            # noqa: E402
from models.loader import _cached_load                                   # noqa: E402
from data import load_behavior_suite, SUITES                             # noqa: E402
from utils.helpers import ensure_dir, get_logger, relative_to_repo, set_seed, write_json  # noqa: E402
from utils.visualization import grouped_bar_chart                        # noqa: E402

DEFAULT_MODELS = ["gpt2", "pythia-160m", "pythia-410m"]


def score_model(model_name: str, cfg, log) -> tuple[pd.DataFrame, dict]:
    """Every prompt in every suite, scored by one model. Returns (rows, model facts)."""
    model_cfg = load_config(model_name=model_name, device=cfg.device, seed=cfg.seed)
    model = load_model(model_cfg)
    facts = {
        "model": model_name,
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "n_layers": model.cfg.n_layers,
        "n_heads": model.cfg.n_heads,
        "d_model": model.cfg.d_model,
    }

    frames = []
    for suite in SUITES:
        records = load_behavior_suite(suite, model_cfg.abs_path(model_cfg.data_dir))
        df = pd.DataFrame(suite_logit_diffs(model, records))
        df["suite"], df["model"] = suite, model_name
        frames.append(df)
        log.info("%-12s %-16s n=%3d  mean logit diff %+.3f  accuracy %.2f",
                 model_name, suite, len(df), df["logit_diff"].mean(),
                 (df["logit_diff"] > 0).mean())

    # One model at a time: three float32 models at once is needless memory pressure.
    _cached_load.cache_clear()
    del model
    gc.collect()
    return pd.concat(frames, ignore_index=True), facts


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """Per (model, suite) mean logit difference and accuracy, plus an 'all' row."""
    def agg(df: pd.DataFrame, model: str, suite: str) -> dict:
        return {"model": model, "suite": suite, "n": int(len(df)),
                "mean_logit_diff": round(float(df["logit_diff"].mean()), 3),
                "accuracy": round(float((df["logit_diff"] > 0).mean()), 3)}

    out = [agg(g, m, s) for (m, s), g in rows.groupby(["model", "suite"], sort=False)]
    out += [agg(g, m, "all") for m, g in rows.groupby("model", sort=False)]
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Cross-model / cross-scale behavioral comparison.")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--device", default=None, help="cpu | cuda")
    ap.add_argument("--output-dir", default="outputs/study", dest="output_dir",
                    help="where the committed summary CSV and figure are written")
    args = ap.parse_args()

    cfg = load_config(device=args.device)
    out_dir = ensure_dir(cfg.abs_path(args.output_dir))
    log = get_logger("scaling", log_file=os.path.join(out_dir, "run_scaling.log"))
    set_seed(cfg.seed)

    frames, facts = [], []
    for name in args.models:
        log.info("=== %s ===", name)
        rows, model_facts = score_model(name, cfg, log)
        frames.append(rows)
        facts.append(model_facts)

    rows = pd.concat(frames, ignore_index=True)
    summary = summarize(rows)

    keep = ["model", "suite", "id", "prompt", "correct", "incorrect", "logit_diff"]
    rows[[c for c in keep if c in rows]].to_csv(
        os.path.join(out_dir, "scaling_logit_diffs.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "scaling_summary.csv"), index=False)

    groups = {m: {s: float(g[g.suite == s]["mean_logit_diff"].iloc[0])
                  for s in [*SUITES, "all"] if (g.suite == s).any()}
              for m, g in summary.groupby("model", sort=False)}
    fig = grouped_bar_chart(groups, os.path.join(out_dir, "scaling_logit_diff.png"),
                            title="Mean logit difference by behavior and model",
                            ylabel="logit(correct) - logit(incorrect)")

    write_json(os.path.join(out_dir, "scaling_models.json"), facts)
    log.info("\n%s", summary.to_string(index=False))
    log.info("saved -> %s", relative_to_repo(fig))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
