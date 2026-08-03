"""Experiment 7 - are the components that carry each behavior distinct components?

`scripts/run_ablation.py` scans one prompt. This scans several prompts per behavior and
asks the comparative question the project is really about: does knocking out the heads that
matter for *agreement* also break *factual recall* and *induction*, or is each behavior
carried by its own circuitry?

For every behavior it
  1. zero-ablates each attention head and each MLP in turn, averaging the drop in logit
     difference over `--per-suite` prompts;
  2. ranks the components by that mean drop;
  3. scores the top head of each behavior with `selectivity_score` — +1 means the head hurts
     only its own behavior, 0 means it is equally important everywhere.

Usage:
    python scripts/run_selectivity.py
    python scripts/run_selectivity.py --model gpt2 --per-suite 3
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import (                                                     # noqa: E402
    ablate_component, head_ablation_scan, load_config, load_model, logit_diff,
    selectivity_score,
)
from data import load_behavior_suite, SUITES                             # noqa: E402
from utils.helpers import ensure_dir, get_logger, relative_to_repo, set_seed, write_json  # noqa: E402
from utils.visualization import PALETTE                                  # noqa: E402


def mlp_ablation_scan(model, prompt: str, correct: str, incorrect: str) -> np.ndarray:
    """Drop in logit difference caused by zero-ablating each MLP block."""
    base = logit_diff(model, prompt, correct, incorrect)
    return np.array([base - ablate_component(model, prompt, correct, incorrect,
                                             {"type": "mlp", "layer": l})
                     for l in range(model.cfg.n_layers)])


def scan_suite(model, records: list[dict], log, suite: str) -> tuple[np.ndarray, np.ndarray]:
    """Mean head grid [n_layers, n_heads] and mean MLP vector [n_layers] over `records`."""
    heads, mlps = [], []
    for r in records:
        heads.append(head_ablation_scan(model, r["prompt"], r["correct"], r["incorrect"]))
        mlps.append(mlp_ablation_scan(model, r["prompt"], r["correct"], r["incorrect"]))
        log.info("  scanned %-12s %s", r["id"], r["prompt"][:48])
    head_grid, mlp_vec = np.mean(heads, axis=0), np.mean(mlps, axis=0)
    l, h = np.unravel_index(np.argmax(head_grid), head_grid.shape)
    log.info("%-16s top head L%d.H%d (drop %+.3f) | top MLP L%d (drop %+.3f)",
             suite, l, h, head_grid[l, h], int(np.argmax(mlp_vec)), mlp_vec.max())
    return head_grid, mlp_vec


def plot_heatmaps(grids: dict[str, np.ndarray], path: str, model_name: str) -> str:
    """One head-ablation heatmap per behavior, on a shared colour scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(grids)
    vmax = max(np.abs(g).max() for g in grids.values())
    fig, axes = plt.subplots(1, len(names), figsize=(4.0 * len(names), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        im = ax.imshow(grids[name], aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("head")
    axes[0].set_ylabel("layer")
    fig.colorbar(im, ax=axes.tolist(), label="mean logit-diff drop when ablated")
    fig.suptitle(f"Head ablation by behavior ({model_name})", fontsize=11)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_mlp(mlps: dict[str, np.ndarray], path: str, model_name: str) -> str:
    """MLP-ablation profile per behavior — the attention/MLP division of labour."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for i, (name, vec) in enumerate(mlps.items()):
        ax.plot(range(len(vec)), vec, "-o", markersize=4, color=PALETTE[i % len(PALETTE)],
                label=name)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("MLP layer"); ax.set_ylabel("mean logit-diff drop when ablated")
    ax.set_title(f"MLP ablation by behavior ({model_name})", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-behavior ablation and selectivity.")
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--per-suite", type=int, default=4, dest="per_suite",
                    help="prompts averaged per behavior (default 4)")
    ap.add_argument("--top", type=int, default=5, help="components listed per behavior")
    ap.add_argument("--output-dir", default="outputs/study", dest="output_dir")
    args = ap.parse_args()

    cfg = load_config(model_name=args.model, device=args.device)
    out_dir = ensure_dir(cfg.abs_path(args.output_dir))
    log = get_logger("selectivity", log_file=os.path.join(out_dir, "run_selectivity.log"))
    set_seed(cfg.seed)

    model = load_model(cfg)
    head_grids, mlp_vecs, rows = {}, {}, []
    for suite in SUITES:
        records = load_behavior_suite(suite, cfg.abs_path(cfg.data_dir))[: args.per_suite]
        log.info("=== %s (%d prompts) ===", suite, len(records))
        head_grids[suite], mlp_vecs[suite] = scan_suite(model, records, log, suite)

    for suite, grid in head_grids.items():
        flat = [dict(suite=suite, component=f"L{l}.H{h}", type="head", layer=l, head=h,
                     mean_drop=round(float(grid[l, h]), 4))
                for l in range(grid.shape[0]) for h in range(grid.shape[1])]
        flat += [dict(suite=suite, component=f"L{l}.MLP", type="mlp", layer=l, head=None,
                      mean_drop=round(float(mlp_vecs[suite][l]), 4))
                 for l in range(len(mlp_vecs[suite]))]
        rows += flat
    scan = pd.DataFrame(rows)

    # Selectivity of each behavior's top head, measured against the other two behaviors.
    sel_rows = []
    for suite, grid in head_grids.items():
        l, h = np.unravel_index(np.argmax(grid), grid.shape)
        target_drop = float(grid[l, h])
        others = {s: float(g[l, h]) for s, g in head_grids.items() if s != suite}
        control_drop = float(np.mean(list(others.values())))
        sel_rows.append({"suite": suite, "top_head": f"L{l}.H{h}",
                         "drop_on_own_behavior": round(target_drop, 4),
                         "mean_drop_on_others": round(control_drop, 4),
                         "selectivity": round(selectivity_score(target_drop, control_drop), 4),
                         **{f"drop_on_{s}": round(v, 4) for s, v in others.items()}})
    selectivity = pd.DataFrame(sel_rows)

    scan.to_csv(os.path.join(out_dir, "ablation_scan_by_suite.csv"), index=False)
    selectivity.to_csv(os.path.join(out_dir, "head_selectivity.csv"), index=False)
    top = (scan.sort_values("mean_drop", ascending=False).groupby("suite").head(args.top))
    top.to_csv(os.path.join(out_dir, "ablation_top_components.csv"), index=False)
    fig1 = plot_heatmaps(head_grids, os.path.join(out_dir, "ablation_heatmaps.png"), cfg.model_name)
    fig2 = plot_mlp(mlp_vecs, os.path.join(out_dir, "ablation_mlp.png"), cfg.model_name)
    write_json(os.path.join(out_dir, "ablation_meta.json"),
               {"model": cfg.model_name, "per_suite": args.per_suite,
                "n_layers": model.cfg.n_layers, "n_heads": model.cfg.n_heads})

    log.info("\ntop %d components per behavior:\n%s", args.top,
             top[["suite", "component", "mean_drop"]].to_string(index=False))
    log.info("\nselectivity of each behavior's top head:\n%s", selectivity.to_string(index=False))
    log.info("saved -> %s, %s", relative_to_repo(fig1), relative_to_repo(fig2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
