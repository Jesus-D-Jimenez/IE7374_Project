"""Fit and save the rank-one edit — the only optimization in this project.

The pretrained model is never fine-tuned: we study behavior it already has. The one
"training" step is the ROME-style fit of Meng et al. (2022), which optimizes a single
target vector v* by gradient descent and folds it into one MLP output matrix as a
rank-one update. This script exposes that fit on its own so the optimization can be
inspected (loss curve, chosen layer, update norm) and the fitted edit reused without
re-running the fit.

    python src/train.py                       # fit every target in the processed dataset
    python src/train.py --target 0            # fit only the first target
    python src/train.py --layer 5 --steps 50  # force the layer, run longer

Artifacts land in `outputs/edits/`:
    edit_<slug>.pt          the fitted vectors (torch), reloadable with load_edit()
    edit_<slug>.json        layer, norms, final loss — small enough to commit
    edit_fit_losses.csv     loss per step for every target
    ../images/edit_fit_loss.png   the loss curves
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from typing import Optional

import pandas as pd

# Make the repository root importable when this file is run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.config import ProjectConfig, load_config                      # noqa: E402
from src.data_loader import load_processed_edit_targets                   # noqa: E402
from utils.helpers import ensure_dir, get_logger, relative_to_repo, set_seed, write_json  # noqa: E402
from utils.visualization import line_plot                                 # noqa: E402

EDITS_SUBDIR = "edits"
LOSSES_CSV = "edit_fit_losses.csv"


def slugify(text: str) -> str:
    """'The Eiffel Tower' -> 'the_eiffel_tower' (safe, stable file names)."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def choose_layer(model, target: dict, cfg: ProjectConfig, logger=None) -> int:
    """Return the MLP layer to edit: the causal-tracing peak, or the configured override."""
    log = logger or get_logger("train")
    if cfg.edit_layer != "auto":
        return int(cfg.edit_layer)

    from models.interpret import best_edit_layer, causal_trace
    log.info("  causal tracing to localize '%s' (%d layers)...",
             target["subject"], model.cfg.n_layers)
    recovery, subject_last, _ = causal_trace(model, target["prompt"], target["subject"],
                                             target["true"], cfg)
    layer = best_edit_layer(recovery, subject_last)
    log.info("  peak recovery at the subject's last token -> layer %d", layer)
    return layer


def fit_target(model, target: dict, cfg: ProjectConfig, outputs_dir: str,
               layer: Optional[int] = None, logger=None) -> dict:
    """Fit v* for one factual target, save the edit, and return a summary record."""
    import torch
    from models.editing import optimize_v_star

    log = logger or get_logger("train")
    log.info("fitting '%s': %s -> %s", target["subject"], target["true"], target["new"])

    edit_layer = layer if layer is not None else choose_layer(model, target, cfg, log)
    started = time.time()
    edit = optimize_v_star(model, target["prompt"], target["subject"], target["new"],
                           edit_layer, cfg)
    elapsed = time.time() - started

    delta = (edit.v_star - edit.v_orig)
    slug = slugify(target["subject"])
    edits_dir = ensure_dir(os.path.join(outputs_dir, EDITS_SUBDIR))

    tensor_path = os.path.join(edits_dir, f"edit_{slug}.pt")
    torch.save({"layer": edit.layer, "position": edit.position, "v_star": edit.v_star,
                "v_orig": edit.v_orig, "k_vec": edit.k_vec, "subject": target["subject"],
                "prompt": target["prompt"], "true": target["true"], "new": target["new"],
                "model_name": cfg.model_name}, tensor_path)

    record = {
        "subject": target["subject"],
        "prompt": target["prompt"],
        "true": target["true"],
        "new": target["new"],
        "model_name": cfg.model_name,
        "layer": int(edit.layer),
        "subject_last_token_position": int(edit.position),
        "steps": int(cfg.edit_steps),
        "lr": cfg.edit_lr,
        "l2_penalty": cfg.edit_kl_weight,
        "loss_first": round(float(edit.losses[0]), 4) if edit.losses else None,
        "loss_final": round(float(edit.losses[-1]), 4) if edit.losses else None,
        "delta_norm": round(float(delta.norm()), 4),
        "v_orig_norm": round(float(edit.v_orig.norm()), 4),
        "seconds": round(elapsed, 1),
        "tensor_file": relative_to_repo(tensor_path),
    }
    write_json(os.path.join(edits_dir, f"edit_{slug}.json"), record)
    log.info("  loss %.4f -> %.4f in %d steps (%.1fs) | ||delta||=%.3f | saved %s",
             record["loss_first"], record["loss_final"], cfg.edit_steps, elapsed,
             record["delta_norm"], relative_to_repo(tensor_path))

    record["losses"] = [round(float(x), 5) for x in edit.losses]
    return record


def load_edit(path: str):
    """Reload a saved edit as an `EditResult` (usable with models.editing.applied_edit)."""
    import torch
    from models.editing import EditResult

    blob = torch.load(path, map_location="cpu", weights_only=False)
    return EditResult(layer=blob["layer"], position=blob["position"], v_star=blob["v_star"],
                      v_orig=blob["v_orig"], k_vec=blob["k_vec"])


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Fit the rank-one factual edit and save it.")
    ap.add_argument("--config", default=None, help="path to a YAML config file")
    ap.add_argument("--model", default=None, help="model name (default from config)")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--target", type=int, default=None,
                    help="index of a single edit target to fit (default: all)")
    ap.add_argument("--layer", type=int, default=None,
                    help="force the edited MLP layer instead of causal tracing")
    ap.add_argument("--steps", type=int, default=None, dest="edit_steps",
                    help="v* optimization steps")
    ap.add_argument("--lr", type=float, default=None, dest="edit_lr")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, model_name=args.model, device=args.device, seed=args.seed,
                      edit_steps=args.edit_steps, edit_lr=args.edit_lr)
    outputs_dir = ensure_dir(cfg.abs_path(cfg.outputs_dir))
    log = get_logger("train")
    set_seed(cfg.seed)

    targets = load_processed_edit_targets(cfg, logger=log)
    if args.target is not None:
        if not 0 <= args.target < len(targets):
            log.error("--target %d out of range (0..%d)", args.target, len(targets) - 1)
            return 1
        targets = [targets[args.target]]

    try:
        from models.loader import load_model
    except ImportError as exc:
        log.error("%s\nInstall the stack first:  pip install -r requirements.txt", exc)
        return 1
    model = load_model(cfg)

    records, loss_rows = [], []
    for target in targets:
        record = fit_target(model, target, cfg, outputs_dir, layer=args.layer, logger=log)
        for step, loss in enumerate(record.pop("losses")):
            loss_rows.append({"subject": record["subject"], "layer": record["layer"],
                              "step": step, "loss": loss})
        records.append(record)

    if loss_rows:
        losses = pd.DataFrame(loss_rows)
        losses_path = os.path.join(outputs_dir, EDITS_SUBDIR, LOSSES_CSV)
        losses.to_csv(losses_path, index=False)
        log.info("loss curves -> %s", relative_to_repo(losses_path))

        first = records[0]["subject"]
        curve = losses[losses.subject == first]
        figure = line_plot(curve["step"], curve["loss"],
                           os.path.join(outputs_dir, "images", "edit_fit_loss.png"),
                           title=f"v* optimization — {first} ({cfg.model_name})",
                           xlabel="step", ylabel="loss  (-log p(new object) + L2)")
        log.info("loss curve figure -> %s", relative_to_repo(figure))

    log.info("fitted %d edit(s) -> %s", len(records),
             relative_to_repo(os.path.join(outputs_dir, EDITS_SUBDIR)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
