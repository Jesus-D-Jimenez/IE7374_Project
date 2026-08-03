"""Evaluation metrics for both halves of the project.

Interpretability side (Experiments 1-3): logit difference and selectivity.
Generative side (Experiment 4): efficacy, generalization, specificity, and fluency of the
generated text under a ROME-style edit (Meng et al. 2022 evaluation framework).
"""
from __future__ import annotations

import pandas as pd

from .config import ProjectConfig
from .editing import optimize_v_star, apply_rank_one_edit, restore_weights
from .generation import generate
from .interpret import causal_trace, best_edit_layer
from .utils import fluency, top_prediction


# --------------------------------------------------------------------------- #
# Interpretability metric
# --------------------------------------------------------------------------- #
def selectivity_score(target_drop: float, control_drop: float) -> float:
    """How specific an ablation is: 1.0 = hurts only the targeted behavior.

    target_drop / control_drop are drops in logit diff on targeted vs. unrelated prompts.
    """
    denom = abs(target_drop) + abs(control_drop) + 1e-9
    return (target_drop - control_drop) / denom


# --------------------------------------------------------------------------- #
# Generation metrics
# --------------------------------------------------------------------------- #
def snapshot_generations(model, target: dict, config: ProjectConfig) -> pd.DataFrame:
    """Generate + score every evaluation prompt for one target at the current weights.

    Public because the layer sweep (`scripts/run_layer_sweep.py`) reuses one pre-edit
    snapshot across every candidate layer instead of regenerating it once per layer.
    """
    true, new = target["true"], target["new"]
    rows = []

    gen = generate(model, target["prompt"], config)
    rows.append(dict(kind="efficacy", prompt=target["prompt"],
                     pred=top_prediction(model, target["prompt"], 1)[0][0].strip(),
                     says_new=new.lower() in gen.lower(),
                     says_true=true.lower() in gen.lower(),
                     keeps_answer=None, fluency=fluency(model, gen), generation=gen))

    for pp in target.get("paraphrases", []):
        gen = generate(model, pp, config)
        rows.append(dict(kind="generalization", prompt=pp,
                         pred=top_prediction(model, pp, 1)[0][0].strip(),
                         says_new=new.lower() in gen.lower(),
                         says_true=true.lower() in gen.lower(),
                         keeps_answer=None, fluency=fluency(model, gen), generation=gen))

    for np_prompt, np_ans in target.get("neighborhood", []):
        gen = generate(model, np_prompt, config)
        rows.append(dict(kind="specificity", prompt=np_prompt,
                         pred=top_prediction(model, np_prompt, 1)[0][0].strip(),
                         says_new=None, says_true=None,
                         keeps_answer=np_ans.lower() in gen.lower(),
                         fluency=fluency(model, gen), generation=gen))
    return pd.DataFrame(rows)


def evaluate_edit(model, target: dict, config: ProjectConfig | None = None,
                  layer: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Run the full before/after edit evaluation for one target.

    Returns (per-prompt DataFrame with a 'phase' column, summary-scores dict).
    """
    cfg = config or ProjectConfig()

    # Choose the edit layer via causal tracing unless one is given.
    if layer is None:
        if cfg.edit_layer != "auto":
            layer = int(cfg.edit_layer)
        else:
            recovery, subj_last, _ = causal_trace(model, target["prompt"],
                                                  target["subject"], target["true"], cfg)
            layer = best_edit_layer(recovery, subj_last, window=cfg.edit_layer_window)

    restore_weights(model)
    before = snapshot_generations(model, target, cfg)
    before["phase"] = "before"

    edit = optimize_v_star(model, target["prompt"], target["subject"], target["new"], layer, cfg)
    apply_rank_one_edit(model, edit)
    after = snapshot_generations(model, target, cfg)
    after["phase"] = "after"
    restore_weights(model)

    df = pd.concat([before, after], ignore_index=True)
    df["layer"] = layer
    scores = summarize_scores(df)
    scores["layer"] = layer
    return df, scores


def summarize_scores(df: pd.DataFrame) -> dict:
    """Aggregate one target's before/after table into the four ROME-style scores.

    efficacy       - does the edited prompt now state the new object?
    generalization - do paraphrases state it too?
    specificity    - do unrelated ("neighbourhood") prompts still state their own answer?
    specificity_pred_preserved - stricter and fairer companion to `specificity`: the share
                     of neighbourhood prompts whose top-1 next token is unchanged by the
                     edit. Plain `specificity` scores 0 whenever the base model never knew
                     the neighbourhood fact, which confuses "the edit broke it" with "the
                     model never had it"; comparing against the pre-edit prediction
                     measures only the damage the edit itself caused.
    """
    after = df[df.phase == "after"]
    eff = after[after.kind == "efficacy"]["says_new"].mean()
    gen = after[after.kind == "generalization"]["says_new"].mean()
    spec = after[after.kind == "specificity"]["keeps_answer"].mean()

    neighborhood = df[df.kind == "specificity"]
    before_pred = neighborhood[neighborhood.phase == "before"].set_index("prompt")["pred"]
    after_pred = neighborhood[neighborhood.phase == "after"].set_index("prompt")["pred"]
    shared = before_pred.index.intersection(after_pred.index)
    preserved = float((before_pred[shared] == after_pred[shared]).mean()) if len(shared) else float("nan")

    return dict(
        efficacy=round(float(eff), 3),
        generalization=round(float(gen), 3),
        specificity=round(float(spec), 3),
        specificity_pred_preserved=round(preserved, 3),
        fluency_before=round(float(df[df.phase == "before"]["fluency"].mean()), 3),
        fluency_after=round(float(after["fluency"].mean()), 3),
    )
