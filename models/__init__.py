"""Models package: model loading, interpretability, ROME-style editing, and metrics.

Typical use:
    from models import load_config, load_model, evaluate_edit
    from data import load_edit_targets

    cfg = load_config()
    model = load_model(cfg)
    df, scores = evaluate_edit(model, load_edit_targets()[0], cfg)

The configuration layer (`load_config`, `ProjectConfig`) has no heavy dependencies and
imports even where torch is not installed; the interpretability/editing API requires torch
and transformer_lens.
"""
from .config import ProjectConfig, load_config

__all__ = ["ProjectConfig", "load_config"]

try:  # full API is available once torch + transformer_lens are installed
    from .loader import load_model
    from .generation import generate
    from .utils import (
        first_token, next_token_logprobs, top_prediction, answer_logprob,
        subject_span, ngram_entropy, fluency,
    )
    from .interpret import (
        logit_diff, suite_logit_diffs, ablate_component, head_ablation_scan,
        causal_trace, best_edit_layer, logit_lens,
    )
    from .editing import (
        EditResult, optimize_v_star, apply_rank_one_edit, restore_weights, applied_edit,
    )
    from .metrics import evaluate_edit, summarize_scores, selectivity_score

    __all__ += [
        "load_model", "generate",
        "first_token", "next_token_logprobs", "top_prediction", "answer_logprob",
        "subject_span", "ngram_entropy", "fluency",
        "logit_diff", "suite_logit_diffs", "ablate_component", "head_ablation_scan",
        "causal_trace", "best_edit_layer", "logit_lens",
        "EditResult", "optimize_v_star", "apply_rank_one_edit", "restore_weights", "applied_edit",
        "evaluate_edit", "summarize_scores", "selectivity_score",
    ]
except ImportError as _e:  # torch / transformer_lens not present
    _TORCH_IMPORT_ERROR = _e
