"""Interpretability tools: logit difference, ablation, causal tracing, and the logit lens.

These implement Experiments 1-3 (observation and intervention). They locate *which*
components drive a behavior; models/editing.py then acts on what they find.
"""
from __future__ import annotations
from typing import Optional

import numpy as np
import torch

from .config import ProjectConfig
from .utils import first_token, subject_span


# --------------------------------------------------------------------------- #
# Experiment 1 - baseline logit difference
# --------------------------------------------------------------------------- #
def logit_diff(model, prompt: str, correct: str, incorrect: str) -> float:
    """logit(correct) - logit(incorrect) for the next token after `prompt`.

    Positive means the model prefers the correct answer. This is the core behavioral
    metric used everywhere else.
    """
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        logits = model(tokens)[0, -1]
    return (logits[first_token(model, correct)] - logits[first_token(model, incorrect)]).item()


def suite_logit_diffs(model, records: list[dict]) -> list[dict]:
    """Baseline logit diff for every record that has correct/incorrect answers."""
    out = []
    for r in records:
        if not r.get("incorrect"):
            continue
        out.append({**r, "logit_diff": logit_diff(model, r["prompt"], r["correct"], r["incorrect"])})
    return out


# --------------------------------------------------------------------------- #
# Experiment 3a - component ablation
# --------------------------------------------------------------------------- #
def ablate_component(model, prompt: str, correct: str, incorrect: str, component: dict) -> float:
    """Return the logit diff after zero-ablating one attention head or one MLP.

    component = {"type": "head", "layer": L, "head": H}  or  {"type": "mlp", "layer": L}
    """
    ctype, layer = component["type"], component["layer"]

    if ctype == "head":
        head = component["head"]
        hook_name = f"blocks.{layer}.attn.hook_z"

        def hook(value, hook):          # value: [batch, pos, n_heads, d_head]
            value[:, :, head, :] = 0.0
            return value
    elif ctype == "mlp":
        hook_name = f"blocks.{layer}.hook_mlp_out"

        def hook(value, hook):          # value: [batch, pos, d_model]
            value[:] = 0.0
            return value
    else:
        raise ValueError(f"unknown component type {ctype!r}")

    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook)])[0, -1]
    return (logits[first_token(model, correct)] - logits[first_token(model, incorrect)]).item()


def head_ablation_scan(model, prompt: str, correct: str, incorrect: str) -> np.ndarray:
    """Grid [n_layers, n_heads] of the DROP in logit diff caused by ablating each head.

    Larger drop => that head contributes more to the correct prediction.
    """
    base = logit_diff(model, prompt, correct, incorrect)
    grid = np.zeros((model.cfg.n_layers, model.cfg.n_heads))
    for l in range(model.cfg.n_layers):
        for h in range(model.cfg.n_heads):
            ld = ablate_component(model, prompt, correct, incorrect,
                                  {"type": "head", "layer": l, "head": h})
            grid[l, h] = base - ld
    return grid


# --------------------------------------------------------------------------- #
# Experiment 3b - causal tracing (ROME-style)
# --------------------------------------------------------------------------- #
def causal_trace(model, prompt: str, subject: str, answer: str,
                 config: Optional[ProjectConfig] = None):
    """Corrupt the subject, patch clean resid_post back layer-by-layer, measure recovery.

    Returns (recovery[n_layers, n_pos], subject_last_pos, str_tokens).
    """
    cfg = config or ProjectConfig()
    tokens = model.to_tokens(prompt)
    n_pos = tokens.shape[1]
    ans_tok = first_token(model, answer)
    s0, s1 = subject_span(model, prompt, subject)
    subj_pos = list(range(s0, s1 + 1))

    _, clean_cache = model.run_with_cache(tokens)
    with torch.no_grad():
        clean_score = torch.log_softmax(model(tokens)[0, -1], dim=-1)[ans_tok].item()

    emb_std = model.W_E.std().item()
    g = torch.Generator(device=tokens.device).manual_seed(cfg.seed)
    noise = torch.randn(len(subj_pos), model.cfg.d_model,
                        generator=g, device=tokens.device) * emb_std * cfg.noise_scale

    def corrupt_hook(value, hook):
        for j, p in enumerate(subj_pos):
            value[0, p] = value[0, p] + noise[j]
        return value

    with torch.no_grad():
        corr_logits = model.run_with_hooks(tokens, fwd_hooks=[("hook_embed", corrupt_hook)])
        corr_score = torch.log_softmax(corr_logits[0, -1], dim=-1)[ans_tok].item()

    grid = np.zeros((model.cfg.n_layers, n_pos))
    for l in range(model.cfg.n_layers):
        clean_resid = clean_cache[f"blocks.{l}.hook_resid_post"]
        for p in range(n_pos):
            def patch_hook(value, hook, l=l, p=p, clean_resid=clean_resid):
                value[0, p] = clean_resid[0, p]
                return value
            with torch.no_grad():
                logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[("hook_embed", corrupt_hook),
                               (f"blocks.{l}.hook_resid_post", patch_hook)])
            grid[l, p] = torch.log_softmax(logits[0, -1], dim=-1)[ans_tok].item()

    recovery = (grid - corr_score) / (clean_score - corr_score + 1e-9)
    return recovery, s1, model.to_str_tokens(prompt, prepend_bos=True)


def best_edit_layer(recovery: np.ndarray, subject_last_pos: int, window: int = 1) -> int:
    """Layer with maximum recovery at the subject's last token.

    `window` = 1 reproduces the plain per-layer argmax. Larger odd windows average the
    recovery profile over `window` neighbouring layers before taking the argmax, which is
    the small-model analogue of the multi-layer window ROME traces over (Meng et al. 2022,
    §3.1): in a 12-layer model a single layer's restoration score is noisy enough that the
    argmax can land on a layer whose edit does nothing, while the *region* of layers that
    carries the fact is stable. See `scripts/run_layer_sweep.py` for the measurement that
    sets the default.
    """
    profile = recovery[:, subject_last_pos]
    if window > 1:
        kernel = np.ones(window) / window
        profile = np.convolve(profile, kernel, mode="same")
    return int(np.argmax(profile))


# --------------------------------------------------------------------------- #
# Experiment 2 - logit lens
# --------------------------------------------------------------------------- #
def logit_lens(model, prompt: str, answer: str) -> np.ndarray:
    """Per-layer log-prob of `answer` when each layer's resid_post is decoded directly.

    Shows the prediction taking shape across depth.
    """
    tokens = model.to_tokens(prompt)
    ans_tok = first_token(model, answer)
    _, cache = model.run_with_cache(tokens)
    out = np.zeros(model.cfg.n_layers)
    for l in range(model.cfg.n_layers):
        resid = cache[f"blocks.{l}.hook_resid_post"][:, -1]
        resid = model.ln_final(resid)
        logits = model.unembed(resid)[0]
        out[l] = torch.log_softmax(logits, dim=-1)[ans_tok].item()
    return out
