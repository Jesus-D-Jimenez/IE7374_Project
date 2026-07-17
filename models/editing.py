"""ROME-style rank-one factual editing.

Given a fact-storing MLP (localized by causal tracing), we:
  1. optimize a target value v* at the subject's last token that makes the model predict
     the NEW object (models.editing.optimize_v_star);
  2. solve the closed-form rank-one update to W_out that maps the layer's key k to v*
     (apply_rank_one_edit).

This uses the identity-covariance simplification of Meng et al. (2022): the full ROME
update whitens k by a second-moment matrix C estimated from a corpus. Replacing C with the
identity keeps the method self-contained; the generalization/specificity metrics in
models/metrics.py reveal the quality trade-off that simplification makes.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch

from .config import ProjectConfig
from .utils import first_token, subject_span


@dataclass
class EditResult:
    layer: int
    position: int
    v_star: torch.Tensor
    v_orig: torch.Tensor
    k_vec: torch.Tensor


def optimize_v_star(model, prompt: str, subject: str, new_object: str,
                    layer: int, config: ProjectConfig | None = None) -> EditResult:
    """Find the MLP-output vector at the subject's last token that elicits `new_object`."""
    cfg = config or ProjectConfig()
    tokens = model.to_tokens(prompt)
    pos = subject_span(model, prompt, subject)[1]
    tgt_tok = first_token(model, new_object)

    _, clean_cache = model.run_with_cache(tokens)
    v_orig = clean_cache[f"blocks.{layer}.hook_mlp_out"][0, pos].detach()
    k_vec = clean_cache[f"blocks.{layer}.mlp.hook_post"][0, pos].detach()

    delta = torch.zeros(model.cfg.d_model, device=tokens.device, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=cfg.edit_lr)

    def add_delta(value, hook):
        value[0, pos] = value[0, pos] + delta
        return value

    for step in range(cfg.edit_steps):
        opt.zero_grad()
        logits = model.run_with_hooks(
            tokens, fwd_hooks=[(f"blocks.{layer}.hook_mlp_out", add_delta)])
        logp = torch.log_softmax(logits[0, -1], dim=-1)
        loss = -logp[tgt_tok] + cfg.edit_kl_weight * delta.norm() ** 2
        loss.backward()
        opt.step()

    return EditResult(layer=layer, position=pos,
                      v_star=(v_orig + delta.detach()), v_orig=v_orig, k_vec=k_vec)


# Registry of original weights so edits can be undone.
_ORIGINAL_W_OUT: dict[tuple[int, int], torch.Tensor] = {}


def apply_rank_one_edit(model, edit: EditResult) -> None:
    """Add a rank-one update to blocks[layer].mlp.W_out so that k -> v*."""
    layer = edit.layer
    W = model.blocks[layer].mlp.W_out          # [d_mlp, d_model]
    key = (id(model), layer)
    if key not in _ORIGINAL_W_OUT:
        _ORIGINAL_W_OUT[key] = W.detach().clone()
    r = (edit.v_star - edit.v_orig)            # residual to inject (== optimized delta)
    k_unit = edit.k_vec / (edit.k_vec @ edit.k_vec + 1e-9)
    with torch.no_grad():
        W += torch.outer(k_unit, r)


def restore_weights(model) -> None:
    """Undo all edits applied to `model`."""
    with torch.no_grad():
        for (mid, layer), W0 in list(_ORIGINAL_W_OUT.items()):
            if mid == id(model):
                model.blocks[layer].mlp.W_out.copy_(W0)
                del _ORIGINAL_W_OUT[(mid, layer)]


class applied_edit:
    """Context manager: apply an edit, run a block, then restore the weights.

    with applied_edit(model, edit):
        ... generate under the edit ...
    """
    def __init__(self, model, edit: EditResult):
        self.model, self.edit = model, edit

    def __enter__(self):
        apply_rank_one_edit(self.model, self.edit)
        return self.model

    def __exit__(self, *exc):
        restore_weights(self.model)
        return False
