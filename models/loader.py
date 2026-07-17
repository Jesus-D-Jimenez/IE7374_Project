"""Load a pretrained model as a TransformerLens HookedTransformer.

We do not train or fine-tune any weights. The model is loaded exactly as released; the only
weight change anywhere in the project is the deliberate rank-one edit in models/editing.py.
"""
from __future__ import annotations
from functools import lru_cache

from .config import ProjectConfig


@lru_cache(maxsize=4)
def _cached_load(model_name: str, device: str):
    from transformer_lens import HookedTransformer  # imported lazily so config/tests need no torch
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()
    return model


def load_model(config: ProjectConfig):
    """Return a HookedTransformer for `config.model_name` on the resolved device."""
    device = config.resolved_device()
    model = _cached_load(config.model_name, device)
    print(
        f"Loaded {config.model_name} on {device} | "
        f"n_layers={model.cfg.n_layers}, d_model={model.cfg.d_model}, d_mlp={model.cfg.d_mlp}"
    )
    return model
