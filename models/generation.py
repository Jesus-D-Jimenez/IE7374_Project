"""Text generation wrapper with reproducible decoding."""
from __future__ import annotations

import torch

from .config import ProjectConfig


def generate(model, prompt: str, config: ProjectConfig | None = None,
             max_new_tokens: int | None = None) -> str:
    """Generate a continuation for `prompt`.

    Defaults to greedy decoding so before/after-edit generations are directly comparable.
    """
    cfg = config or ProjectConfig()
    max_new = max_new_tokens or cfg.gen_max_new_tokens
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
    return model.generate(
        prompt,
        max_new_tokens=max_new,
        do_sample=cfg.gen_do_sample,
        verbose=False,
    )
