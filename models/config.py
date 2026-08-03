"""Typed configuration loaded from configs/model_config.yaml (with CLI overrides)."""
from __future__ import annotations
import os
from dataclasses import dataclass, replace, fields
from typing import Optional

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "configs", "model_config.yaml")


@dataclass
class ProjectConfig:
    """All experiment hyperparameters in one place."""
    model_name: str = "pythia-160m"
    device: str = "auto"                 # auto | cpu | cuda
    seed: int = 0

    data_dir: str = "data"
    processed_dir: str = "data/processed"
    outputs_dir: str = "outputs"
    results_dir: str = "results"

    n_samples: int = 10                  # behavioral prompts per inference run
    samples_per_suite: int = 4           # cap per behavior, keeps the batch balanced
    top_k: int = 5                       # top-k next tokens recorded per prompt

    noise_scale: float = 3.0             # causal-tracing subject corruption

    edit_lr: float = 0.5                 # v* optimization
    edit_steps: int = 25
    edit_kl_weight: float = 0.0625
    edit_layer: str | int = "auto"       # "auto" -> causal-trace peak
    edit_layer_window: int = 1           # layers averaged before the peak is taken (1 = raw argmax)

    gen_max_new_tokens: int = 30
    gen_do_sample: bool = False

    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def abs_path(self, sub: str) -> str:
        """Resolve a config path (data_dir/results_dir) against the repo root."""
        return sub if os.path.isabs(sub) else os.path.join(REPO_ROOT, sub)


def load_config(path: Optional[str] = None, **overrides) -> ProjectConfig:
    """Load ProjectConfig from YAML, then apply keyword overrides (e.g. from argparse).

    Overrides whose value is None are ignored, so `argparse` defaults of None mean
    "leave the YAML value alone".
    """
    path = path or DEFAULT_CONFIG
    data = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    elif path != DEFAULT_CONFIG:
        raise FileNotFoundError(f"config file not found: {path}")

    valid = {f.name for f in fields(ProjectConfig)}
    unknown = set(data) - valid
    if unknown:                                  # typo in the YAML should not fail silently
        print(f"[config] ignoring unknown key(s) in {os.path.basename(path)}: {sorted(unknown)}")
    data = {k: v for k, v in data.items() if k in valid}
    if "split_ratios" in data:
        data["split_ratios"] = tuple(data["split_ratios"])

    cfg = ProjectConfig(**data)
    clean = {k: v for k, v in overrides.items() if v is not None and k in valid}
    return replace(cfg, **clean) if clean else cfg


if __name__ == "__main__":
    print(load_config())
