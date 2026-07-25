"""Plotting helpers for the analysis figures.

Matplotlib is imported lazily and forced onto the non-interactive "Agg" backend, so these
functions work identically in a headless container, in CI, and inside a notebook.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

from .helpers import ensure_parent_dir

# A small, colour-blind-safe palette reused across every figure in the project.
PALETTE = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]


def _plt():
    """Return pyplot with the Agg backend selected (headless-safe)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def bar_chart(values: Mapping[str, float], path: str, title: str = "",
              ylabel: str = "", ylim: Optional[tuple[float, float]] = (0.0, 1.0),
              annotate: bool = True) -> str:
    """Save a labelled bar chart of `values` (name -> height) to `path`."""
    plt = _plt()
    names, heights = list(values.keys()), list(values.values())
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, heights, color=colors)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if annotate:
        span = (ylim[1] - ylim[0]) if ylim else (max(heights) or 1.0)
        for i, v in enumerate(heights):
            ax.text(i, v + 0.02 * span, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()

    ensure_parent_dir(path)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def grouped_bar_chart(groups: Mapping[str, Mapping[str, float]], path: str,
                      title: str = "", ylabel: str = "",
                      ylim: Optional[tuple[float, float]] = None,
                      annotate: bool = True) -> str:
    """Save a grouped bar chart: {group_label: {series_name: value}}.

    Bars are annotated by default because a zero-valued bar is otherwise invisible —
    and in this project a score of exactly 0.00 is a result worth reading.
    """
    plt = _plt()
    group_names = list(groups.keys())
    series_names = list(next(iter(groups.values()), {}).keys())
    n_series = max(len(series_names), 1)
    width = 0.8 / n_series

    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(group_names)), 4))
    for s, series in enumerate(series_names):
        xs = [g + (s - (n_series - 1) / 2) * width for g in range(len(group_names))]
        ys = [groups[g].get(series, 0.0) for g in group_names]
        ax.bar(xs, ys, width=width, label=series, color=PALETTE[s % len(PALETTE)])
        if annotate:
            offset = 0.02 * ((ylim[1] - ylim[0]) if ylim else (max(ys) or 1.0))
            for x, y in zip(xs, ys):
                ax.text(x, y + offset, f"{y:.2f}", ha="center", fontsize=7)

    ax.set_xticks(range(len(group_names)))
    ax.set_xticklabels(group_names, rotation=0, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    if series_names:
        ax.legend(fontsize=8)
    fig.tight_layout()

    ensure_parent_dir(path)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def line_plot(x: Sequence[float], y: Sequence[float], path: str, title: str = "",
              xlabel: str = "", ylabel: str = "") -> str:
    """Save a single-series line plot (used for the edit-fitting loss curve)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(x), list(y), color=PALETTE[0], linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    ensure_parent_dir(path)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
