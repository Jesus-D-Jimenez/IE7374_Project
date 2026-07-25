"""Shared utility functions: paths, logging, seeding, and file I/O.

Everything here is deliberately dependency-light (standard library only, plus an
optional torch import for seeding) so the data pipeline, the tests, and the notebooks
can all use it whether or not the deep-learning stack is installed.

Model-specific token/probability helpers live in `models/utils.py`; plotting lives in
`utils/visualization.py`. This module holds the plumbing the whole repo shares.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
from typing import Any, Iterable, Optional

# Repository root = parent of the directory holding this file.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_LOG_DATEFMT = "%H:%M:%S"


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def repo_path(*parts: str) -> str:
    """Absolute path inside the repository, e.g. repo_path('outputs', 'samples.txt').

    Absolute inputs are returned unchanged, so config values may be either form.
    """
    if parts and os.path.isabs(parts[0]):
        return os.path.join(*parts)
    return os.path.join(REPO_ROOT, *parts)


def ensure_dir(path: str) -> str:
    """Create `path` (a directory) if needed and return it."""
    os.makedirs(path, exist_ok=True)
    return path


def ensure_parent_dir(path: str) -> str:
    """Create the parent directory of a file path and return the path."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


def relative_to_repo(path: str) -> str:
    """Repo-relative, forward-slashed path — used for readable log/manifest entries."""
    try:
        return os.path.relpath(path, REPO_ROOT).replace(os.sep, "/")
    except ValueError:                      # different drive on Windows
        return path


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str = "pipeline", level: int = logging.INFO,
               log_file: Optional[str] = None) -> logging.Logger:
    """Return a console logger (optionally also writing to `log_file`).

    Handlers are only attached once, so repeated calls from notebooks or nested
    modules do not duplicate every line.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(logging.Formatter(_LOG_FORMAT, _LOG_DATEFMT))
        logger.addHandler(stream)

    if log_file:
        log_file = os.path.abspath(log_file)
        already = any(getattr(h, "baseFilename", None) == log_file for h in logger.handlers)
        if not already:
            ensure_parent_dir(log_file)
            file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _LOG_DATEFMT))
            logger.addHandler(file_handler)

    return logger


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> int:
    """Seed Python, NumPy, and torch (when installed). Returns the seed for logging."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed


# --------------------------------------------------------------------------- #
# File I/O — JSON / JSONL / text
# --------------------------------------------------------------------------- #
def read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any, indent: int = 2) -> str:
    """Write JSON with LF newlines so committed files are identical on every OS."""
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
        f.write("\n")
    return path


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: Iterable[dict]) -> str:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def write_text(path: str, text: str) -> str:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return path


def file_sha256(path: str, chunk_size: int = 65536) -> str:
    """SHA-256 of a file, used in the dataset manifest to pin the raw inputs."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Small formatting helpers
# --------------------------------------------------------------------------- #
def one_line(text: str, limit: int = 300) -> str:
    """Collapse newlines and clip, so generations stay readable in logs and tables."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def banner(title: str, width: int = 78, char: str = "=") -> str:
    """A section header for the human-readable samples file."""
    return f"{char * width}\n{title}\n{char * width}"
