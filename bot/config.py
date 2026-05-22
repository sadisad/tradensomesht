"""Configuration loader.

Reads ``config.yaml`` (or a path supplied via CONFIG_PATH env var) and exposes
a typed-ish dict-of-dicts. Keeps things simple on purpose: trading config
should be human-editable, not buried in code.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: str | os.PathLike[str] | None = None) -> Dict[str, Any]:
    """Load YAML config from disk.

    Resolution order:
      1. explicit ``path`` argument
      2. ``CONFIG_PATH`` env var
      3. ``<project_root>/config.yaml``
    """
    cfg_path = Path(path or os.environ.get("CONFIG_PATH") or _DEFAULT_PATH)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping")
    cfg["_path"] = str(cfg_path)
    return cfg


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent
