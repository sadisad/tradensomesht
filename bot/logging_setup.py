"""Centralised logging setup.

One stream handler + one rotating-ish file handler. Keep it boring.
"""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Any, Dict


_CONFIGURED = False


def setup_logging(cfg: Dict[str, Any]) -> logging.Logger:
    global _CONFIGURED
    log_cfg = cfg.get("logging", {}) or {}
    level_name = str(log_cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    file_path = log_cfg.get("file") or "data/bot.log"

    log_path = Path(file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    if _CONFIGURED:
        root.setLevel(level)
        return root

    root.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)

    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=2_000_000, backupCount=5, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)

    _CONFIGURED = True
    root.debug("logging configured (level=%s file=%s)", level_name, log_path)
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
