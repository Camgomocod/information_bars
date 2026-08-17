"""
logging_setup.py — Structured logging for trading-core.

Migrates pipeline modules from ad-hoc ``print()`` to structured ``logging``
entries tagged with symbol/period/stage/run_id. Existing scripts still use
print for human CLI output; new pipeline modules use this.

Run IDs make runs traceable end-to-end (logs, DB, parquet metadata).
"""

from __future__ import annotations

import logging
import sys
import uuid

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure a single console handler on the root logger (idempotent)."""
    root = logging.getLogger()
    if any(getattr(h, "_trading_core", False) for h in root.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handler._trading_core = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def make_run_logger(name: str, run_id: str | None = None) -> tuple[logging.Logger, str]:
    """Return (logger, run_id) so every message can carry the run id."""
    run_id = run_id or new_run_id()
    logger = get_logger(name)
    return logger, run_id
