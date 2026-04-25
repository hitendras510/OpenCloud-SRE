"""
utils/logger.py
================
Centralized structured logger for OpenCloud-SRE.
Provides a get_logger() factory with consistent formatting,
W&B artifact upload support, and JSONL incident log writing.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


# ── ANSI colour codes (disabled automatically when not a TTY) ─────────────────
_COLOURS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
}
_RESET = "\033[0m"
_USE_COLOUR = sys.stdout.isatty()


class _ColourFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if _USE_COLOUR:
            colour = _COLOURS.get(record.levelname, "")
            return f"{colour}{msg}{_RESET}"
        return msg


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a named logger with consistent formatting.
    Idempotent – calling twice with the same name returns the same logger.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.
    level : int
        Logging level (default INFO).
    """
    logger = logging.getLogger(name)
    if logger.handlers:           # already configured
        return logger
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_ColourFormatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# ── JSONL incident writer ─────────────────────────────────────────────────────

class IncidentLogger:
    """
    Appends structured incident records to a JSONL file.
    Used by the Streamlit UI and RL trainer to persist episode history.

    Example
    -------
    >>> ilog = IncidentLogger("data/eval_logs.jsonl")
    >>> ilog.log(action="scale_out", slo_before=0.32, slo_after=0.71)
    """

    def __init__(self, path: str = "data/eval_logs.jsonl") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._logger = get_logger(f"IncidentLogger[{self._path.name}]")

    def log(self, **fields: Any) -> None:
        """Append one JSON record with an auto-generated UTC timestamp."""
        record: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **fields,
        }
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._logger.debug("Logged: %s", record)

    def tail(self, n: int = 10) -> list[Dict[str, Any]]:
        """Return the last *n* records from the log file."""
        if not self._path.exists():
            return []
        lines = self._path.read_text(encoding="utf-8").strip().splitlines()
        return [json.loads(l) for l in lines[-n:] if l.strip()]

    def clear(self) -> None:
        """Truncate the log file."""
        self._path.write_text("", encoding="utf-8")
