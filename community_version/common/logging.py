"""Central logging helpers.

This module keeps logger configuration in one place so every script and module
prints consistent, grep-friendly messages.
"""
from __future__ import annotations

import logging
from typing import Optional


_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _configure_root_logger(logger: logging.Logger) -> None:
    """Attach a simple console handler to ``logger`` if none exist."""

    if logger.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger with concise console formatting.

    The first call sets up formatting; subsequent calls reuse the same logger
    instance without reattaching handlers.
    """

    logger = logging.getLogger(name if name else __name__)
    _configure_root_logger(logger)
    logger.propagate = False
    return logger


__all__ = ["get_logger"]
