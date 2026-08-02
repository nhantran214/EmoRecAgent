"""File + console logging for CLI pipeline scripts."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


def default_log_path(task: str, log_dir: str | Path = "logs") -> Path:
    """Timestamped log path: ``logs/{task}_YYYYMMDD_HHMMSS.log``."""
    log_root = Path(log_dir)
    log_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_root / f"{task}_{stamp}.log"


def configure_run_logging(
    task: str,
    *,
    log_file: str | Path | None = None,
    log_dir: str | Path = "logs",
    level: int = logging.INFO,
) -> tuple[logging.Logger, Path]:
    """Attach file + stdout handlers; return logger and resolved log path."""
    path = Path(log_file) if log_file else default_log_path(task, log_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(f"emorecagent.{task}")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    return logger, path
