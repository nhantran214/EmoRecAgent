"""File + console logging for RecBole TiSASRec experiments."""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator


def default_log_path(task: str, log_dir: str | Path) -> Path:
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
    path = Path(log_file) if log_file else default_log_path(task, log_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(f"recbole_tisasrec.{task}")
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


class _TeeStream:
    def __init__(self, stream, log_handle) -> None:
        self._stream = stream
        self._log = log_handle

    def write(self, data: str) -> None:
        self._stream.write(data)
        self._log.write(data)
        self._log.flush()

    def flush(self) -> None:
        self._stream.flush()
        self._log.flush()

    def fileno(self):
        return self._stream.fileno()

    def isatty(self) -> bool:
        return self._stream.isatty()


@contextmanager
def tee_stdout_stderr(log_path: Path) -> Iterator[None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n{'=' * 60}\n")
        fh.write(f"RecBole-TiSASRec run started at {datetime.now().isoformat()}\n")
        fh.write(f"{'=' * 60}\n")
        fh.flush()
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = _TeeStream(old_out, fh)  # type: ignore[assignment]
        sys.stderr = _TeeStream(old_err, fh)  # type: ignore[assignment]
        try:
            yield
        finally:
            sys.stdout = old_out
            sys.stderr = old_err
            fh.write(f"\n{'=' * 60}\n")
            fh.write(f"RecBole-TiSASRec run finished at {datetime.now().isoformat()}\n")
            fh.write(f"{'=' * 60}\n")
