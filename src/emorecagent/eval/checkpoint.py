"""Per-row eval checkpoints for resumable long experiments."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from ..data.types import Interaction

CHECKPOINT_VERSION = 1


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be loaded or does not match the run."""


def row_key(interaction: Interaction) -> str:
    """Stable id for one test interaction (user, held-out item, query time)."""
    return f"{interaction.user_id}\t{interaction.item}\t{interaction.timestamp}"


def default_checkpoint_stem(out_path: str | Path) -> Path:
    """``results/foo.json`` → ``results/foo.checkpoint``."""
    out = Path(out_path)
    return out.parent / f"{out.stem}.checkpoint"


def pass_checkpoint_path(stem: Path, *, sampled: bool) -> Path:
    label = "sampled" if sampled else "full"
    return stem.parent / f"{stem.name}.{label}.jsonl"


def build_fingerprint(
    *,
    method: str,
    seed: int,
    protocol: str,
    n_negatives: int | None,
    verified_only: bool,
    cumulative_history: bool,
    max_test_rows: int | None,
    eval_protocol: str,
    aggregation: str,
    k_values: list[int],
    hr_avg_k: tuple[int, ...],
    parallel_workers: int,
    llm_batch: bool = False,
    batch_size: int = 12,
) -> dict[str, Any]:
    return {
        "version": CHECKPOINT_VERSION,
        "method": method,
        "seed": seed,
        "protocol": protocol,
        "n_negatives": n_negatives,
        "verified_only": verified_only,
        "cumulative_history": cumulative_history,
        "max_test_rows": max_test_rows,
        "eval_protocol": eval_protocol,
        "aggregation": aggregation,
        "k_values": list(k_values),
        "hr_avg_k": list(hr_avg_k),
        "parallel_workers": parallel_workers,
        "llm_batch": llm_batch,
        "batch_size": batch_size,
    }


class EvalCheckpoint:
    """Append-only JSONL cache of per-row rankings."""

    def __init__(
        self,
        path: Path,
        fingerprint: dict[str, Any],
        *,
        resume: bool = True,
        fresh: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        self.path = path
        self.meta_path = path.with_name(path.stem + ".meta.json")
        self._fingerprint = fingerprint
        self._logger = logger
        self._lock = threading.Lock()
        self._rows: dict[str, list[str]] = {}
        self._fh = None

        if fresh:
            self._remove_files()

        if resume and self.path.exists():
            self._load_existing()
            self._validate_meta()
            if self._logger and self._rows:
                self._logger.info(
                    "resume checkpoint %s: %s rows cached",
                    self.path.name,
                    f"{len(self._rows):,}",
                )
        else:
            if self.path.exists():
                self._remove_files()
            self._write_meta()

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def _remove_files(self) -> None:
        self.path.unlink(missing_ok=True)
        self.meta_path.unlink(missing_ok=True)

    def _write_meta(self) -> None:
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(
            json.dumps(self._fingerprint, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _validate_meta(self) -> None:
        if not self.meta_path.exists():
            raise CheckpointError(
                f"Checkpoint data exists without metadata: {self.path}. "
                "Delete the checkpoint or pass --fresh-checkpoint."
            )
        stored = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if stored != self._fingerprint:
            raise CheckpointError(
                "Checkpoint fingerprint does not match this run "
                f"({self.meta_path}). Use --fresh-checkpoint to restart."
            )

    def _load_existing(self) -> None:
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                self._rows[str(payload["row_key"])] = list(payload["ranked"])

    def has(self, interaction: Interaction) -> bool:
        return row_key(interaction) in self._rows

    def get(self, interaction: Interaction) -> list[str]:
        return list(self._rows[row_key(interaction)])

    def __len__(self) -> int:
        return len(self._rows)

    def save(self, interaction: Interaction, ranked: list[str]) -> None:
        key = row_key(interaction)
        with self._lock:
            if key in self._rows:
                return
            self._rows[key] = list(ranked)
            assert self._fh is not None
            self._fh.write(
                json.dumps({"row_key": key, "ranked": ranked}, ensure_ascii=False)
                + "\n"
            )
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
