"""Export ABSA target reviews scoped to the processed train split."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..data.review_index import (
    TrainAbsaScope,
    build_train_scope,
    review_id_from_row,
)
from ..data.types import Interaction
from ..eval.runner import load_split_jsonl

# Re-export for backward compatibility.
__all__ = [
    "AbsaTargetExportStats",
    "TrainAbsaScope",
    "build_train_scope",
    "export_absa_targets",
]


@dataclass(frozen=True, slots=True)
class AbsaTargetExportStats:
    n_train_interactions: int
    n_targets_written: int
    n_raw_scanned: int
    n_raw_matched: int
    n_skipped_no_text: int
    n_skipped_duplicate: int


def _row_in_scope(row: dict, scope: TrainAbsaScope) -> bool:
    uid = str(row.get("user_id") or "")
    item = str(row.get("parent_asin") or row.get("asin") or "")
    ts = int(row.get("timestamp") or 0)
    if not uid or not item:
        return False
    return (
        uid in scope.train_users
        and item in scope.train_items
        and ts <= scope.cutoff_ts
    )


def export_absa_targets(
    *,
    train_path: str | Path,
    raw_review_path: str | Path,
    out_path: str | Path,
) -> AbsaTargetExportStats:
    """Write review_id+text JSONL for reviews in the train KG scope.

    Scans the raw category file once and keeps only reviews whose
    (user_id, item, timestamp) fall within the processed train split —
    the same filter used when loading ABSA into the in-memory KG.
    """
    train = load_split_jsonl(train_path)
    scope = build_train_scope(train)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    n_raw = n_matched = n_no_text = n_dup = n_written = 0

    with Path(raw_review_path).open(encoding="utf-8") as src, out_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            n_raw += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not _row_in_scope(row, scope):
                continue
            n_matched += 1
            text = (row.get("text") or "").strip()
            if not text:
                n_no_text += 1
                continue
            rid = review_id_from_row(row)
            if not rid or rid in seen:
                if rid:
                    n_dup += 1
                continue
            seen.add(rid)
            dst.write(
                json.dumps({"review_id": rid, "text": text}, ensure_ascii=False) + "\n"
            )
            n_written += 1

    return AbsaTargetExportStats(
        n_train_interactions=len(train),
        n_targets_written=n_written,
        n_raw_scanned=n_raw,
        n_raw_matched=n_matched,
        n_skipped_no_text=n_no_text,
        n_skipped_duplicate=n_dup,
    )
