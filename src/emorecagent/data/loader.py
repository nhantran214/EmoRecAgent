"""Streaming JSONL ingestion for Amazon Reviews 2023 (and native Yelp).

Never loads the full file into memory at once: reviews are streamed line by
line. De-duplication keeps the earliest interaction per (user, item) pair.

Native Yelp review dumps (``business_id`` / ``stars`` / ``date``) are normalized
to the Amazon-shaped fields expected by the rest of the pipeline.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from .types import Interaction
from .yelp import is_yelp_review_row, resolve_review_source, yelp_review_to_amazon


def normalize_review_row(obj: dict) -> dict | None:
    """Return an Amazon-shaped review dict, or None if the row is unusable."""
    if is_yelp_review_row(obj):
        return yelp_review_to_amazon(obj)
    return obj


def stream_reviews(path: str | Path) -> Iterator[dict]:
    """Yield Amazon-shaped review dicts from a JSONL file or Yelp dataset dir.

    Malformed lines and unusable Yelp rows are skipped.
    """
    src = resolve_review_source(path)
    with src.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(raw, dict):
                continue
            row = normalize_review_row(raw)
            if row is not None:
                yield row


def _to_interaction(obj: dict) -> Interaction | None:
    user = obj.get("user_id")
    item = obj.get("parent_asin") or obj.get("asin")
    ts = obj.get("timestamp")
    rating = obj.get("rating")
    if not user or not item or ts is None or rating is None:
        return None
    return Interaction(
        user_id=str(user),
        item=str(item),
        rating=float(rating),
        timestamp=int(ts),
        helpful_vote=int(obj.get("helpful_vote", 0) or 0),
        verified_purchase=bool(obj.get("verified_purchase", False)),
    )


def load_interactions(
    path: str | Path, max_scan: int | None = None
) -> list[Interaction]:
    """Stream reviews into Interaction records.

    `max_scan` caps the number of raw lines read (useful for smoke tests on the
    11 GB file); None scans the whole file.
    """
    out: list[Interaction] = []
    for i, obj in enumerate(stream_reviews(path)):
        if max_scan is not None and i >= max_scan:
            break
        inter = _to_interaction(obj)
        if inter is not None:
            out.append(inter)
    return out


def dedup_earliest(interactions: list[Interaction]) -> list[Interaction]:
    """Collapse repeated (user, item) pairs, keeping the earliest timestamp."""
    best: dict[tuple[str, str], Interaction] = {}
    for it in interactions:
        key = (it.user_id, it.item)
        cur = best.get(key)
        if cur is None or it.timestamp < cur.timestamp:
            best[key] = it
    return list(best.values())


def load_split_jsonl(path: str | Path) -> list[Interaction]:
    """Read interactions written by data.split.write_split."""
    out: list[Interaction] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(
                Interaction(
                    user_id=d["user_id"],
                    item=d["item"],
                    rating=float(d.get("rating", 0.0)),
                    timestamp=int(d.get("timestamp", 0)),
                    helpful_vote=int(d.get("helpful_vote", 0)),
                    verified_purchase=bool(d.get("verified_purchase", False)),
                )
            )
    return out
