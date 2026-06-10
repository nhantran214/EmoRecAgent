"""Streaming JSONL ingestion for Amazon Reviews 2023.

Never loads the full file into memory at once: reviews are streamed line by
line. De-duplication keeps the earliest interaction per (user, item) pair.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from .types import Interaction


def stream_reviews(path: str | Path) -> Iterator[dict]:
    """Yield parsed JSON objects from a JSONL file, one per line.

    Malformed lines are skipped (the raw dumps occasionally contain them).
    """
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


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
