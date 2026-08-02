"""Review text lookup for prefix interactions before a query timestamp."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..data.loader import stream_reviews
from ..data.types import Interaction


@dataclass(frozen=True, slots=True)
class PrefixReview:
    item_id: str
    timestamp_ms: int
    review_text: str


def load_review_text_index(
    review_path: str | Path,
) -> dict[tuple[str, str, int], str]:
    """Map (user_id, item_id, timestamp_ms) -> review text."""
    out: dict[tuple[str, str, int], str] = {}
    for row in stream_reviews(review_path):
        uid = str(row.get("user_id") or "")
        item = str(row.get("parent_asin") or row.get("asin") or "")
        ts = int(row.get("timestamp") or 0)
        text = str(
            row.get("text") or row.get("reviewText") or row.get("review_text") or ""
        ).strip()
        if uid and item and ts and text:
            out[(uid, item, ts)] = text
    return out


def prefix_reviews_for_user(
    user_id: str,
    interactions: list[Interaction],
    query_ts_ms: int,
    review_index: dict[tuple[str, str, int], str],
) -> list[PrefixReview]:
    """Chronological reviews strictly before query_ts for user's past items."""
    rows: list[PrefixReview] = []
    for it in sorted(interactions, key=lambda x: (x.timestamp, x.item)):
        if it.user_id != user_id or it.timestamp >= query_ts_ms:
            continue
        text = review_index.get((it.user_id, it.item, it.timestamp), "")
        if text:
            rows.append(
                PrefixReview(
                    item_id=it.item, timestamp_ms=it.timestamp, review_text=text
                )
            )
    return rows
