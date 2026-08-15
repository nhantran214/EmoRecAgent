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


def item_review_snippets_from_index(
    review_index: dict[tuple[str, str, int], str],
    *,
    keep_ids: set[str] | None = None,
    allowed_reviews: set[tuple[str, str, int]] | None = None,
    max_chars: int = 100,
    max_per_item: int = 1,
) -> dict[str, list[str]]:
    """Collapse ``(user,item,ts)→text`` into per-item review candidates for cards.

    ``allowed_reviews`` restricts to Protocol-B history keys (train or
    train+valid). Without it, test-split reviews of the gold item can land on
    that item's card (label leak). Keeps up to ``max_per_item`` non-empty
    reviews per item (stable by iteration order of ``review_index``).
    """
    out: dict[str, list[str]] = {}
    cap = max(1, int(max_per_item))
    # Keep more text when storing multiple candidates for T_u matching.
    width = max(16, int(max_chars) * (2 if cap > 1 else 1))
    width = min(width, 280)
    for key, text in review_index.items():
        uid, item, ts = key
        if keep_ids is not None and item not in keep_ids:
            continue
        if allowed_reviews is not None and (uid, item, int(ts)) not in allowed_reviews:
            continue
        raw = str(text or "").strip()
        if not raw:
            continue
        bucket = out.setdefault(item, [])
        if len(bucket) >= cap:
            continue
        snippet = raw if len(raw) <= width else raw[: width - 1] + "…"
        bucket.append(snippet)
    return out
