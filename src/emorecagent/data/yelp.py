"""Convert Yelp Open Dataset JSON into Amazon-Reviews-2023-shaped JSONL.

The EmoRecAgent method is unchanged: ``build_dataset`` / loaders still expect
``user_id``, ``parent_asin``, ``rating``, ``timestamp`` (ms), ``text``.
Native Yelp rows are also accepted at read time via ``loader.stream_reviews``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path

from .yelp_paths import (
    BUSINESS_FILENAMES,
    DEFAULT_YELP_DATASET_DIR,
    REVIEW_FILENAMES,
    find_yelp_json,
    is_yelp_review_row,
    resolve_meta_source,
    resolve_review_source,
)

__all__ = [
    "BUSINESS_FILENAMES",
    "DEFAULT_YELP_DATASET_DIR",
    "REVIEW_FILENAMES",
    "convert_businesses",
    "convert_reviews",
    "find_yelp_json",
    "is_yelp_review_row",
    "resolve_meta_source",
    "resolve_review_source",
    "yelp_business_to_meta",
    "yelp_date_to_timestamp_ms",
    "yelp_review_to_amazon",
]


def yelp_date_to_timestamp_ms(raw: object) -> int | None:
    """Parse Yelp ``date`` (``YYYY-MM-DD`` or ``YYYY-MM-DD HH:MM:SS``) → unix ms."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            continue
    return None


def yelp_review_to_amazon(obj: dict) -> dict | None:
    """Map one Yelp review object to the Amazon-like schema used by loaders."""
    user = obj.get("user_id")
    item = obj.get("business_id")
    rating = obj.get("stars", obj.get("rating"))
    ts = yelp_date_to_timestamp_ms(obj.get("date"))
    text = (obj.get("text") or "").strip()
    if not user or not item or rating is None or ts is None:
        return None
    review_id = str(obj.get("review_id") or f"{user}|{item}|{ts}")
    return {
        "user_id": str(user),
        "parent_asin": str(item),
        "asin": str(item),
        "rating": float(rating),
        "timestamp": int(ts),
        "helpful_vote": int(obj.get("useful", 0) or 0),
        "verified_purchase": False,
        "text": text,
        "review_id": review_id,
        "title": "",
    }


def yelp_business_to_meta(obj: dict) -> dict | None:
    """Map one Yelp business object to a lightweight Amazon-like meta row."""
    bid = obj.get("business_id")
    if not bid:
        return None
    cats = obj.get("categories")
    if isinstance(cats, str):
        cat_list = [c.strip() for c in cats.split(",") if c.strip()]
    elif isinstance(cats, list):
        cat_list = [str(c) for c in cats]
    else:
        cat_list = []
    return {
        "parent_asin": str(bid),
        "asin": str(bid),
        "title": str(obj.get("name") or ""),
        "main_category": cat_list[0] if cat_list else "Yelp",
        "categories": cat_list,
        "average_rating": float(obj.get("stars") or 0.0),
        "rating_number": int(obj.get("review_count") or 0),
        "features": [],
        "description": [],
        "store": str(obj.get("city") or ""),
    }


def iter_json_lines(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def convert_reviews(
    review_path: Path,
    out_path: Path,
    *,
    keep_business_ids: set[str] | None = None,
    max_reviews: int | None = None,
) -> int:
    """Write Amazon-like review JSONL. Returns number of rows written."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as out:
        for obj in iter_json_lines(review_path):
            if keep_business_ids is not None:
                bid = obj.get("business_id")
                if bid is None or str(bid) not in keep_business_ids:
                    continue
            row = yelp_review_to_amazon(obj)
            if row is None:
                continue
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if max_reviews is not None and n >= max_reviews:
                break
    return n


def convert_businesses(
    business_path: Path,
    out_path: Path,
    *,
    cities: Iterable[str] | None = None,
    categories_substr: Iterable[str] | None = None,
) -> tuple[int, set[str]]:
    """Write Amazon-like meta JSONL. Returns (rows, kept business_ids)."""
    city_set = {c.strip().lower() for c in (cities or []) if c.strip()}
    cat_needles = [c.strip().lower() for c in (categories_substr or []) if c.strip()]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept: set[str] = set()
    n = 0
    with out_path.open("w", encoding="utf-8") as out:
        for obj in iter_json_lines(business_path):
            if city_set:
                city = str(obj.get("city") or "").strip().lower()
                if city not in city_set:
                    continue
            if cat_needles:
                cats = str(obj.get("categories") or "").lower()
                if not any(needle in cats for needle in cat_needles):
                    continue
            row = yelp_business_to_meta(obj)
            if row is None:
                continue
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept.add(str(row["parent_asin"]))
            n += 1
    return n, kept
