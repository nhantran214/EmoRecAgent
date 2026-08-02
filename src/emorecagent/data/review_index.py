"""Map review rows to stable cache keys scoped to the train split."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .types import Interaction
from ..eval.runner import load_split_jsonl


@dataclass(frozen=True, slots=True)
class TrainAbsaScope:
    """Reviews eligible for KG hydration (matches build_recommend_context)."""

    train_users: frozenset[str]
    train_items: frozenset[str]
    cutoff_ts: int


def build_train_scope(train: list[Interaction]) -> TrainAbsaScope:
    if not train:
        return TrainAbsaScope(frozenset(), frozenset(), 0)
    return TrainAbsaScope(
        train_users=frozenset(it.user_id for it in train),
        train_items=frozenset(it.item for it in train),
        cutoff_ts=max(it.timestamp for it in train),
    )


def review_id_from_row(row: dict) -> str:
    """Stable review id: explicit id, else user+item+timestamp composite."""
    explicit = str(row.get("review_id") or "").strip()
    if explicit:
        return explicit
    uid = str(row.get("user_id") or "")
    item = str(row.get("parent_asin") or row.get("asin") or "")
    ts = int(row.get("timestamp") or 0)
    if uid and item and ts:
        return f"{uid}|{item}|{ts}"
    return ""


def build_review_index_from_scope(
    train: list[Interaction],
    raw_review_path: str | Path,
) -> dict[str, tuple[str, str, int]]:
    """Map review_id -> (user_id, item_id, timestamp_ms) for train-scoped rows."""
    from .loader import stream_reviews

    scope = build_train_scope(train)
    out: dict[str, tuple[str, str, int]] = {}
    for row in stream_reviews(raw_review_path):
        uid = str(row.get("user_id") or "")
        item = str(row.get("parent_asin") or row.get("asin") or "")
        ts = int(row.get("timestamp") or 0)
        if not uid or not item:
            continue
        if (
            uid not in scope.train_users
            or item not in scope.train_items
            or ts > scope.cutoff_ts
        ):
            continue
        rid = review_id_from_row(row)
        if rid:
            out[rid] = (uid, item, ts)
    return out


def build_review_index_from_train_path(
    train_path: str | Path,
    raw_review_path: str | Path,
) -> dict[str, tuple[str, str, int]]:
    train = load_split_jsonl(train_path)
    return build_review_index_from_scope(train, raw_review_path)
