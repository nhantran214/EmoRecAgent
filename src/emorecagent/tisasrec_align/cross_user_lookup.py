"""Cross-user co-purchase lookup from train review/purchase patterns."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ..data.types import Interaction
from .review_context import load_review_text_index


CrossUserLookup = dict[str, dict[str, int]]


def build_lookup_from_train(
    train: list[Interaction],
    review_index: dict[tuple[str, str, int], str],
) -> CrossUserLookup:
    """Map reviewed anchor item -> co-purchased item counts (same user)."""
    per_user: dict[str, list[Interaction]] = defaultdict(list)
    for it in train:
        per_user[it.user_id].append(it)
    lookup: CrossUserLookup = defaultdict(lambda: defaultdict(int))
    for uid, events in per_user.items():
        purchased = {it.item for it in events}
        reviewed = {
            it.item
            for it in events
            if review_index.get((uid, it.item, it.timestamp), "").strip()
        }
        for anchor in reviewed:
            for co in purchased:
                if co != anchor:
                    lookup[anchor][co] += 1
    return {a: dict(co_map) for a, co_map in lookup.items()}


def build_lookup_id_only(train: list[Interaction]) -> CrossUserLookup:
    """Map visited anchor item -> co-visited item counts (no review text gate)."""
    per_user: dict[str, list[Interaction]] = defaultdict(list)
    for it in train:
        per_user[it.user_id].append(it)
    lookup: CrossUserLookup = defaultdict(lambda: defaultdict(int))
    for events in per_user.values():
        visited = {it.item for it in events}
        for anchor in visited:
            for co in visited:
                if co != anchor:
                    lookup[anchor][co] += 1
    return {a: dict(co_map) for a, co_map in lookup.items()}


def save_lookup(path: str | Path, lookup: CrossUserLookup) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(lookup, indent=2), encoding="utf-8")


def load_lookup(path: str | Path) -> CrossUserLookup:
    p = Path(path)
    if not p.is_file():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    out: CrossUserLookup = {}
    for anchor, co_map in raw.items():
        if isinstance(co_map, dict):
            out[str(anchor)] = {str(k): int(v) for k, v in co_map.items()}
    return out


def lookup_co_items(
    anchor_items: list[str],
    pool: set[str],
    lookup: CrossUserLookup,
) -> dict[str, float]:
    """Aggregate co-purchase scores for pool items; empty when no evidence."""
    scores: dict[str, float] = defaultdict(float)
    for anchor in anchor_items:
        co_map = lookup.get(anchor)
        if not co_map:
            continue
        for co_item, count in co_map.items():
            if co_item in pool:
                scores[co_item] += float(count)
    if not scores:
        return {}
    max_score = max(scores.values())
    if max_score <= 0:
        return {}
    return {item: score / max_score for item, score in scores.items()}


def build_lookup_from_config(
    train: list[Interaction],
    review_path: str | Path,
    *,
    mode: str = "review_text",
) -> CrossUserLookup:
    if mode == "id_only":
        return build_lookup_id_only(train)
    review_index = load_review_text_index(review_path)
    return build_lookup_from_train(train, review_index)
