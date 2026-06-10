"""Batch loaders from split files and ABSA cache into Neo4j (U5)."""

from __future__ import annotations

import json
from pathlib import Path

from ..absa.cache import AbsaCache
from ..data.types import Interaction
from ..eval.runner import load_split_jsonl
from ..scoring.sentiment_agg import ItemAspectTriple, aggregate_raw
from .repository import KGRepository, KGStore


def load_interactions(repo: KGStore, interactions: list[Interaction]) -> int:
    for it in interactions:
        repo.upsert_interaction(it)
    return len(interactions)


def load_absa_cache(
    repo: KGStore,
    cache: AbsaCache,
    review_index: dict[str, tuple[str, str, int]],
) -> int:
    """Load cached triples.

    `review_index` maps review_id -> (user_id, item_id, timestamp_ms).
    """
    n = 0
    for review_id, (user_id, item_id, ts) in review_index.items():
        triples = cache.get(review_id)
        if triples is None or not triples.triples:
            continue
        repo.upsert_triples(user_id, item_id, triples.triples, ts)
        n += 1
    return n


def aggregate_item_sentiment_from_cache(
    repo: KGStore,
    cache: AbsaCache,
    item_reviews: dict[str, list[tuple[str, int]]],
    *,
    helpful_cap: int,
    cutoff_ts: int | None = None,
) -> int:
    """Aggregate ABSA polarities per (item, aspect) and write HAS_SENTIMENT edges.

    `item_reviews` maps item_id -> list of (review_id, helpful_vote).
    """
    n_edges = 0
    for item_id, refs in item_reviews.items():
        triples: list[ItemAspectTriple] = []
        for review_id, helpful in refs:
            cached = cache.get(review_id)
            if cached is None:
                continue
            for t in cached.triples:
                polarity = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}[
                    t.sentiment
                ]
                triples.append(
                    ItemAspectTriple(
                        aspect=t.aspect, polarity=polarity, helpful_vote=helpful
                    )
                )
        if not triples:
            continue
        raw = aggregate_raw(triples, helpful_cap)
        for aspect, score in raw.items():
            support = sum(1 for t in triples if t.aspect == aspect)
            ts = cutoff_ts if cutoff_ts is not None else 0
            repo.upsert_item_sentiment(item_id, aspect, score, support, ts)
            n_edges += 1
    return n_edges


def load_split_dir(repo: KGRepository, split_dir: str | Path) -> dict[str, int]:
    """Load train+valid+test interactions from a processed split directory."""
    split_dir = Path(split_dir)
    counts = {}
    for name in ("train", "valid", "test"):
        path = split_dir / f"{name}.jsonl"
        if path.exists():
            counts[name] = load_interactions(repo, load_split_jsonl(path))
    return counts


def build_review_index_from_jsonl(
    path: str | Path, max_rows: int | None = None
) -> dict[str, tuple[str, str, int]]:
    """Map review_id -> (user_id, parent_asin, timestamp) from raw reviews."""
    out: dict[str, tuple[str, str, int]] = {}
    n = 0
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if max_rows is not None and n >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = str(row.get("review_id") or "")
            uid = str(row.get("user_id") or "")
            item = str(row.get("parent_asin") or row.get("asin") or "")
            ts = int(row.get("timestamp", 0))
            if rid and uid and item:
                out[rid] = (uid, item, ts)
                n += 1
    return out
