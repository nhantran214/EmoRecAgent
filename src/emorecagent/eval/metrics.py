"""Top-K ranking metrics (U12 / R11).

All metrics take a best-first `ranked` list of item ids and the `relevant` set of
ground-truth items (a single held-out item under leave-last-out, but the
functions support multiple). Rank positions are 1-indexed. Each metric is
validated against hand-computed cases in tests.
"""

from __future__ import annotations

import math

METRIC_NAMES = ("recall", "ndcg", "hr", "mrr")


def _first_hit_rank(ranked: list[str], relevant: set[str], k: int) -> int | None:
    """1-indexed rank of the first relevant item within the top-k, else None."""
    for pos, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            return pos
    return None


def hr_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Hit rate: 1.0 if any relevant item is in the top-k, else 0.0."""
    return 1.0 if _first_hit_rank(ranked, relevant, k) is not None else 0.0


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items retrieved in the top-k."""
    if not relevant:
        return 0.0
    hits = sum(1 for item in ranked[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Normalized DCG@k with binary relevance."""
    if not relevant:
        return 0.0
    dcg = 0.0
    for pos, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(pos + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(pos + 1) for pos in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Reciprocal rank of the first relevant item within the top-k (else 0)."""
    rank = _first_hit_rank(ranked, relevant, k)
    return 1.0 / rank if rank is not None else 0.0


def evaluate_ranking(
    ranked: list[str], relevant: set[str], k: int
) -> dict[str, float]:
    """All four metrics at a single k."""
    return {
        "recall": recall_at_k(ranked, relevant, k),
        "ndcg": ndcg_at_k(ranked, relevant, k),
        "hr": hr_at_k(ranked, relevant, k),
        "mrr": mrr_at_k(ranked, relevant, k),
    }
