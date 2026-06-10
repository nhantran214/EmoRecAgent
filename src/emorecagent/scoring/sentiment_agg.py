"""Item-aspect sentiment aggregation: E_i(a) and its [0, 1] rescale Ê_i(a) (U7).

E_raw is a helpfulness-capped mean polarity in [-1, 1]; rescaling to [0, 1]
(via (x+1)/2) keeps the affective term on the same scale as the CF base so the
alpha blend is a clean convex interpolation. Helpfulness weighting is *capped*
to avoid importing item-popularity/age bias into the affective term.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ItemAspectTriple:
    """One ABSA triple's contribution to an item's aspect sentiment."""

    aspect: str
    polarity: float  # in [-1, 1]
    helpful_vote: int = 0


def rescale(e_raw: float) -> float:
    """Map E_raw in [-1, 1] to Ê in [0, 1], clamped."""
    return max(0.0, min(1.0, (e_raw + 1.0) / 2.0))


def aggregate_raw(
    triples: list[ItemAspectTriple], helpful_cap: int
) -> dict[str, float]:
    """Helpfulness-capped weighted mean polarity per aspect, in [-1, 1]."""
    num: dict[str, float] = {}
    den: dict[str, float] = {}
    for t in triples:
        w = 1.0 + min(max(t.helpful_vote, 0), helpful_cap)
        num[t.aspect] = num.get(t.aspect, 0.0) + w * t.polarity
        den[t.aspect] = den.get(t.aspect, 0.0) + w
    return {a: num[a] / den[a] for a in num if den[a] > 0}


def aggregate_rescaled(
    triples: list[ItemAspectTriple], helpful_cap: int
) -> dict[str, float]:
    """Per-aspect Ê_i(a) in [0, 1]."""
    return {a: rescale(v) for a, v in aggregate_raw(triples, helpful_cap).items()}


class ItemAspectSentimentSource(Protocol):
    """Supplies an item's rescaled aspect sentiment (the Neo4j repo in prod)."""

    def get_item_aspect_sentiment(self, item_id: str) -> dict[str, float]: ...
