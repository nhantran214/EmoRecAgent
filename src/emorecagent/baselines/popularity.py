"""Global popularity baseline (U11).

User-independent: every user is scored by item interaction frequency in train.
The weakest sanity baseline; a personalized method that cannot beat it is broken.
"""

from __future__ import annotations

from collections import Counter

from ..data.types import Interaction
from .base import Recommender


class PopularityRecommender(Recommender):
    name = "popularity"

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()

    def fit(self, interactions: list[Interaction]) -> "PopularityRecommender":
        self._counts = Counter(it.item for it in interactions)
        return self

    def score(self, user_id: str, candidates: list[str]) -> dict[str, float]:
        return {i: float(self._counts.get(i, 0)) for i in candidates}
