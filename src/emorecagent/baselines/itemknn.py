"""ItemKNN baseline.

Thin wrapper over the shared CFBase cosine ItemKNN backend.
"""

from __future__ import annotations

from ..data.types import Interaction
from ..scoring.cf_base import CFBase
from .base import Recommender


class ItemKNNRecommender(Recommender):
    name = "itemknn"

    def __init__(self, seed: int = 42) -> None:
        self._cf = CFBase(backend="itemknn", seed=seed)

    def fit(self, interactions: list[Interaction]) -> "ItemKNNRecommender":
        self._cf.fit(interactions)
        return self

    def score(self, user_id: str, candidates: list[str]) -> dict[str, float]:
        return self._cf.score(user_id, candidates)
