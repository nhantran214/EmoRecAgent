"""SVD / matrix-factorization baseline.

Thin wrapper over the shared CFBase truncated-SVD backend so the standard
CF baseline and the full system's S_base share one implementation.
"""

from __future__ import annotations

from ..data.types import Interaction
from ..scoring.cf_base import CFBase
from .base import Recommender


class SVDRecommender(Recommender):
    name = "svd"

    def __init__(self, factors: int = 64, seed: int = 42) -> None:
        self._cf = CFBase(backend="svd", factors=factors, seed=seed)

    def fit(self, interactions: list[Interaction]) -> "SVDRecommender":
        self._cf.fit(interactions)
        return self

    def score(self, user_id: str, candidates: list[str]) -> dict[str, float]:
        return self._cf.score(user_id, candidates)
