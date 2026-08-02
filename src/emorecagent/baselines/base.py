"""Shared recommender interface.

Every baseline and the full EmoRecAgent system expose the same two operations:
`fit` on train interactions, and `score` a candidate set for a user. Ranking and
top-K selection are provided once here so the eval harness treats all methods
identically. Scores need not be normalized; only their order within a candidate
set matters for ranking metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..data.types import Interaction


class Recommender(ABC):
    """Abstract recommender over a fixed item catalog."""

    name: str = "recommender"

    @abstractmethod
    def fit(self, interactions: list[Interaction]) -> "Recommender":
        """Train on the train-split interactions and return self."""

    @abstractmethod
    def score(self, user_id: str, candidates: list[str]) -> dict[str, float]:
        """Score each candidate item for the user (higher = better)."""

    def rank(self, user_id: str, candidates: list[str]) -> list[str]:
        """Candidates ranked best-first; ties broken by item id for determinism."""
        scores = self.score(user_id, candidates)
        return sorted(candidates, key=lambda i: (-scores.get(i, 0.0), i))

    def recommend(
        self, user_id: str, candidates: list[str], k: int
    ) -> list[str]:
        """Top-k candidate items for the user."""
        return self.rank(user_id, candidates)[:k]
