"""Aspect-aware static baseline, EFM / TriRank-style (U11).

This is the most important baseline for our novelty claim. It is fed the *same*
ABSA item-aspect sentiment Ê_i(a) and the *same* user aspect signals as the full
system, but its user aspect weights are **static** — the salience accumulation
with no time decay (the λ → 0 limit of `dynamic_weights`). Comparing the full
system against this isolates the contribution of the *dynamic / temporal*
mechanism from the contribution of "using aspects at all".

Score: S_aspect(u, i) = sum_a w_static_u(a) * Ê_i(a).
"""

from __future__ import annotations

from ..data.types import Interaction
from ..scoring.dynamic_weights import AspectSignal, intensity
from .base import Recommender


def static_weights(signals: list[AspectSignal]) -> dict[str, float]:
    """Time-agnostic normalized aspect salience (no decay)."""
    interest: dict[str, float] = {}
    for s in signals:
        interest[s.aspect] = interest.get(s.aspect, 0.0) + intensity(s.polarity)
    total = sum(interest.values())
    if total <= 0.0:
        return {}
    return {a: v / total for a, v in interest.items()}


class AspectAwareRecommender(Recommender):
    name = "aspect_aware"

    def __init__(
        self,
        user_signals: dict[str, list[AspectSignal]],
        item_aspect_sentiment: dict[str, dict[str, float]],
    ) -> None:
        self._user_signals = user_signals
        self._item_aspects = item_aspect_sentiment
        self._weights: dict[str, dict[str, float]] = {}

    def fit(self, interactions: list[Interaction]) -> "AspectAwareRecommender":
        self._weights = {
            u: static_weights(sigs) for u, sigs in self._user_signals.items()
        }
        return self

    def score(self, user_id: str, candidates: list[str]) -> dict[str, float]:
        weights = self._weights.get(user_id, {})
        out: dict[str, float] = {}
        for item in candidates:
            item_aspects = self._item_aspects.get(item, {})
            out[item] = sum(
                w * item_aspects.get(a, 0.0) for a, w in weights.items()
            )
        return out
