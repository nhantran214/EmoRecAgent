"""Aspect-aware baseline wired for temporal eval queries."""

from __future__ import annotations

from ..baselines.aspect_aware import AspectAwareRecommender, static_weights
from ..data.types import Interaction
from .context import RecommendContext


class AspectAwareEvalRecommender(AspectAwareRecommender):
    """Static aspect weights recomputed at each user's query timestamp."""

    name = "aspect_aware"

    def __init__(self, context: RecommendContext) -> None:
        item_aspects = {
            item: context.kg.get_item_aspect_sentiment(item, rescaled=True)
            for item in {it.item for it in context.kg.interactions}
        }
        super().__init__(user_signals={}, item_aspect_sentiment=item_aspects)
        self._ctx = context
        self._query_ts: dict[str, int] = {}
        self._default_query_ts: dict[str, int] = {}

    def fit(self, interactions: list[Interaction]) -> "AspectAwareEvalRecommender":
        per_user: dict[str, int] = {}
        for it in interactions:
            per_user[it.user_id] = max(per_user.get(it.user_id, 0), it.timestamp)
        self._default_query_ts = {u: ts + 1 for u, ts in per_user.items()}
        return self

    def prepare_user_query(self, user_id: str, timestamp_ms: int) -> None:
        self._query_ts[user_id] = timestamp_ms

    def score(self, user_id: str, candidates: list[str]) -> dict[str, float]:
        t_query = self._query_ts.get(
            user_id, self._default_query_ts.get(user_id, 0)
        )
        signals = self._ctx.kg.get_user_signals(user_id, before_ts=t_query)
        self._weights[user_id] = static_weights(signals)
        return super().score(user_id, candidates)
