"""Full-system recommender for the eval harness."""

from __future__ import annotations

from ..agents.profiling_agent import DynamicUserProfilingAgent
from ..baselines.aspect_aware import static_weights
from ..baselines.base import Recommender
from ..data.types import Interaction
from ..scoring.dynamic_weights import AspectSignal
from ..scoring.score import rank_items
from .context import RecommendContext


class _KGAspectSource:
    def __init__(self, ctx: RecommendContext) -> None:
        self._ctx = ctx

    def get_item_aspects(self, item_id: str) -> dict[str, float]:
        return self._ctx.kg.get_item_aspect_sentiment(
            item_id, rescaled=self._ctx.affective_rescaled
        )


class _KGSignalSource:
    def __init__(self, ctx: RecommendContext) -> None:
        self._ctx = ctx

    def get_user_aspect_signals(self, user_id: str) -> list[AspectSignal]:
        out: list[AspectSignal] = []
        for uid, _, aspect, pol, ts in self._ctx.kg.signals:
            if uid == user_id:
                out.append(
                    AspectSignal(aspect=aspect, polarity=pol, timestamp_ms=ts)
                )
        return sorted(out, key=lambda s: s.timestamp_ms)

    def upsert_user_preferences(self, user_id, weights, updated_ts):  # noqa: ANN001
        for aspect, weight in weights.items():
            self._ctx.kg.upsert_user_preference(user_id, aspect, weight, updated_ts)


class EmoRecRecommender(Recommender):
    """Scores candidates with dynamic (or static) aspect weights and S(u,i)."""

    name = "emorecagent_fast"

    def __init__(self, context: RecommendContext) -> None:
        self._ctx = context
        self._query_ts: dict[str, int] = {}
        self._default_query_ts: dict[str, int] = {}
        self._profiling = DynamicUserProfilingAgent(
            _KGSignalSource(context), lambda_per_day=context.lambda_decay
        )

    def fit(self, interactions: list[Interaction]) -> "EmoRecRecommender":
        per_user: dict[str, int] = {}
        for it in interactions:
            per_user[it.user_id] = max(per_user.get(it.user_id, 0), it.timestamp)
        self._default_query_ts = {u: ts + 1 for u, ts in per_user.items()}
        return self

    def prepare_user_query(self, user_id: str, timestamp_ms: int) -> None:
        """Set the evaluation query time (test interaction timestamp)."""
        self._query_ts[user_id] = timestamp_ms

    def _query_time(self, user_id: str) -> int:
        return self._query_ts.get(
            user_id, self._default_query_ts.get(user_id, 0)
        )

    def _weights(self, user_id: str, t_query: int) -> dict[str, float]:
        if self._ctx.use_dynamic_weights:
            return self._profiling.profile(
                user_id, t_query, self._ctx.top_k_aspects, persist=False
            )
        signals = self._ctx.kg.get_user_signals(user_id, before_ts=t_query)
        return static_weights(signals)

    def score(self, user_id: str, candidates: list[str]) -> dict[str, float]:
        if not candidates:
            return {}
        t_query = self._query_time(user_id)
        weights = self._weights(user_id, t_query)

        # Score the full candidate set (eval protocol), not only the CF/aspect pool.
        aspect_src = _KGAspectSource(self._ctx)
        item_aspect_maps = {
            item: aspect_src.get_item_aspects(item) for item in candidates
        }
        s_base = self._ctx.cf.score(user_id, candidates)
        ranked = rank_items(self._ctx.alpha, s_base, weights, item_aspect_maps)
        return {item: bd.total for item, bd in ranked}
