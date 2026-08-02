"""ABSA-cache-backed aspect signals for profiling (no Neo4j)."""

from __future__ import annotations

from pathlib import Path

from ..absa.cache import AbsaCache
from ..data.loader import stream_reviews
from ..data.review_index import review_id_from_row
from ..scoring.dynamic_weights import AspectSignal


class AbsaCacheSignalSource:
    """Build per-user AspectSignal lists from ABSA cache + raw reviews."""

    def __init__(
        self,
        cache_path: str | Path,
        review_path: str | Path,
        *,
        train_users: set[str] | frozenset[str] | None = None,
    ) -> None:
        self._cache = AbsaCache(cache_path, readonly=True)
        self._signals: dict[str, list[AspectSignal]] = {}
        self._load(review_path, train_users=train_users)
        self._cache.close()

    def _load(
        self,
        review_path: str | Path,
        *,
        train_users: set[str] | frozenset[str] | None,
    ) -> None:
        polarity_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
        for row in stream_reviews(review_path):
            uid = str(row.get("user_id") or "")
            if train_users is not None and uid not in train_users:
                continue
            rid = review_id_from_row(row)
            if not rid:
                continue
            cached = self._cache.get(rid)
            if cached is None or not cached.triples:
                continue
            ts = int(row.get("timestamp") or 0)
            for t in cached.triples:
                aspect = str(t.aspect or "other").strip().lower() or "other"
                pol = polarity_map.get(str(t.sentiment), 0.0)
                self._signals.setdefault(uid, []).append(
                    AspectSignal(aspect=aspect, polarity=pol, timestamp_ms=ts)
                )

    def get_user_aspect_signals(self, user_id: str) -> list[AspectSignal]:
        return list(self._signals.get(user_id, []))

    def upsert_user_preferences(
        self, user_id: str, weights: dict[str, float], updated_ts: int
    ) -> None:
        del user_id, weights, updated_ts
