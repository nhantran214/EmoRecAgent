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
        allowed_reviews: (
            set[tuple[str, str, int]] | frozenset[tuple[str, str, int]] | None
        ) = None,
    ) -> None:
        """
        Parameters
        ----------
        train_users:
            If set, only load reviews for these user ids.
        allowed_reviews:
            Optional ``(user_id, item_id, timestamp_ms)`` whitelist. Use the
            Protocol-B history set (train or train+valid) so test-split reviews
            cannot leak into $T_u$ profiling.
        """
        self._cache = AbsaCache(cache_path, readonly=True)
        self._signals: dict[str, list[AspectSignal]] = {}
        self._load(
            review_path,
            train_users=train_users,
            allowed_reviews=allowed_reviews,
        )
        self._cache.close()

    def _load(
        self,
        review_path: str | Path,
        *,
        train_users: set[str] | frozenset[str] | None,
        allowed_reviews: (
            set[tuple[str, str, int]] | frozenset[tuple[str, str, int]] | None
        ),
    ) -> None:
        polarity_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
        for row in stream_reviews(review_path):
            uid = str(row.get("user_id") or "")
            if train_users is not None and uid not in train_users:
                continue
            item = str(row.get("parent_asin") or row.get("asin") or "")
            ts = int(row.get("timestamp") or 0)
            if allowed_reviews is not None and (uid, item, ts) not in allowed_reviews:
                continue
            rid = review_id_from_row(row)
            if not rid:
                continue
            cached = self._cache.get(rid)
            if cached is None or not cached.triples:
                continue
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
