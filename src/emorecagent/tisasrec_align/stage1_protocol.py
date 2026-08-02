"""Shared Stage-1 scoring surface used by Stage-2 rerank / eval."""

from __future__ import annotations

from typing import Protocol

from ..data.types import Interaction


class Stage1Scorer(Protocol):
    """Minimal API AlignFullRank / RecBole Stage-1 backends must provide."""

    name: str

    def fit(self, interactions: list[Interaction]) -> Stage1Scorer: ...

    def prepare_user_query(self, user_id: str, timestamp_ms: int) -> None: ...

    def catalog_items(self) -> list[str]: ...

    def score(
        self,
        user_id: str,
        candidates: list[str],
        *,
        query_ts_ms: int | None = None,
    ) -> dict[str, float]: ...

    def rank(
        self,
        user_id: str,
        candidates: list[str],
        *,
        query_ts_ms: int | None = None,
    ) -> list[str]: ...
