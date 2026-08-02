"""Dynamic User Profiling Agent.

Reads the user's affective signals from a signal source (the Neo4j repository in
production, any object implementing the Protocol in tests), computes the
time-decayed weight vector at the query time, optionally persists the resulting
PREFERS weights, and returns the top-k aspects.

The numeric core lives in `scoring.dynamic_weights`; this agent is the thin I/O
wrapper so it is testable without a running graph database.
"""

from __future__ import annotations

from typing import Protocol

from ..scoring.dynamic_weights import AspectSignal, aspect_gammas, compute_weights, top_k_aspects


class UserSignalSource(Protocol):
    """Supplies a user's past aspect signals and (optionally) stores weights."""

    def get_user_aspect_signals(self, user_id: str) -> list[AspectSignal]: ...

    def upsert_user_preferences(
        self, user_id: str, weights: dict[str, float], updated_ts: int
    ) -> None: ...


class DynamicUserProfilingAgent:
    def __init__(self, source: UserSignalSource, lambda_per_day: float) -> None:
        self._source = source
        self._lambda = lambda_per_day

    def profile(
        self, user_id: str, t_query_ms: int, top_k: int, persist: bool = True
    ) -> dict[str, float]:
        """Return the user's normalized aspect-preference weights at t_query.

        Persists the weights back to the source when `persist` is True and the
        source supports it.
        """
        signals = self._source.get_user_aspect_signals(user_id)
        weights = compute_weights(signals, t_query_ms, self._lambda)
        if persist and hasattr(self._source, "upsert_user_preferences"):
            self._source.upsert_user_preferences(user_id, weights, t_query_ms)
        return weights

    def top_aspects(
        self, user_id: str, t_query_ms: int, top_k: int
    ) -> list[tuple[str, float]]:
        weights = self.profile(user_id, t_query_ms, top_k, persist=False)
        return top_k_aspects(weights, top_k)

    def profile_gammas(self, user_id: str, t_query_ms: int) -> dict[str, float]:
        """Signed aspect salience for HGT user-embedding injection at query time."""
        signals = self._source.get_user_aspect_signals(user_id)
        return aspect_gammas(signals, t_query_ms, self._lambda)
