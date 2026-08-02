"""Neo4j-backed accessors for the LangGraph pipeline."""

from __future__ import annotations

from typing import Protocol

from ..llm.schemas import AbsaTriple
from ..scoring.dynamic_weights import AspectSignal
from ..scoring.sentiment_agg import rescale
from .memory import InMemoryKG
from .repository import KGRepository


class GraphKGBackend(Protocol):
    def get_user_signals(self, user_id: str, before_ts: int) -> list[AspectSignal]: ...

    def get_item_aspects_rescaled(self, item_id: str) -> dict[str, float]: ...

    def get_aspect_support(self, item_id: str) -> dict[str, int]: ...

    def load_user_triples(self, user_id: str, before_ts: int) -> list[AbsaTriple]: ...


class Neo4jGraphKG:
    def __init__(self, repo: KGRepository, *, affective_rescaled: bool = True) -> None:
        self._repo = repo
        self._rescaled = affective_rescaled

    def get_user_signals(self, user_id: str, before_ts: int) -> list[AspectSignal]:
        return self._repo.get_user_signals(user_id, before_ts)

    def get_item_aspects_rescaled(self, item_id: str) -> dict[str, float]:
        raw = self._repo.get_item_aspects(item_id)
        if not self._rescaled:
            return raw
        return {a: rescale(v) for a, v in raw.items()}

    def get_aspect_support(self, item_id: str) -> dict[str, int]:
        return self._repo.get_aspect_support(item_id)

    def load_user_triples(self, user_id: str, before_ts: int) -> list[AbsaTriple]:
        return self._repo.get_user_absa_triples(user_id, before_ts)

    def items_strong_on(
        self,
        aspects: list[str],
        limit: int,
        exclude: set[str],
        *,
        tau: float = 0.65,
        min_support: int = 3,
        weights: dict[str, float] | None = None,
    ) -> list[str]:
        return self._repo.items_strong_on(
            aspects,
            limit,
            exclude,
            tau=tau,
            min_support=min_support,
            weights=weights,
        )


class InMemoryGraphKG:
    """Test/dev backend wrapping InMemoryKG."""

    def __init__(self, kg: InMemoryKG, *, affective_rescaled: bool = True) -> None:
        self._kg = kg
        self._rescaled = affective_rescaled

    def get_user_signals(self, user_id: str, before_ts: int) -> list[AspectSignal]:
        return self._kg.get_user_signals(user_id, before_ts)

    def get_item_aspects_rescaled(self, item_id: str) -> dict[str, float]:
        return self._kg.get_item_aspect_sentiment(
            item_id, rescaled=self._rescaled
        )

    def get_aspect_support(self, item_id: str) -> dict[str, int]:
        return self._kg.get_aspect_support(item_id)

    def load_user_triples(self, user_id: str, before_ts: int) -> list[AbsaTriple]:
        out: list[AbsaTriple] = []
        for uid, _item, aspect, pol, ts in self._kg.signals:
            if uid != user_id or ts >= before_ts:
                continue
            if pol > 0.1:
                sentiment = "positive"
            elif pol < -0.1:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            out.append(
                AbsaTriple(
                    aspect=aspect,
                    opinion="",
                    sentiment=sentiment,
                    confidence=1.0,
                )
            )
        return out

    def items_strong_on(
        self,
        aspects: list[str],
        limit: int,
        exclude: set[str],
        *,
        tau: float = 0.65,
        min_support: int = 3,
        weights: dict[str, float] | None = None,
    ) -> list[str]:
        return self._kg.items_strong_on(
            aspects,
            limit,
            exclude,
            tau=tau,
            min_support=min_support,
            weights=weights,
        )
