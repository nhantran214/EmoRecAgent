"""In-memory knowledge graph for offline eval and tests (no Docker required)."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..data.types import Interaction
from ..llm.schemas import AbsaTriple
from ..scoring.dynamic_weights import AspectSignal
from ..scoring.sentiment_agg import rescale


@dataclass
class InMemoryKG:
    """Minimal KG implementing the repository contract used by agents and eval."""

    interactions: list[Interaction] = field(default_factory=list)
    signals: list[tuple[str, str, str, float, int]] = field(default_factory=list)
    sentiments: dict[tuple[str, str], tuple[float, int, int]] = field(
        default_factory=dict
    )
    preferences: dict[tuple[str, str], tuple[float, int]] = field(default_factory=dict)

    def upsert_interaction(self, interaction: Interaction) -> None:
        self.interactions.append(interaction)

    def upsert_triples(
        self, user_id: str, item_id: str, triples: list[AbsaTriple], ts: int
    ) -> None:
        for t in triples:
            pol = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}[t.sentiment]
            self.signals.append((user_id, item_id, t.aspect, pol, ts))

    def upsert_item_sentiment(
        self, item_id: str, aspect: str, score: float, n_support: int, ts: int
    ) -> None:
        self.sentiments[(item_id, aspect)] = (score, n_support, ts)

    def get_item_aspects(self, item_id: str) -> dict[str, float]:
        """Raw E_i(a) in [-1, 1]."""
        return {
            aspect: score
            for (iid, aspect), (score, _, _) in self.sentiments.items()
            if iid == item_id
        }

    def get_item_aspect_sentiment(
        self, item_id: str, *, rescaled: bool = True
    ) -> dict[str, float]:
        raw = self.get_item_aspects(item_id)
        if not rescaled:
            return raw
        return {a: rescale(v) for a, v in raw.items()}

    def get_aspect_support(self, item_id: str) -> dict[str, int]:
        return {
            aspect: n_sup
            for (iid, aspect), (_, n_sup, _) in self.sentiments.items()
            if iid == item_id
        }

    def get_user_signals(self, user_id: str, before_ts: int) -> list[AspectSignal]:
        out: list[AspectSignal] = []
        for uid, _, aspect, pol, ts in self.signals:
            if uid == user_id and ts < before_ts:
                out.append(
                    AspectSignal(aspect=aspect, polarity=pol, timestamp_ms=ts)
                )
        return sorted(out, key=lambda s: s.timestamp_ms)

    def upsert_user_preference(
        self, user_id: str, aspect: str, weight: float, updated_ts: int
    ) -> None:
        self.preferences[(user_id, aspect)] = (weight, updated_ts)
