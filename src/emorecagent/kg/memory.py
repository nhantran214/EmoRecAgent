"""In-memory knowledge graph for offline eval and tests (no Docker required)."""

from __future__ import annotations

from collections import defaultdict
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
    _sentiments_by_item: dict[str, dict[str, tuple[float, int, int]]] = field(
        default_factory=lambda: defaultdict(dict), repr=False
    )
    _signals_by_user: dict[str, list[tuple[str, str, float, int]]] = field(
        default_factory=lambda: defaultdict(list), repr=False
    )
    _items_by_aspect: dict[str, dict[str, tuple[float, int]]] = field(
        default_factory=lambda: defaultdict(dict), repr=False
    )

    def upsert_interaction(self, interaction: Interaction) -> None:
        self.interactions.append(interaction)

    def upsert_triples(
        self, user_id: str, item_id: str, triples: list[AbsaTriple], ts: int
    ) -> None:
        for t in triples:
            pol = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}[t.sentiment]
            row = (user_id, item_id, t.aspect, pol, ts)
            self.signals.append(row)
            self._signals_by_user[user_id].append((item_id, t.aspect, pol, ts))

    def upsert_item_sentiment(
        self, item_id: str, aspect: str, score: float, n_support: int, ts: int
    ) -> None:
        value = (score, n_support, ts)
        self.sentiments[(item_id, aspect)] = value
        self._sentiments_by_item[item_id][aspect] = value
        self._items_by_aspect[aspect][item_id] = (score, n_support)

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
        """Items with high rescaled sentiment on salient aspects (SAR recall)."""
        scores: dict[str, float] = {}
        for aspect in aspects:
            w = (weights or {}).get(aspect, 1.0)
            for item_id, (raw, n_sup) in self._items_by_aspect.get(aspect, {}).items():
                if item_id in exclude or n_sup < min_support:
                    continue
                if rescale(raw) < tau:
                    continue
                scores[item_id] = scores.get(item_id, 0.0) + w * rescale(raw)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [item for item, _ in ranked[:limit]]

    def get_item_aspects(self, item_id: str) -> dict[str, float]:
        """Raw E_i(a) in [-1, 1]."""
        return {
            aspect: score
            for aspect, (score, _, _) in self._sentiments_by_item.get(item_id, {}).items()
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
            for aspect, (_, n_sup, _) in self._sentiments_by_item.get(item_id, {}).items()
        }

    def get_user_signals(self, user_id: str, before_ts: int) -> list[AspectSignal]:
        out: list[AspectSignal] = []
        for _item_id, aspect, pol, ts in self._signals_by_user.get(user_id, ()):
            if ts < before_ts:
                out.append(
                    AspectSignal(aspect=aspect, polarity=pol, timestamp_ms=ts)
                )
        return sorted(out, key=lambda s: s.timestamp_ms)

    def upsert_user_preference(
        self, user_id: str, aspect: str, weight: float, updated_ts: int
    ) -> None:
        self.preferences[(user_id, aspect)] = (weight, updated_ts)
