"""U5 KG repository tests using an in-memory store (no Docker required)."""

from __future__ import annotations

from dataclasses import dataclass, field

from emorecagent.data.types import Interaction
from emorecagent.kg.loaders import load_interactions
from emorecagent.llm.schemas import AbsaTriple
from emorecagent.scoring.dynamic_weights import AspectSignal
from emorecagent.scoring.sentiment_agg import ItemAspectTriple, aggregate_raw


@dataclass
class InMemoryKG:
    """Minimal in-memory KG implementing the repository contract for tests."""

    interactions: list[Interaction] = field(default_factory=list)
    signals: list[tuple[str, str, str, float, int]] = field(default_factory=list)
    sentiments: dict[tuple[str, str], tuple[float, int, int]] = field(default_factory=dict)
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
        return {
            aspect: score
            for (iid, aspect), (score, _, _) in self.sentiments.items()
            if iid == item_id
        }

    def get_user_signals(self, user_id: str, before_ts: int) -> list[AspectSignal]:
        out = []
        for uid, _, aspect, pol, ts in self.signals:
            if uid == user_id and ts < before_ts:
                out.append(AspectSignal(aspect=aspect, polarity=pol, timestamp_ms=ts))
        return sorted(out, key=lambda s: s.timestamp_ms)

    def upsert_user_preference(
        self, user_id: str, aspect: str, weight: float, updated_ts: int
    ) -> None:
        self.preferences[(user_id, aspect)] = (weight, updated_ts)


def test_load_interactions_creates_expected_count() -> None:
    kg = InMemoryKG()
    data = [
        Interaction("u1", "i1", 5.0, 1000),
        Interaction("u1", "i2", 4.0, 2000),
    ]
    n = load_interactions(kg, data)
    assert n == 2
    assert len(kg.interactions) == 2


def test_triples_become_user_signals() -> None:
    kg = InMemoryKG()
    triples = [
        AbsaTriple(aspect="scent", opinion="nice", sentiment="positive"),
        AbsaTriple(aspect="pump", opinion="hard", sentiment="negative"),
    ]
    kg.upsert_triples("u1", "i1", triples, ts=5000)
    signals = kg.get_user_signals("u1", before_ts=6000)
    assert len(signals) == 2
    assert signals[0].aspect == "scent"
    assert signals[1].polarity < 0


def test_item_sentiment_query_returns_aspect_map() -> None:
    kg = InMemoryKG()
    kg.upsert_item_sentiment("i1", "scent", 0.8, n_support=10, ts=100)
    kg.upsert_item_sentiment("i1", "texture", -0.2, n_support=3, ts=100)
    aspects = kg.get_item_aspects("i1")
    assert aspects["scent"] == 0.8
    assert aspects["texture"] == -0.2


def test_helpfulness_weighting_favors_high_helpful_reviews() -> None:
    triples_hi = [ItemAspectTriple("scent", 1.0, helpful_vote=10)]
    triples_lo = [ItemAspectTriple("scent", -1.0, helpful_vote=0)]
    hi = aggregate_raw(triples_hi, helpful_cap=10)["scent"]
    lo = aggregate_raw(triples_lo, helpful_cap=10)["scent"]
    mixed = aggregate_raw(triples_hi + triples_lo, helpful_cap=10)["scent"]
    assert hi > mixed > lo
