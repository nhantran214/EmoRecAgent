"""Typed KG repository — no Cypher outside this module (U5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from typing import Any

from ..data.types import Interaction
from ..llm.schemas import AbsaTriple
from ..scoring.dynamic_weights import AspectSignal


@dataclass(frozen=True, slots=True)
class ItemAspectSentiment:
    aspect: str
    score: float       # E_i(a) raw in [-1, 1] or rescaled per caller
    n_support: int
    ts: int | None = None


class KGStore(Protocol):
    """Test double interface mirroring KGRepository."""

    def upsert_interaction(self, interaction: Interaction) -> None: ...
    def upsert_triples(
        self, user_id: str, item_id: str, triples: list[AbsaTriple], ts: int
    ) -> None: ...
    def upsert_item_sentiment(
        self, item_id: str, aspect: str, score: float, n_support: int, ts: int
    ) -> None: ...
    def get_item_aspects(self, item_id: str) -> dict[str, float]: ...
    def get_user_signals(self, user_id: str, before_ts: int) -> list[AspectSignal]: ...
    def upsert_user_preference(
        self, user_id: str, aspect: str, weight: float, updated_ts: int
    ) -> None: ...


class KGRepository:
    def __init__(self, driver: Any) -> None:
        self._driver = driver

    def upsert_interaction(self, interaction: Interaction) -> None:
        query = """
        MERGE (u:User {id: $user_id})
        MERGE (i:Item {asin: $item})
        MERGE (u)-[r:REVIEWED]->(i)
        SET r.rating = $rating, r.ts = $ts, r.helpful_vote = $helpful_vote
        """
        with self._driver.session() as session:
            session.run(
                query,
                user_id=interaction.user_id,
                item=interaction.item,
                rating=interaction.rating,
                ts=interaction.timestamp,
                helpful_vote=interaction.helpful_vote,
            )

    def upsert_triples(
        self,
        user_id: str,
        item_id: str,
        triples: list[AbsaTriple],
        ts: int,
    ) -> None:
        """Attach ABSA triples to a user review context (aspect nodes + signals)."""
        if not triples:
            return
        rows = [
            {
                "aspect": t.aspect,
                "polarity": _sentiment_to_polarity(t.sentiment),
                "opinion": t.opinion,
                "confidence": t.confidence,
            }
            for t in triples
        ]
        query = """
        MERGE (u:User {id: $user_id})
        MERGE (i:Item {asin: $item_id})
        UNWIND $rows AS row
        MERGE (a:Aspect {name: row.aspect})
        MERGE (u)-[s:SIGNAL {ts: $ts, item: $item_id}]->(a)
        SET s.polarity = row.polarity,
            s.opinion = row.opinion,
            s.confidence = row.confidence
        """
        with self._driver.session() as session:
            session.run(
                query, user_id=user_id, item_id=item_id, ts=ts, rows=rows
            )

    def upsert_item_sentiment(
        self,
        item_id: str,
        aspect: str,
        score: float,
        n_support: int,
        ts: int,
    ) -> None:
        query = """
        MERGE (i:Item {asin: $item_id})
        MERGE (a:Aspect {name: $aspect})
        MERGE (i)-[r:HAS_SENTIMENT]->(a)
        SET r.score = $score, r.n_support = $n_support, r.ts = $ts
        """
        with self._driver.session() as session:
            session.run(
                query,
                item_id=item_id,
                aspect=aspect,
                score=score,
                n_support=n_support,
                ts=ts,
            )

    def get_item_aspects(self, item_id: str) -> dict[str, float]:
        query = """
        MATCH (i:Item {asin: $item_id})-[r:HAS_SENTIMENT]->(a:Aspect)
        RETURN a.name AS aspect, r.score AS score
        """
        with self._driver.session() as session:
            result = session.run(query, item_id=item_id)
            return {rec["aspect"]: float(rec["score"]) for rec in result}

    def get_user_signals(self, user_id: str, before_ts: int) -> list[AspectSignal]:
        query = """
        MATCH (u:User {id: $user_id})-[s:SIGNAL]->(a:Aspect)
        WHERE s.ts < $before_ts
        RETURN a.name AS aspect, s.polarity AS polarity, s.ts AS ts
        ORDER BY s.ts
        """
        with self._driver.session() as session:
            result = session.run(query, user_id=user_id, before_ts=before_ts)
            return [
                AspectSignal(
                    aspect=rec["aspect"],
                    polarity=float(rec["polarity"]),
                    timestamp_ms=int(rec["ts"]),
                )
                for rec in result
            ]

    def upsert_user_preference(
        self, user_id: str, aspect: str, weight: float, updated_ts: int
    ) -> None:
        query = """
        MERGE (u:User {id: $user_id})
        MERGE (a:Aspect {name: $aspect})
        MERGE (u)-[p:PREFERS]->(a)
        SET p.weight = $weight, p.updated_ts = $updated_ts
        """
        with self._driver.session() as session:
            session.run(
                query,
                user_id=user_id,
                aspect=aspect,
                weight=weight,
                updated_ts=updated_ts,
            )


def _sentiment_to_polarity(sentiment: str) -> float:
    return {"positive": 1.0, "negative": -1.0, "neutral": 0.0}.get(sentiment, 0.0)
