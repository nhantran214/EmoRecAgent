"""Typed KG repository — no Cypher outside this module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from typing import Any

from ..data.types import Interaction
from ..llm.schemas import AbsaTriple
from ..scoring.dynamic_weights import AspectSignal
from ..scoring.sentiment_agg import rescale


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
        SET r.rating = $rating, r.ts = $ts, r.helpful_vote = $helpful_vote,
            r.verified_purchase = $verified_purchase
        """
        with self._driver.session() as session:
            session.run(
                query,
                user_id=interaction.user_id,
                item=interaction.item,
                rating=interaction.rating,
                ts=interaction.timestamp,
                helpful_vote=interaction.helpful_vote,
                verified_purchase=interaction.verified_purchase,
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
            s.confidence = row.confidence,
            s.item = $item_id,
            s.ts = $ts
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

    def get_aspect_support(self, item_id: str) -> dict[str, int]:
        query = """
        MATCH (i:Item {asin: $item_id})-[r:HAS_SENTIMENT]->(a:Aspect)
        RETURN a.name AS aspect, r.n_support AS n_support
        """
        with self._driver.session() as session:
            result = session.run(query, item_id=item_id)
            return {rec["aspect"]: int(rec["n_support"]) for rec in result}

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
        if not aspects:
            return []
        query = """
        MATCH (i:Item)-[r:HAS_SENTIMENT]->(a:Aspect)
        WHERE a.name IN $aspects AND r.n_support >= $min_support
        RETURN i.asin AS item_id, a.name AS aspect, r.score AS score
        """
        scores: dict[str, float] = {}
        with self._driver.session() as session:
            result = session.run(
                query,
                aspects=aspects,
                min_support=min_support,
            )
            for rec in result:
                item_id = str(rec["item_id"])
                if item_id in exclude:
                    continue
                aspect = str(rec["aspect"])
                raw = float(rec["score"])
                if rescale(raw) < tau:
                    continue
                w = (weights or {}).get(aspect, 1.0)
                scores[item_id] = scores.get(item_id, 0.0) + w * rescale(raw)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [item for item, _ in ranked[:limit]]

    def get_user_absa_triples(self, user_id: str, before_ts: int) -> list[AbsaTriple]:
        query = """
        MATCH (u:User {id: $user_id})-[s:SIGNAL]->(a:Aspect)
        WHERE s.ts < $before_ts
        RETURN a.name AS aspect,
               coalesce(s.polarity, 0.0) AS polarity,
               coalesce(s.opinion, '') AS opinion,
               coalesce(s.confidence, 1.0) AS confidence
        ORDER BY s.ts
        """
        with self._driver.session() as session:
            result = session.run(
                query, user_id=user_id, before_ts=before_ts
            )
            triples: list[AbsaTriple] = []
            for rec in result:
                pol = float(rec["polarity"] or 0.0)
                if pol > 0.1:
                    sentiment = "positive"
                elif pol < -0.1:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
                triples.append(
                    AbsaTriple(
                        aspect=str(rec["aspect"]),
                        opinion=str(rec.get("opinion") or ""),
                        sentiment=sentiment,
                        confidence=float(rec.get("confidence") or 1.0),
                    )
                )
            return triples

    def get_user_signals(self, user_id: str, before_ts: int) -> list[AspectSignal]:
        query = """
        MATCH (u:User {id: $user_id})-[s:SIGNAL]->(a:Aspect)
        WHERE s.ts < $before_ts
        RETURN a.name AS aspect,
               coalesce(s.polarity, 0.0) AS polarity,
               coalesce(s.ts, 0) AS ts
        ORDER BY s.ts
        """
        with self._driver.session() as session:
            result = session.run(query, user_id=user_id, before_ts=before_ts)
            return [
                AspectSignal(
                    aspect=rec["aspect"],
                    polarity=float(rec["polarity"] or 0.0),
                    timestamp_ms=int(rec["ts"] or 0),
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
