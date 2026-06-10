"""ABSA extract → judge → normalize pipeline (U4)."""

from __future__ import annotations

from dataclasses import dataclass

from ..llm.schemas import AbsaTriple, TripleSet
from .cache import AbsaCache
from .extractor import AbsaExtractor
from .judge import AbsaJudge
from .normalize import normalize_aspect


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    review_id: str
    text: str


class AbsaPipeline:
    def __init__(
        self,
        extractor: AbsaExtractor,
        judge: AbsaJudge,
        cache: AbsaCache | None = None,
    ) -> None:
        self.extractor = extractor
        self.judge = judge
        self.cache = cache

    def process(self, record: ReviewRecord, *, use_cache: bool = True) -> TripleSet:
        if use_cache and self.cache is not None:
            hit = self.cache.get(record.review_id)
            if hit is not None:
                return hit

        candidates = self.extractor.extract(record.text).triples
        validated = self.judge.judge(record.text, candidates).triples
        normalized = [
            AbsaTriple(
                aspect=normalize_aspect(t.aspect),
                opinion=t.opinion,
                sentiment=t.sentiment,
                confidence=t.confidence,
            )
            for t in validated
        ]
        # de-duplicate by (aspect, sentiment) keeping highest confidence
        best: dict[tuple[str, str], AbsaTriple] = {}
        for t in normalized:
            key = (t.aspect, t.sentiment)
            if key not in best or t.confidence > best[key].confidence:
                best[key] = t
        result = TripleSet(triples=list(best.values()))

        if use_cache and self.cache is not None:
            self.cache.put(record.review_id, result)
        return result
