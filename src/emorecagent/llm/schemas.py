"""Pydantic schemas for structured LLM output.

Downstream agents bind these via `with_structured_output` so extraction,
judging, reasoning, and reflection all share typed contracts.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_TRIPLES_PER_REVIEW = 24
SENTIMENTS = frozenset({"positive", "negative", "neutral"})
_SENTIMENT_ALIASES: dict[str, str] = {
    "pos": "positive",
    "positive": "positive",
    "neg": "negative",
    "negative": "negative",
    "neu": "neutral",
    "neutral": "neutral",
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AbsaTriple(_Strict):
    """One validated aspect–opinion–sentiment triple."""

    aspect: str = Field(description="Canonical aspect name, lowercase")
    opinion: str = Field(description="Opinion phrase from the review")
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, value: object) -> float:
        return _coerce_confidence(value)


class TripleSet(_Strict):
    """ABSA extraction / judge output: a list of triples."""

    triples: list[AbsaTriple] = Field(default_factory=list)


class HybridAbsaVerdict(_Strict):
    """Hybrid agent validate-step output."""

    triples: list[AbsaTriple] = Field(default_factory=list)
    needs_repair: bool = False
    missing_aspect_hints: list[str] = Field(default_factory=list)


def _normalize_sentiment(raw: object) -> str | None:
    if raw is None:
        return None
    key = str(raw).strip().lower()
    return _SENTIMENT_ALIASES.get(key)


def _coerce_confidence(raw: object) -> float:
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return 1.0
    if val > 1.0:
        val /= 100.0
    return max(0.0, min(1.0, val))


def coerce_triple_set(data: TripleSet | dict[str, Any] | str) -> TripleSet:
    """Best-effort parse: drop invalid rows, dedupe, cap list length."""
    if isinstance(data, TripleSet):
        raw_triples = [t.model_dump() for t in data.triples]
    elif isinstance(data, str):
        raw_triples = json.loads(data).get("triples", [])
    else:
        raw_triples = data.get("triples", [])

    seen: set[tuple[str, str, str]] = set()
    cleaned: list[AbsaTriple] = []
    for row in raw_triples:
        if not isinstance(row, dict):
            continue
        aspect = str(row.get("aspect") or "").strip().lower()
        opinion = str(row.get("opinion") or "").strip()
        sentiment = _normalize_sentiment(row.get("sentiment"))
        if not aspect or not opinion or sentiment is None:
            continue
        if sentiment not in SENTIMENTS:
            continue
        key = (aspect, opinion, sentiment)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(
            AbsaTriple(
                aspect=aspect,
                opinion=opinion,
                sentiment=sentiment,  # type: ignore[arg-type]
                confidence=_coerce_confidence(row.get("confidence", 1.0)),
            )
        )
        if len(cleaned) >= MAX_TRIPLES_PER_REVIEW:
            break
    return TripleSet(triples=cleaned)


def coerce_hybrid_verdict(
    data: HybridAbsaVerdict | dict[str, Any] | str,
) -> HybridAbsaVerdict:
    """Parse hybrid validate output; normalize triple confidences."""
    if isinstance(data, HybridAbsaVerdict):
        raw = data.model_dump()
    elif isinstance(data, str):
        raw = json.loads(data)
    else:
        raw = data
    triples = coerce_triple_set({"triples": raw.get("triples", [])})
    hints = raw.get("missing_aspect_hints") or []
    return HybridAbsaVerdict(
        triples=triples.triples,
        needs_repair=bool(raw.get("needs_repair", False)),
        missing_aspect_hints=[str(h) for h in hints if str(h).strip()],
    )


def salvage_triple_set_from_error(exc: BaseException) -> TripleSet | None:
    """Recover a partial TripleSet from a structured-output parse failure."""
    llm_output = getattr(exc, "llm_output", None)
    if llm_output is not None:
        try:
            return coerce_triple_set(llm_output)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    text = str(exc)
    match = re.search(r'\{"triples"\s*:', text)
    if not match:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[match.start() :])
        return coerce_triple_set(obj)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def salvage_hybrid_verdict_from_error(exc: BaseException) -> HybridAbsaVerdict | None:
    """Recover HybridAbsaVerdict when structured parsing fails (e.g. bad confidence)."""
    llm_output = getattr(exc, "llm_output", None)
    if llm_output is not None:
        try:
            return coerce_hybrid_verdict(llm_output)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    text = str(exc)
    match = re.search(r'\{"triples"\s*:', text)
    if not match:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[match.start() :])
        return coerce_hybrid_verdict(obj)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


class ReflectionVerdict(_Strict):
    """Reflection Agent structured verdict."""

    approved: bool
    critique: str = ""
    violated_constraints: list[str] = Field(default_factory=list)


class ExplanationClaims(_Strict):
    """Rationalized explanation claims for faithfulness scoring."""

    summary: str
    cited_aspects: list[str] = Field(default_factory=list)
    claims: list[str] = Field(default_factory=list)
