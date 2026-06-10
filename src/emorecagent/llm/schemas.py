"""Pydantic schemas for structured LLM output (U3).

Downstream agents bind these via `with_structured_output` so extraction,
judging, reasoning, and reflection all share typed contracts.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AbsaTriple(_Strict):
    """One validated aspect–opinion–sentiment triple."""

    aspect: str = Field(description="Canonical aspect name, lowercase")
    opinion: str = Field(description="Opinion phrase from the review")
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class TripleSet(_Strict):
    """ABSA extraction / judge output: a list of triples."""

    triples: list[AbsaTriple] = Field(default_factory=list)


class ReflectionVerdict(_Strict):
    """Reflection Agent structured verdict."""

    approved: bool
    critique: str = ""
    violated_constraints: list[str] = Field(default_factory=list)


class ExplanationClaims(_Strict):
    """Rationalized explanation claims for faithfulness scoring (U10)."""

    summary: str
    cited_aspects: list[str] = Field(default_factory=list)
    claims: list[str] = Field(default_factory=list)
