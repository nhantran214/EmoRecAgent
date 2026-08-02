"""Shared LangGraph state."""

from __future__ import annotations

from typing import TypedDict

from ..explain.rationalize import RationalizedExplanation
from ..llm.schemas import AbsaTriple, ReflectionVerdict
from ..scoring.score import ScoreBreakdown


class EmoRecState(TypedDict, total=False):
    user_id: str
    t_query_ms: int
    exclude_items: set[str]
    triples: list[AbsaTriple]
    weights: dict[str, float]
    candidate_pool: list[str]
    eval_candidates: list[str]
    breakdowns: dict[str, ScoreBreakdown]
    recommendations: list[str]
    ranked_pool_order: list[str]
    rationale: str
    reflection: ReflectionVerdict
    reflection_iters: int
    approved: bool
    user_budget: float | None
    item_prices: dict[str, float | None]
    item_e_hat: dict[str, dict[str, float]]
    aspect_support: dict[str, dict[str, int]]
    recent_complaint_aspects: list[str]
    explanation: RationalizedExplanation
    constraints_json: dict  # serialized ReasoningConstraints for re-entry
