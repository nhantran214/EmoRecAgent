"""Reasoning & Recommendation Agent — CoT matching over a scored candidate pool."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..llm.client import LLMClient
from ..llm.prompts import REASONING_COT_V1, format_prompt
from ..scoring.score import ScoreBreakdown, rank_items


class CFScorer(Protocol):
    def score(self, user_id: str, candidates: list[str]) -> dict[str, float]: ...


class ItemAspectSource(Protocol):
    """Returns Ê_i(a) in [0, 1] per item."""

    def get_item_aspects(self, item_id: str) -> dict[str, float]: ...


class AspectStrongSource(Protocol):
    """Items that rate highly on the user's salient aspects."""

    def items_strong_on(
        self, aspects: list[str], limit: int, exclude: set[str]
    ) -> list[str]: ...


@dataclass(frozen=True, slots=True)
class Recommendation:
    item_id: str
    breakdown: ScoreBreakdown
    rank: int


@dataclass
class ReasoningConstraints:
    """Hard filters applied before scoring (from reflection critique)."""

    exclude_items: set[str] = field(default_factory=set)
    max_price: float | None = None
    min_aspect_score: dict[str, float] = field(default_factory=dict)


@dataclass
class ReasoningResult:
    recommendations: list[Recommendation]
    rationale: str
    candidate_pool: list[str]
    breakdowns: dict[str, ScoreBreakdown]


def build_candidate_pool(
    cf_scorer: CFScorer,
    aspect_source: AspectStrongSource,
    user_id: str,
    weights: dict[str, float],
    *,
    pool_size: int,
    exclude: set[str],
    top_aspects: int = 3,
) -> list[str]:
    """CF top-N ∪ items strong on the user's top aspects."""
    # Placeholder universe: aspect-strong items seed the pool; CF ranks within it.
    salient = [a for a, _ in sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))[:top_aspects]]
    aspect_items = aspect_source.items_strong_on(salient, pool_size, exclude) if salient else []

    cf_seed = list(exclude) + aspect_items
    # Expand with synthetic high-score slots so CF has candidates beyond train history.
    if len(cf_seed) < pool_size:
        cf_seed.extend([f"__cf_pad_{i}" for i in range(pool_size - len(cf_seed))])
    cf_scores = cf_scorer.score(user_id, cf_seed[: pool_size * 2])
    cf_top = sorted(
        (i for i in cf_scores if i not in exclude and not i.startswith("__cf_pad_")),
        key=lambda i: (-cf_scores[i], i),
    )[: pool_size]

    pool: list[str] = []
    seen: set[str] = set()
    for item in aspect_items + cf_top:
        if item in exclude or item in seen:
            continue
        pool.append(item)
        seen.add(item)
        if len(pool) >= pool_size:
            break
    return pool


def _passes_constraints(
    item_id: str,
    constraints: ReasoningConstraints | None,
    price_lookup: dict[str, float | None] | None,
    item_aspects: dict[str, float],
) -> bool:
    if constraints is None:
        return True
    if item_id in constraints.exclude_items:
        return False
    if constraints.max_price is not None and price_lookup is not None:
        price = price_lookup.get(item_id)
        if price is not None and price > constraints.max_price:
            return False
    for aspect, min_score in constraints.min_aspect_score.items():
        if item_aspects.get(aspect, 0.0) < min_score:
            return False
    return True


class ReasoningAgent:
    def __init__(
        self,
        cf_scorer: CFScorer,
        aspect_source: ItemAspectSource,
        strong_source: AspectStrongSource,
        llm: LLMClient | None,
        *,
        alpha: float,
        pool_size: int = 200,
    ) -> None:
        self._cf = cf_scorer
        self._aspects = aspect_source
        self._strong = strong_source
        self._llm = llm
        self._alpha = alpha
        self._pool_size = pool_size

    def recommend(
        self,
        user_id: str,
        weights: dict[str, float],
        *,
        exclude: set[str],
        k: int,
        constraints: ReasoningConstraints | None = None,
        price_lookup: dict[str, float | None] | None = None,
        use_llm_cot: bool = True,
        pool_override: list[str] | None = None,
    ) -> ReasoningResult:
        if pool_override is not None:
            pool = list(pool_override)[: self._pool_size]
        else:
            pool = build_candidate_pool(
                self._cf,
                self._strong,
                user_id,
                weights,
                pool_size=self._pool_size,
                exclude=exclude,
            )

        filtered: list[str] = []
        item_aspect_maps: dict[str, dict[str, float]] = {}
        for item in pool:
            aspects = self._aspects.get_item_aspects(item)
            item_aspect_maps[item] = aspects
            if _passes_constraints(item, constraints, price_lookup, aspects):
                filtered.append(item)

        s_base = self._cf.score(user_id, filtered)
        ranked = rank_items(self._alpha, s_base, weights, item_aspect_maps)
        top = ranked[:k]

        breakdowns = {item: bd for item, bd in ranked}
        recommendations = [
            Recommendation(item_id=item, breakdown=bd, rank=i + 1)
            for i, (item, bd) in enumerate(top)
        ]

        rationale = self._build_rationale(
            weights, recommendations, breakdowns, use_llm=use_llm_cot
        )
        return ReasoningResult(
            recommendations=recommendations,
            rationale=rationale,
            candidate_pool=filtered,
            breakdowns=breakdowns,
        )

    def _build_rationale(
        self,
        weights: dict[str, float],
        recs: list[Recommendation],
        breakdowns: dict[str, ScoreBreakdown],
        *,
        use_llm: bool,
    ) -> str:
        if not recs:
            return "No candidates satisfied the current constraints."
        top_aspects = sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        aspect_str = ", ".join(f"{a} ({w:.2f})" for a, w in top_aspects)
        summary_lines = []
        for rec in recs[:3]:
            drivers = sorted(
                rec.breakdown.aspect_contributions.items(),
                key=lambda kv: -kv[1],
            )[:2]
            driver_str = ", ".join(f"{a} (+{c:.3f})" for a, c in drivers) or "CF base"
            summary_lines.append(f"{rec.item_id}: {driver_str} (S={rec.breakdown.total:.3f})")
        candidate_summary = "; ".join(summary_lines)

        if use_llm and self._llm is not None:
            prompt = format_prompt(
                REASONING_COT_V1,
                top_aspects=aspect_str,
                candidate_summary=candidate_summary,
            )
            return self._llm.invoke_text(prompt)

        return (
            f"Top aspects: {aspect_str}. "
            f"Ranked by S(u,i): {candidate_summary}"
        )
