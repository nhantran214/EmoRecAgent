"""Reasoning & Recommendation Agent — listwise LLM rerank over a scored candidate pool."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from ..llm.client import LLMClient, LLMError
from ..llm.prompts import REASONING_RANK_BATCH_V1, REASONING_RANK_V1, format_prompt
from ..llm.schemas import ranking_max_tokens
from ..scoring.score import ScoreBreakdown, rank_items

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True, slots=True)
class BatchRowContext:
    """One row's inputs for a batched LLM pool rerank."""

    row_id: str
    user_id: str
    weights: dict[str, float]
    pool: list[str]
    breakdowns: dict[str, ScoreBreakdown]
    numeric_order: list[str]


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
    ranked_pool_order: list[str] = field(default_factory=list)


def _salient_aspects(weights: dict[str, float], top_k: int = 3) -> list[str]:
    return [
        a
        for a, _ in sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    ]


def expand_pool_with_sar(
    core_pool: list[str],
    strong_source: AspectStrongSource,
    weights: dict[str, float],
    *,
    exclude: set[str],
    pool_size: int,
    aspect_recall_max: int = 15,
    top_aspects: int = 3,
) -> list[str]:
    """Merge retriever/HGT core with aspect-recall items (SAR), cap at pool_size."""
    salient = _salient_aspects(weights, top_k=top_aspects)
    aspect_items = (
        strong_source.items_strong_on(salient, aspect_recall_max, exclude)
        if salient
        else []
    )
    merged: list[str] = []
    seen: set[str] = set()
    for item in list(core_pool) + aspect_items:
        if item in exclude or item in seen:
            continue
        merged.append(item)
        seen.add(item)
        if len(merged) >= pool_size:
            break
    return merged


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
    salient = _salient_aspects(weights, top_k=top_aspects)
    aspect_items = (
        aspect_source.items_strong_on(salient, pool_size, exclude) if salient else []
    )

    cf_seed = list(exclude) + aspect_items
    if len(cf_seed) < pool_size:
        cf_seed.extend([f"__cf_pad_{i}" for i in range(pool_size - len(cf_seed))])
    cf_scores = cf_scorer.score(user_id, cf_seed[: pool_size * 2])
    cf_top = sorted(
        (i for i in cf_scores if i not in exclude and not i.startswith("__cf_pad_")),
        key=lambda i: (-cf_scores[i], i),
    )[:pool_size]

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
        aspect_recall_max: int = 15,
        ranking_num_predict: int = 4096,
    ) -> None:
        self._cf = cf_scorer
        self._aspects = aspect_source
        self._strong = strong_source
        self._llm = llm
        self._alpha = alpha
        self._pool_size = pool_size
        self._aspect_recall_max = aspect_recall_max
        self._ranking_num_predict = ranking_num_predict

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
            pool = expand_pool_with_sar(
                list(pool_override),
                self._strong,
                weights,
                exclude=exclude,
                pool_size=self._pool_size,
                aspect_recall_max=self._aspect_recall_max,
            )
        else:
            pool = build_candidate_pool(
                self._cf,
                self._strong,
                user_id,
                weights,
                pool_size=self._pool_size,
                exclude=exclude,
            )

        filtered, item_aspect_maps = self._filter_pool(
            pool, constraints, price_lookup
        )
        ranked = self._numeric_rank(user_id, filtered, weights, item_aspect_maps)
        numeric_order = [item for item, _ in ranked]
        breakdowns = {item: bd for item, bd in ranked}

        ranked_pool_order = self._resolve_pool_order(
            user_id,
            weights,
            filtered,
            numeric_order,
            breakdowns,
            use_llm_cot=use_llm_cot,
        )
        top = [(item, breakdowns[item]) for item in ranked_pool_order[:k] if item in breakdowns]
        recommendations = [
            Recommendation(item_id=item, breakdown=bd, rank=i + 1)
            for i, (item, bd) in enumerate(top)
        ]
        rationale = self._template_rationale(weights, recommendations)
        return ReasoningResult(
            recommendations=recommendations,
            rationale=rationale,
            candidate_pool=filtered,
            breakdowns=breakdowns,
            ranked_pool_order=ranked_pool_order,
        )

    def repair_ranked_pool(
        self,
        ranked_order: list[str],
        pool: list[str],
        breakdowns: dict[str, ScoreBreakdown],
        *,
        constraints: ReasoningConstraints | None,
        price_lookup: dict[str, float | None] | None,
    ) -> list[str]:
        """Constraint-only repair for reflection retry (no LLM rerank)."""
        numeric_order = [
            item
            for item, _ in sorted(
                ((i, breakdowns[i].total) for i in pool if i in breakdowns),
                key=lambda kv: (-kv[1], kv[0]),
            )
        ]
        kept: list[str] = []
        for item in ranked_order:
            aspects = self._aspects.get_item_aspects(item)
            if _passes_constraints(item, constraints, price_lookup, aspects):
                kept.append(item)
        seen = set(kept)
        for item in numeric_order:
            if item in seen:
                continue
            aspects = self._aspects.get_item_aspects(item)
            if _passes_constraints(item, constraints, price_lookup, aspects):
                kept.append(item)
                seen.add(item)
        logger.info("reflection_repair=true pool_size=%s", len(kept))
        return kept

    def llm_rerank_pool_batch(
        self,
        contexts: list[BatchRowContext],
        *,
        use_llm_cot: bool = True,
    ) -> dict[str, list[str]]:
        """Batched listwise rerank; missing rows fall back to numeric order."""
        if not contexts:
            return {}
        fallback = {ctx.row_id: list(ctx.numeric_order) for ctx in contexts}
        if not use_llm_cot or self._llm is None:
            return fallback
        blocks = [self._format_batch_row_block(ctx) for ctx in contexts]
        prompt = format_prompt(
            REASONING_RANK_BATCH_V1,
            task_blocks="\n\n---\n\n".join(blocks),
        )
        pools = {ctx.row_id: ctx.pool for ctx in contexts}
        batch_max_tokens = ranking_max_tokens(
            [len(ctx.pool) for ctx in contexts],
            cap=self._ranking_num_predict,
        )
        try:
            coerced = self._llm.invoke_batch_ranking_json(
                prompt,
                pools_by_row=pools,
                max_tokens=batch_max_tokens,
            )
            return {
                ctx.row_id: coerced.get(ctx.row_id, fallback[ctx.row_id])
                for ctx in contexts
            }
        except LLMError as exc:
            logger.warning("batch_ranking_fallback=true reason=%s", exc)
            return fallback

    def _format_batch_row_block(self, ctx: BatchRowContext) -> str:
        top_aspects = sorted(
            ctx.weights.items(), key=lambda kv: (-kv[1], kv[0])
        )[:3]
        aspect_str = ", ".join(f"{a}: {w:.2f}" for a, w in top_aspects)
        cards: list[str] = []
        for item in ctx.numeric_order:
            bd = ctx.breakdowns[item]
            drivers = sorted(
                bd.aspect_contributions.items(), key=lambda kv: -kv[1]
            )[:2]
            driver_str = ", ".join(f"{a}:{c:.2f}" for a, c in drivers) or "none"
            cards.append(
                f"{item} | S={bd.total:.3f} | drivers={driver_str} | base={bd.base_contribution:.3f}"
            )
        return (
            f"row_id: {ctx.row_id}\n"
            f"user_id: {ctx.user_id}\n"
            f"top_aspects: {aspect_str}\n"
            f"candidates:\n" + "\n".join(cards)
        )

    def _filter_pool(
        self,
        pool: list[str],
        constraints: ReasoningConstraints | None,
        price_lookup: dict[str, float | None] | None,
    ) -> tuple[list[str], dict[str, dict[str, float]]]:
        filtered: list[str] = []
        item_aspect_maps: dict[str, dict[str, float]] = {}
        for item in pool:
            aspects = self._aspects.get_item_aspects(item)
            item_aspect_maps[item] = aspects
            if _passes_constraints(item, constraints, price_lookup, aspects):
                filtered.append(item)
        return filtered, item_aspect_maps

    def _numeric_rank(
        self,
        user_id: str,
        filtered: list[str],
        weights: dict[str, float],
        item_aspect_maps: dict[str, dict[str, float]],
    ) -> list[tuple[str, ScoreBreakdown]]:
        s_base = self._cf.score(user_id, filtered)
        return rank_items(self._alpha, s_base, weights, item_aspect_maps)

    def _resolve_pool_order(
        self,
        user_id: str,
        weights: dict[str, float],
        pool: list[str],
        numeric_order: list[str],
        breakdowns: dict[str, ScoreBreakdown],
        *,
        use_llm_cot: bool,
    ) -> list[str]:
        if use_llm_cot and self._llm is not None and pool:
            try:
                return self._llm_rerank_pool(
                    user_id, weights, pool, breakdowns, numeric_order
                )
            except LLMError as exc:
                logger.warning("ranking_fallback=true reason=%s", exc)
        return list(numeric_order)

    def _llm_rerank_pool(
        self,
        user_id: str,
        weights: dict[str, float],
        pool: list[str],
        breakdowns: dict[str, ScoreBreakdown],
        numeric_fallback: list[str],
    ) -> list[str]:
        del user_id
        top_aspects = sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        aspect_str = ", ".join(f"{a}: {w:.2f}" for a, w in top_aspects)
        cards: list[str] = []
        for item in numeric_fallback:
            bd = breakdowns[item]
            drivers = sorted(
                bd.aspect_contributions.items(), key=lambda kv: -kv[1]
            )[:2]
            driver_str = ", ".join(f"{a}:{c:.2f}" for a, c in drivers) or "none"
            cards.append(
                f"{item} | S={bd.total:.3f} | drivers={driver_str} | base={bd.base_contribution:.3f}"
            )
        prompt = format_prompt(
            REASONING_RANK_V1,
            top_aspects=aspect_str,
            candidate_cards="\n".join(cards),
        )
        return self._llm.invoke_ranking_json(
            prompt,
            pool_ids=pool,
            max_tokens=ranking_max_tokens([len(pool)], cap=self._ranking_num_predict),
        )

    def _template_rationale(
        self,
        weights: dict[str, float],
        recs: list[Recommendation],
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
            summary_lines.append(
                f"{rec.item_id}: {driver_str} (S={rec.breakdown.total:.3f})"
            )
        return (
            f"Top aspects: {aspect_str}. "
            f"Ranked by agent listwise order: {'; '.join(summary_lines)}"
        )
