"""Reflection Agent — deterministic checks + optional LLM judge."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..llm.client import LLMClient
from ..llm.prompts import REFLECTION_JUDGE_V1, format_prompt
from ..llm.schemas import ReflectionVerdict
from ..scoring.score import ScoreBreakdown
from .reasoning_agent import Recommendation


@dataclass
class ReflectionInput:
    recommendations: list[Recommendation]
    breakdowns: dict[str, ScoreBreakdown]
    user_budget: float | None = None
    item_prices: dict[str, float | None] = field(default_factory=dict)
    item_e_hat: dict[str, dict[str, float]] = field(default_factory=dict)
    recent_complaint_aspects: list[str] = field(default_factory=list)
    category_price_percentile: dict[str, float] = field(default_factory=dict)
    complaint_threshold: float = 0.4


class ReflectionAgent:
    def __init__(
        self,
        llm: LLMClient | None = None,
        *,
        use_llm_judge: bool = True,
    ) -> None:
        self._llm = llm
        self._use_llm = use_llm_judge

    def evaluate(self, ctx: ReflectionInput) -> ReflectionVerdict:
        violations: list[str] = []

        for rec in ctx.recommendations:
            price = ctx.item_prices.get(rec.item_id)
            if ctx.user_budget is not None and price is not None:
                if price > ctx.user_budget:
                    violations.append(f"budget:{rec.item_id}:{price}>{ctx.user_budget}")
            elif ctx.user_budget is not None and price is None:
                pct = ctx.category_price_percentile.get(rec.item_id)
                if pct is not None and pct > 0.75:
                    violations.append(f"price_percentile:{rec.item_id}:{pct:.2f}")

            aspects = ctx.item_e_hat.get(rec.item_id, {})
            for asp in ctx.recent_complaint_aspects:
                if aspects.get(asp, 0.5) < ctx.complaint_threshold:
                    violations.append(f"complaint_aspect:{rec.item_id}:{asp}")

        if violations:
            return ReflectionVerdict(
                approved=False,
                critique="; ".join(violations),
                violated_constraints=violations,
            )

        if self._use_llm and self._llm is not None and ctx.recommendations:
            return self._llm_judge(ctx)

        return ReflectionVerdict(approved=True, critique="", violated_constraints=[])

    def constraints_from_verdict(self, verdict: ReflectionVerdict):
        """Translate reflection failures into ReasoningConstraints fields."""
        from .reasoning_agent import ReasoningConstraints

        exclude: set[str] = set()
        max_price: float | None = None
        min_aspect: dict[str, float] = {}

        for v in verdict.violated_constraints:
            if v.startswith("budget:"):
                parts = v.split(":")
                if len(parts) >= 2:
                    exclude.add(parts[1])
            elif v.startswith("complaint_aspect:"):
                parts = v.split(":")
                if len(parts) >= 3:
                    min_aspect[parts[2]] = 0.5

        return ReasoningConstraints(
            exclude_items=exclude,
            max_price=max_price,
            min_aspect_score=min_aspect,
        )

    def _llm_judge(self, ctx: ReflectionInput) -> ReflectionVerdict:
        lines = []
        for rec in ctx.recommendations:
            lines.append(f"{rec.item_id} S={rec.breakdown.total:.3f}")
        constraints = []
        if ctx.user_budget is not None:
            constraints.append(f"budget<={ctx.user_budget}")
        if ctx.recent_complaint_aspects:
            constraints.append(
                f"high score on {', '.join(ctx.recent_complaint_aspects)}"
            )
        prompt = format_prompt(
            REFLECTION_JUDGE_V1,
            constraints="\n".join(constraints) or "none",
            recommendations="\n".join(lines),
        )
        try:
            return self._llm.invoke_structured(prompt, ReflectionVerdict)
        except Exception:
            return ReflectionVerdict(approved=True, critique="", violated_constraints=[])
