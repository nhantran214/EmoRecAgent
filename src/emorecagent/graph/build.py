"""Assemble the four-agent LangGraph with reflection loop."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from ..agents.profiling_agent import DynamicUserProfilingAgent
from ..agents.reasoning_agent import ReasoningAgent, ReasoningConstraints
from ..agents.reflection_agent import ReflectionAgent, ReflectionInput
from ..explain.rationalize import explain_recommendation
from ..llm.schemas import AbsaTriple, ReflectionVerdict
from ..scoring.dynamic_weights import AspectSignal
from .state import EmoRecState
from .timing import GraphStageTimer


@dataclass
class GraphDeps:
    profiling: DynamicUserProfilingAgent
    reasoning: ReasoningAgent
    reflection: ReflectionAgent
    load_triples: Callable[[str], list[AbsaTriple]]
    user_signals: Callable[[str, int], list[AspectSignal]]
    aspect_support: Callable[[str], dict[str, int]]
    max_reflection_iters: int = 2
    top_k: int = 5
    stage_timer: GraphStageTimer | None = None


def _timed(stage: str, timer: GraphStageTimer | None, fn: Callable) -> Callable:
    def wrapped(state: EmoRecState) -> EmoRecState:
        if timer is None:
            return fn(state)
        t0 = time.monotonic()
        try:
            return fn(state)
        finally:
            timer.record(stage, time.monotonic() - t0)

    return wrapped


def _constraints_from_state(state: EmoRecState) -> ReasoningConstraints | None:
    raw = state.get("constraints_json")
    if not raw:
        return None
    return ReasoningConstraints(
        exclude_items=set(raw.get("exclude_items", [])),
        max_price=raw.get("max_price"),
        min_aspect_score=dict(raw.get("min_aspect_score", {})),
    )


def _serialize_constraints(c: ReasoningConstraints) -> dict:
    return {
        "exclude_items": list(c.exclude_items),
        "max_price": c.max_price,
        "min_aspect_score": dict(c.min_aspect_score),
    }


def build_emorec_graph(deps: GraphDeps) -> Any:
    """ABSA(cache) → Profiling → Reasoning → Reflection → (loop|Explanation) → END."""

    def absa_node(state: EmoRecState) -> EmoRecState:
        user_id = state["user_id"]
        triples = deps.load_triples(user_id)
        return {"triples": triples}

    def profiling_node(state: EmoRecState) -> EmoRecState:
        weights = deps.profiling.profile(
            state["user_id"], state["t_query_ms"], deps.top_k, persist=False
        )
        return {"weights": weights}

    def reasoning_node(state: EmoRecState) -> EmoRecState:
        pool_override = state.get("eval_candidates")
        result = deps.reasoning.recommend(
            state["user_id"],
            state.get("weights", {}),
            exclude=state.get("exclude_items", set()),
            k=deps.top_k,
            constraints=_constraints_from_state(state),
            price_lookup=state.get("item_prices"),
            use_llm_cot=deps.reasoning._llm is not None,
            pool_override=pool_override,
        )
        return {
            "recommendations": [r.item_id for r in result.recommendations],
            "breakdowns": result.breakdowns,
            "candidate_pool": result.candidate_pool,
            "rationale": result.rationale,
        }

    def reflection_node(state: EmoRecState) -> EmoRecState:
        recs = []
        breakdowns = state.get("breakdowns", {})
        for i, item_id in enumerate(state.get("recommendations", [])):
            from ..agents.reasoning_agent import Recommendation

            bd = breakdowns[item_id]
            recs.append(Recommendation(item_id=item_id, breakdown=bd, rank=i + 1))

        verdict = deps.reflection.evaluate(
            ReflectionInput(
                recommendations=recs,
                breakdowns=breakdowns,
                user_budget=state.get("user_budget"),
                item_prices=state.get("item_prices", {}),
                item_e_hat=state.get("item_e_hat", {}),
                recent_complaint_aspects=state.get("recent_complaint_aspects", []),
            )
        )
        iters = state.get("reflection_iters", 0) + (0 if verdict.approved else 1)
        out: EmoRecState = {
            "reflection": verdict,
            "approved": verdict.approved,
            "reflection_iters": iters,
        }
        if not verdict.approved:
            constraints = deps.reflection.constraints_from_verdict(verdict)
            out["constraints_json"] = _serialize_constraints(constraints)
        return out

    def explanation_node(state: EmoRecState) -> EmoRecState:
        recs = state.get("recommendations", [])
        if not recs:
            return {}
        top_item = recs[0]
        breakdown = state.get("breakdowns", {})[top_item]
        weights = state.get("weights", {})
        e_hat = state.get("item_e_hat", {}).get(top_item, {})
        support_map = state.get("aspect_support", {}).get(top_item, {})
        if not support_map and deps.aspect_support:
            support_map = deps.aspect_support(top_item)
        signals = deps.user_signals(state["user_id"], state["t_query_ms"])
        explanation = explain_recommendation(
            top_item,
            breakdown,
            weights,
            e_hat,
            support_map,
            signals,
            llm=deps.reasoning._llm,
            polish_with_llm=False,
        )
        return {"explanation": explanation}

    def route_after_reflection(state: EmoRecState) -> str:
        if state.get("approved"):
            return "explanation"
        if state.get("reflection_iters", 0) >= deps.max_reflection_iters:
            return "explanation"
        return "reasoning"

    timer = deps.stage_timer
    graph = StateGraph(EmoRecState)
    graph.add_node("absa", _timed("absa", timer, absa_node))
    graph.add_node("profiling", _timed("profiling", timer, profiling_node))
    graph.add_node("reasoning", _timed("reasoning", timer, reasoning_node))
    graph.add_node("reflection", _timed("reflection", timer, reflection_node))
    graph.add_node("explanation", _timed("explanation", timer, explanation_node))

    graph.add_edge(START, "absa")
    graph.add_edge("absa", "profiling")
    graph.add_edge("profiling", "reasoning")
    graph.add_edge("reasoning", "reflection")
    graph.add_conditional_edges(
        "reflection",
        route_after_reflection,
        {"reasoning": "reasoning", "explanation": "explanation"},
    )
    graph.add_edge("explanation", END)

    return graph.compile()
