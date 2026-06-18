"""U9 full LangGraph pipeline integration tests."""

from __future__ import annotations

from emorecagent.agents.profiling_agent import DynamicUserProfilingAgent
from emorecagent.agents.reasoning_agent import ReasoningAgent


class _CF:
    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def score(self, user_id: str, candidates: list[str]) -> dict[str, float]:
        return {c: self._scores.get(c, 0.0) for c in candidates}


class _Aspects:
    def __init__(self, data: dict[str, dict[str, float]]) -> None:
        self._data = data

    def get_item_aspects(self, item_id: str) -> dict[str, float]:
        return self._data.get(item_id, {})


class _Strong:
    def __init__(self, mapping: dict[str, list[str]]) -> None:
        self._mapping = mapping

    def items_strong_on(
        self, aspects: list[str], limit: int, exclude: set[str]
    ) -> list[str]:
        out: list[str] = []
        for a in aspects:
            for item in self._mapping.get(a, []):
                if item not in exclude and item not in out:
                    out.append(item)
                if len(out) >= limit:
                    return out
        return out
from emorecagent.agents.reflection_agent import ReflectionAgent
from emorecagent.graph.build import GraphDeps, build_emorec_graph
from emorecagent.llm.schemas import AbsaTriple
from emorecagent.scoring.dynamic_weights import AspectSignal

DAY = 86_400_000


class _SignalSource:
    def __init__(self, signals: list[AspectSignal]) -> None:
        self._signals = signals

    def get_user_aspect_signals(self, user_id: str) -> list[AspectSignal]:
        return self._signals

    def upsert_user_preferences(self, user_id, weights, updated_ts):  # noqa: ANN001
        pass


def _build_graph(*, max_iters: int = 2, budget: float | None = None):
    signals = [
        AspectSignal("comfort", -0.9, 5 * DAY),
        AspectSignal("scent", 0.4, 2 * DAY),
    ]
    profiling = DynamicUserProfilingAgent(_SignalSource(signals), lambda_per_day=0.01)

    aspects = {
        "i_ok": {"comfort": 0.85, "scent": 0.5},
        "i_bad": {"comfort": 0.2, "scent": 0.9},
    }
    cf = _CF({"i_ok": 0.7, "i_bad": 0.95})
    reasoning = ReasoningAgent(
        cf,
        _Aspects(aspects),
        _Strong({"comfort": ["i_ok", "i_bad"], "scent": ["i_bad"]}),
        llm=None,
        alpha=0.5,
        pool_size=10,
    )
    reflection = ReflectionAgent(llm=None, use_llm_judge=False)

    deps = GraphDeps(
        profiling=profiling,
        reasoning=reasoning,
        reflection=reflection,
        load_triples=lambda uid: [
            AbsaTriple(aspect="comfort", opinion="hurts", sentiment="negative")
        ],
        user_signals=lambda uid, ts: [s for s in signals if s.timestamp_ms < ts],
        aspect_support=lambda item: {"comfort": 50, "scent": 10},
        max_reflection_iters=max_iters,
        top_k=2,
    )
    return build_emorec_graph(deps), budget


def test_pipeline_approved_flow_reaches_explanation() -> None:
    graph, _ = _build_graph()
    out = graph.invoke(
        {
            "user_id": "u1",
            "t_query_ms": 10 * DAY,
            "exclude_items": set(),
            "recent_complaint_aspects": [],
            "item_e_hat": {
                "i_ok": {"comfort": 0.85},
                "i_bad": {"comfort": 0.85},
            },
            "item_prices": {},
            "aspect_support": {"i_ok": {"comfort": 50}},
        },
        config={"recursion_limit": 20},
    )
    assert out.get("approved") is True
    assert out.get("explanation") is not None
    assert out["explanation"].item_id in out.get("recommendations", [])


def test_over_budget_triggers_retry_then_terminates() -> None:
    graph, _ = _build_graph(max_iters=2)
    out = graph.invoke(
        {
            "user_id": "u1",
            "t_query_ms": 10 * DAY,
            "exclude_items": set(),
            "user_budget": 15.0,
            "item_prices": {"i_ok": 10.0, "i_bad": 50.0},
            "item_e_hat": {
                "i_ok": {"comfort": 0.9},
                "i_bad": {"comfort": 0.9},
            },
            "recent_complaint_aspects": [],
        },
        config={"recursion_limit": 20},
    )
    assert out.get("reflection_iters", 0) >= 1
    assert out.get("explanation") is not None
    # After retry, expensive item should be excluded from top recs when possible
    if out.get("recommendations"):
        assert "i_bad" not in out["recommendations"][:1] or not out.get("approved")


def test_max_reflection_iters_caps_loop() -> None:
    graph, _ = _build_graph(max_iters=1)
    out = graph.invoke(
        {
            "user_id": "u1",
            "t_query_ms": 10 * DAY,
            "exclude_items": set(),
            "user_budget": 5.0,
            "item_prices": {"i_ok": 100.0, "i_bad": 100.0},
            "item_e_hat": {"i_ok": {"comfort": 0.9}, "i_bad": {"comfort": 0.9}},
            "recent_complaint_aspects": [],
        },
        config={"recursion_limit": 20},
    )
    assert out.get("reflection_iters", 0) <= 1
    assert "explanation" in out
