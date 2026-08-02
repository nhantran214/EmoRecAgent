"""U8 Reasoning Agent tests."""

from __future__ import annotations

from emorecagent.agents.reasoning_agent import (
    ReasoningAgent,
    ReasoningConstraints,
    build_candidate_pool,
    expand_pool_with_sar,
)
from emorecagent.llm.client import FakeLLM, LLMClient
from emorecagent.llm.schemas import ReasoningRankingVerdict
from emorecagent.scoring.score import score_item


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


def test_candidate_pool_includes_cf_and_aspect_strong() -> None:
    cf = _CF({"i_cf": 0.9, "i_weak": 0.1})
    strong = _Strong({"comfort": ["i_comfy", "i_cf"]})
    pool = build_candidate_pool(
        cf, strong, "u1", {"comfort": 0.7, "scent": 0.3},
        pool_size=5, exclude=set(),
    )
    assert "i_comfy" in pool
    assert "i_cf" in pool


def test_ranking_respects_s_with_higher_weight_on_strong_aspect() -> None:
    weights = {"comfort": 0.9, "scent": 0.1}
    aspects = {
        "i_comfy": {"comfort": 0.95, "scent": 0.2},
        "i_scent": {"comfort": 0.2, "scent": 0.95},
    }
    cf = _CF({"i_comfy": 0.5, "i_scent": 0.5})
    agent = ReasoningAgent(
        cf, _Aspects(aspects), _Strong({"comfort": ["i_comfy"], "scent": ["i_scent"]}),
        llm=None, alpha=0.5, pool_size=10,
    )
    result = agent.recommend("u1", weights, exclude=set(), k=2)
    assert result.recommendations[0].item_id == "i_comfy"


def test_critique_constraint_excludes_over_budget_item() -> None:
    aspects = {"i_exp": {"comfort": 0.9}, "i_ok": {"comfort": 0.85}}
    cf = _CF({"i_exp": 0.99, "i_ok": 0.5})
    agent = ReasoningAgent(
        cf, _Aspects(aspects), _Strong({"comfort": ["i_exp", "i_ok"]}),
        llm=None, alpha=0.3, pool_size=10,
    )
    constraints = ReasoningConstraints(exclude_items={"i_exp"})
    result = agent.recommend(
        "u1", {"comfort": 1.0}, exclude=set(), k=2, constraints=constraints
    )
    ids = [r.item_id for r in result.recommendations]
    assert "i_exp" not in ids
    assert "i_ok" in ids


def test_sar_expansion_adds_aspect_strong_item() -> None:
    strong = _Strong({"comfort": ["i_sar"]})
    expanded = expand_pool_with_sar(
        ["i_hgt"],
        strong,
        {"comfort": 0.9},
        exclude=set(),
        pool_size=5,
        aspect_recall_max=5,
    )
    assert expanded[0] == "i_hgt"
    assert "i_sar" in expanded


def test_llm_rerank_sets_ranked_pool_order() -> None:
    aspects = {
        "i1": {"comfort": 0.8},
        "i2": {"comfort": 0.7},
    }
    cf = _CF({"i1": 0.9, "i2": 0.5})
    llm = LLMClient(
        FakeLLM([ReasoningRankingVerdict(ranked_item_ids=["i2", "i1"])])
    )
    agent = ReasoningAgent(
        cf,
        _Aspects(aspects),
        _Strong({"comfort": ["i1", "i2"]}),
        llm=llm,
        alpha=0.5,
        pool_size=5,
    )
    result = agent.recommend("u1", {"comfort": 1.0}, exclude=set(), k=2)
    assert result.ranked_pool_order[:2] == ["i2", "i1"]
    assert result.recommendations[0].item_id == "i2"


def test_numeric_order_when_llm_disabled() -> None:
    aspects = {
        "i1": {"comfort": 0.8},
        "i2": {"comfort": 0.95},
    }
    cf = _CF({"i1": 0.4, "i2": 0.5})
    agent = ReasoningAgent(
        cf,
        _Aspects(aspects),
        _Strong({"comfort": ["i1", "i2"]}),
        llm=None,
        alpha=0.3,
        pool_size=5,
    )
    result = agent.recommend(
        "u1", {"comfort": 1.0}, exclude=set(), k=2, use_llm_cot=False
    )
    assert result.ranked_pool_order[0] == "i2"


def test_llm_failure_falls_back_to_numeric_order() -> None:
    aspects = {"i1": {"comfort": 0.8}, "i2": {"comfort": 0.95}}
    cf = _CF({"i1": 0.4, "i2": 0.5})
    llm = LLMClient(FakeLLM([], raise_on=0))
    agent = ReasoningAgent(
        cf,
        _Aspects(aspects),
        _Strong({"comfort": ["i1", "i2"]}),
        llm=llm,
        alpha=0.3,
        pool_size=5,
    )
    result = agent.recommend("u1", {"comfort": 1.0}, exclude=set(), k=2)
    assert result.ranked_pool_order[0] == "i2"
    assert result.recommendations


def test_llm_rerank_pool_batch_three_rows() -> None:
    aspects = {
        "i1": {"comfort": 0.8},
        "i2": {"comfort": 0.7},
        "i3": {"comfort": 0.6},
    }
    cf = _CF({"i1": 0.9, "i2": 0.5, "i3": 0.4})
    from emorecagent.llm.schemas import BatchReasoningRankingVerdict, BatchReasoningRow

    batch_verdict = (
        '{"rows":[{"row_id":"r1","ranked_item_ids":["i2","i1"]},'
        '{"row_id":"r2","ranked_item_ids":["i1","i2"]}]}'
    )
    llm = LLMClient(FakeLLM([batch_verdict]))
    agent = ReasoningAgent(
        cf,
        _Aspects(aspects),
        _Strong({"comfort": ["i1", "i2", "i3"]}),
        llm=llm,
        alpha=0.5,
        pool_size=5,
    )
    from emorecagent.agents.reasoning_agent import BatchRowContext
    from emorecagent.scoring.score import score_item

    bd1 = score_item(0.5, 0.9, {"comfort": 1.0}, aspects["i1"])
    bd2 = score_item(0.5, 0.5, {"comfort": 1.0}, aspects["i2"])
    out = agent.llm_rerank_pool_batch(
        [
            BatchRowContext(
                row_id="r1",
                user_id="u1",
                weights={"comfort": 1.0},
                pool=["i1", "i2"],
                breakdowns={"i1": bd1, "i2": bd2},
                numeric_order=["i1", "i2"],
            ),
            BatchRowContext(
                row_id="r2",
                user_id="u2",
                weights={"comfort": 1.0},
                pool=["i1", "i2"],
                breakdowns={"i1": bd1, "i2": bd2},
                numeric_order=["i2", "i1"],
            ),
        ]
    )
    assert out["r1"] == ["i2", "i1"]
    assert out["r2"] == ["i1", "i2"]


def test_score_item_matches_agent_breakdown() -> None:
    bd = score_item(0.5, 0.6, {"comfort": 1.0}, {"comfort": 0.8})
    assert bd.total == bd.base_contribution + sum(bd.aspect_contributions.values())
