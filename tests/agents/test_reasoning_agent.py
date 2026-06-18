"""U8 Reasoning Agent tests."""

from __future__ import annotations

from emorecagent.agents.reasoning_agent import (
    ReasoningAgent,
    ReasoningConstraints,
    build_candidate_pool,
)
from emorecagent.llm.client import FakeLLM, LLMClient
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


def test_fake_llm_cot_is_invoked() -> None:
    aspects = {"i1": {"comfort": 0.8}}
    cf = _CF({"i1": 0.6})
    llm = LLMClient(FakeLLM(["CoT rationale for comfort match."]))
    agent = ReasoningAgent(
        cf, _Aspects(aspects), _Strong({"comfort": ["i1"]}),
        llm=llm, alpha=0.5, pool_size=5,
    )
    result = agent.recommend("u1", {"comfort": 1.0}, exclude=set(), k=1)
    assert "CoT" in result.rationale


def test_score_item_matches_agent_breakdown() -> None:
    bd = score_item(0.5, 0.6, {"comfort": 1.0}, {"comfort": 0.8})
    assert bd.total == bd.base_contribution + sum(bd.aspect_contributions.values())
