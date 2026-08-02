"""U9 Reflection Agent tests."""

from __future__ import annotations

from emorecagent.agents.reasoning_agent import Recommendation
from emorecagent.agents.reflection_agent import ReflectionAgent, ReflectionInput
from emorecagent.scoring.score import ScoreBreakdown


def _rec(item: str, total: float = 0.5) -> Recommendation:
    return Recommendation(
        item_id=item,
        breakdown=ScoreBreakdown(total=total, base_contribution=total),
        rank=1,
    )


def test_over_budget_item_rejected() -> None:
    agent = ReflectionAgent(llm=None, use_llm_judge=False)
    verdict = agent.evaluate(
        ReflectionInput(
            recommendations=[_rec("i_exp")],
            breakdowns={"i_exp": _rec("i_exp").breakdown},
            user_budget=20.0,
            item_prices={"i_exp": 49.99},
        )
    )
    assert not verdict.approved
    assert any("budget" in v for v in verdict.violated_constraints)


def test_null_price_uses_percentile_fallback() -> None:
    agent = ReflectionAgent(llm=None, use_llm_judge=False)
    verdict = agent.evaluate(
        ReflectionInput(
            recommendations=[_rec("i_null")],
            breakdowns={"i_null": _rec("i_null").breakdown},
            user_budget=30.0,
            item_prices={"i_null": None},
            category_price_percentile={"i_null": 0.90},
        )
    )
    assert not verdict.approved
    assert any("price_percentile" in v for v in verdict.violated_constraints)


def test_complaint_aspect_low_score_rejected() -> None:
    agent = ReflectionAgent(llm=None, use_llm_judge=False)
    verdict = agent.evaluate(
        ReflectionInput(
            recommendations=[_rec("i_bad")],
            breakdowns={"i_bad": _rec("i_bad").breakdown},
            item_e_hat={"i_bad": {"comfort": 0.2}},
            recent_complaint_aspects=["comfort"],
        )
    )
    assert not verdict.approved


def test_constraints_from_verdict_exclude_budget_item() -> None:
    agent = ReflectionAgent(llm=None, use_llm_judge=False)
    verdict = agent.evaluate(
        ReflectionInput(
            recommendations=[_rec("i_exp")],
            breakdowns={"i_exp": _rec("i_exp").breakdown},
            user_budget=10.0,
            item_prices={"i_exp": 99.0},
        )
    )
    c = agent.constraints_from_verdict(verdict)
    assert "i_exp" in c.exclude_items
