"""Rationalized explanation tests."""

from __future__ import annotations

from emorecagent.explain.rationalize import explain_recommendation
from emorecagent.scoring.dynamic_weights import AspectSignal
from emorecagent.scoring.score import ScoreBreakdown

DAY = 86_400_000


def _breakdown() -> ScoreBreakdown:
    return ScoreBreakdown(
        total=0.65,
        base_contribution=0.20,
        aspect_contributions={"comfort": 0.40, "scent": 0.05},
    )


def test_cites_only_breakdown_aspects() -> None:
    exp = explain_recommendation(
        "i1",
        _breakdown(),
        weights={"comfort": 0.8, "scent": 0.2},
        item_e_hat={"comfort": 0.9, "scent": 0.6, "price": 0.5},
        aspect_support={"comfort": 95, "scent": 12},
        user_signals=[],
    )
    assert set(exp.cited_aspects) <= {"comfort", "scent"}
    assert "price" not in exp.cited_aspects


def test_polarity_agrees_with_e_hat() -> None:
    exp = explain_recommendation(
        "i1",
        _breakdown(),
        weights={"comfort": 0.8},
        item_e_hat={"comfort": 0.9},
        aspect_support={"comfort": 10},
        user_signals=[],
    )
    assert any("positively" in c for c in exp.claims)


def test_support_counts_in_claims() -> None:
    exp = explain_recommendation(
        "i1",
        _breakdown(),
        weights={"comfort": 0.8},
        item_e_hat={"comfort": 0.9},
        aspect_support={"comfort": 42},
        user_signals=[],
    )
    assert any("42" in c for c in exp.claims)


def test_numeric_spec_omitted_when_null() -> None:
    exp = explain_recommendation(
        "i1",
        _breakdown(),
        weights={"comfort": 0.8},
        item_e_hat={"comfort": 0.9},
        aspect_support={"comfort": 5},
        user_signals=[],
        numeric_specs={"weight": None, "price": "19.99"},
    )
    assert any("price" in c for c in exp.claims)
    assert not any("weight" in c for c in exp.claims)


def test_contribution_magnitude_round_trips() -> None:
    exp = explain_recommendation(
        "i1",
        _breakdown(),
        weights={"comfort": 0.8},
        item_e_hat={"comfort": 0.9},
        aspect_support={"comfort": 5},
        user_signals=[],
    )
    assert exp.aspects[0].contribution == 0.40


def test_user_prior_sentiment_referenced() -> None:
    signals = [AspectSignal("comfort", -0.8, 5 * DAY)]
    exp = explain_recommendation(
        "i1",
        _breakdown(),
        weights={"comfort": 0.8},
        item_e_hat={"comfort": 0.9},
        aspect_support={"comfort": 5},
        user_signals=signals,
    )
    assert any("negative" in c or "comfort" in c for c in exp.claims)
