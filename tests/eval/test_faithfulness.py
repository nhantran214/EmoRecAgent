"""Faithfulness tests: ERASER-style perturbation and unfaithful control."""

from __future__ import annotations

import pytest

from emorecagent.eval.faithfulness import evaluate_explanation
from emorecagent.scoring.score import ScoreBreakdown


def _breakdowns() -> dict[str, ScoreBreakdown]:
    # i1 is top, driven mostly by "comfort"; i2 is a close competitor.
    i1 = ScoreBreakdown(
        total=0.71,
        base_contribution=0.20,
        aspect_contributions={"comfort": 0.50, "scent": 0.01},
    )
    i2 = ScoreBreakdown(total=0.40, base_contribution=0.40, aspect_contributions={})
    return {"i1": i1, "i2": i2}


def test_faithful_explanation_drops_rank_when_driver_removed() -> None:
    res = evaluate_explanation(_breakdowns(), "i1", {"comfort"})
    assert res.rank_before == 1
    assert res.rank_after == 2          # i1 falls behind i2
    assert res.rank_drop == 1
    assert res.faithful
    assert res.comprehensiveness == pytest.approx(0.50)


def test_unfaithful_control_does_not_change_rank() -> None:
    # Citing "scent" (a non-driver) barely changes the score: not faithful.
    res = evaluate_explanation(_breakdowns(), "i1", {"scent"})
    assert res.rank_before == 1
    assert res.rank_after == 1
    assert res.rank_drop == 0
    assert not res.faithful
    assert res.comprehensiveness == pytest.approx(0.01)


def test_sufficiency_gap_small_for_true_drivers() -> None:
    res = evaluate_explanation(_breakdowns(), "i1", {"comfort"})
    # base(0.20) + comfort(0.50) = 0.70 vs total 0.71 -> gap 0.01
    assert res.sufficiency_gap == pytest.approx(0.01)


def test_citing_all_aspects_is_maximally_comprehensive() -> None:
    res = evaluate_explanation(_breakdowns(), "i1", {"comfort", "scent"})
    assert res.comprehensiveness == pytest.approx(0.51)
