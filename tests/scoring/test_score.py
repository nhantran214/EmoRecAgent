"""Tests for U7 scoring: CF base, sentiment aggregation, and S(u,i) (R5)."""

from __future__ import annotations

import math

from emorecagent.data.types import Interaction
from emorecagent.scoring.cf_base import CFBase
from emorecagent.scoring.score import rank_items, score_item
from emorecagent.scoring.sentiment_agg import (
    ItemAspectTriple,
    aggregate_rescaled,
    rescale,
)


# --- S(u,i) blend ---

def test_alpha_one_reduces_to_base():
    bd = score_item(alpha=1.0, s_base=0.7, weights={"a": 1.0}, item_aspects={"a": 0.9})
    assert math.isclose(bd.total, 0.7)
    assert bd.affective_total == 0.0


def test_alpha_zero_reduces_to_affective():
    bd = score_item(alpha=0.0, s_base=0.7, weights={"a": 1.0}, item_aspects={"a": 0.9})
    assert math.isclose(bd.total, 0.9)
    assert math.isclose(bd.base_contribution, 0.0)


def test_no_aspect_overlap_falls_back_to_alpha_base():
    bd = score_item(alpha=0.5, s_base=0.8, weights={"comfort": 1.0}, item_aspects={"scent": 0.9})
    assert math.isclose(bd.total, 0.5 * 0.8)
    assert bd.aspect_contributions == {}


def test_breakdown_components_sum_to_total():
    bd = score_item(
        alpha=0.5,
        s_base=0.6,
        weights={"a": 0.5, "b": 0.5},
        item_aspects={"a": 1.0, "b": 0.0},
    )
    assert math.isclose(bd.total, bd.base_contribution + bd.affective_total)


def test_higher_weight_on_strong_aspect_raises_rank():
    s_base = {"i1": 0.5, "i2": 0.5}
    # i1 is strong on comfort, i2 strong on scent
    item_aspects = {"i1": {"comfort": 1.0}, "i2": {"scent": 1.0}}
    # user who cares about comfort should rank i1 first
    ranked = rank_items(0.3, s_base, {"comfort": 1.0}, item_aspects)
    assert ranked[0][0] == "i1"


# --- sentiment aggregation ---

def test_rescale_bounds():
    assert rescale(-1.0) == 0.0
    assert rescale(1.0) == 1.0
    assert math.isclose(rescale(0.0), 0.5)


def test_aggregate_helpfulness_capped_weighting():
    # a very-high-helpful positive triple should not fully dominate due to cap
    triples = [
        ItemAspectTriple("comfort", polarity=1.0, helpful_vote=1000),
        ItemAspectTriple("comfort", polarity=-1.0, helpful_vote=0),
    ]
    out = aggregate_rescaled(triples, helpful_cap=10)
    # capped weights: (11*1 + 1*-1)/(11+1) = 10/12 ~ 0.833 raw -> rescaled ~0.917
    assert 0.85 < out["comfort"] < 0.95


# --- CF base ---

def _mk(u, i):
    return Interaction(u, i, 5.0, 1, 0)


def test_cf_base_normalizes_and_ranks():
    # u1 co-occurs with u2 on i1,i2; i3 only seen by u3 -> u1 should prefer i1/i2
    inter = [
        _mk("u1", "i1"), _mk("u1", "i2"),
        _mk("u2", "i1"), _mk("u2", "i2"), _mk("u2", "i3"),
        _mk("u3", "i3"),
    ]
    cf = CFBase(backend="itemknn").fit(inter)
    scores = cf.score("u1", ["i1", "i2", "i3"])
    assert set(scores) == {"i1", "i2", "i3"}
    assert all(0.0 <= v <= 1.0 for v in scores.values())
    # i3 (only via u2's weak link) should not outrank the directly-shared items
    assert scores["i3"] <= max(scores["i1"], scores["i2"])


def test_cf_unknown_user_returns_zeros():
    cf = CFBase(backend="svd", factors=2).fit([_mk("u1", "i1"), _mk("u1", "i2"), _mk("u2", "i1")])
    scores = cf.score("ghost", ["i1", "i2"])
    assert scores == {"i1": 0.0, "i2": 0.0}
