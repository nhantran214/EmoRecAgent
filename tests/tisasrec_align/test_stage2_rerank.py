"""Tests for Stage 2 pool boost, merge, and guardrail."""

from __future__ import annotations

from emorecagent.tisasrec_align.stage2_rerank import (
    apply_cross_user_boosts,
    build_pool,
    check_guardrail,
    merge_ranking,
)


def test_build_pool_caps_at_k() -> None:
    ranked = [f"i{n}" for n in range(10)]
    assert build_pool(ranked, 3) == ["i0", "i1", "i2"]


def test_apply_cross_user_boosts_reorders_with_evidence() -> None:
    pool = ["a", "b", "c"]
    scores = {"a": 1.0, "b": 0.9, "c": 0.8}
    boosts = {"c": 1.0}
    ordered = apply_cross_user_boosts(pool, scores, boosts, boost_scale=0.25)
    assert ordered[0] == "c"


def test_apply_cross_user_boosts_unchanged_without_boosts() -> None:
    pool = ["a", "b"]
    scores = {"a": 1.0, "b": 0.5}
    ordered = apply_cross_user_boosts(pool, scores, {}, boost_scale=0.1)
    assert ordered == ["a", "b"]


def test_merge_ranking_preserves_tail_order() -> None:
    stage1 = ["a", "b", "c", "d", "e"]
    merged = merge_ranking(["b", "a"], stage1, pool_k=2)
    assert merged[:2] == ["b", "a"]
    assert merged[2:] == ["c", "d", "e"]


def test_check_guardrail_fails_when_top_item_drops_too_far() -> None:
    stage1 = ["a", "b", "c", "d", "e", "f"]
    merged = ["b", "c", "d", "e", "f", "a"]
    assert check_guardrail(stage1, merged, top_n=1, max_drop_rank=5) is False


def test_check_guardrail_passes_small_reorder_within_head() -> None:
    stage1 = ["a", "b", "c", "d", "e"]
    merged = ["b", "a", "c", "d", "e"]
    assert check_guardrail(stage1, merged, top_n=5, max_drop_rank=10) is True
