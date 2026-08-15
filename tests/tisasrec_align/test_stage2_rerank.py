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


def test_blend_rank_orders_beta_one_follows_llm() -> None:
    from emorecagent.tisasrec_align.stage2_rerank import blend_rank_orders

    s1 = ["a", "b", "c", "d"]
    llm = ["c", "a", "b", "d"]
    assert blend_rank_orders(s1, llm, beta=1.0) == ["c", "a", "b", "d"]
    assert blend_rank_orders(s1, llm, beta=0.0) == s1
    blended = blend_rank_orders(s1, llm, beta=0.5)
    assert set(blended) == set(s1)
    assert blended[0] in {"a", "c"}


def test_promote_then_fill() -> None:
    from emorecagent.tisasrec_align.stage2_rerank import promote_then_fill

    pool = ["a", "b", "c", "d"]
    llm = ["c", "d", "x"]
    assert promote_then_fill(pool, llm, promote_k=2) == ["c", "d", "a", "b"]


def test_promote_preserving_head_freezes_top_n() -> None:
    from emorecagent.tisasrec_align.stage2_rerank import promote_preserving_head

    pool = ["a", "b", "c", "d", "e", "f"]
    llm = ["f", "e", "a"]  # a is protected — ignored; f,e insert after head
    out = promote_preserving_head(pool, llm, promote_k=2, protect_n=2)
    assert out[:2] == ["a", "b"]
    assert out[2:4] == ["f", "e"]
    assert set(out) == set(pool)


def test_promote_swap_only_tail_slot() -> None:
    from emorecagent.tisasrec_align.stage2_rerank import promote_swap

    pool = [f"i{n}" for n in range(1, 13)]  # i1..i12
    # protect 9 → head window 10; swap i10 with i12
    out = promote_swap(pool, ["i12", "i11"], promote_k=1, protect_n=9)
    assert out[:9] == pool[:9]
    assert out[9] == "i12"
    assert out[10] == "i10"  # displaced
    assert "i11" in out
    assert set(out) == set(pool)


def test_promote_swap_no_op_when_picks_inside_head() -> None:
    from emorecagent.tisasrec_align.stage2_rerank import promote_swap

    pool = [f"i{n}" for n in range(1, 13)]
    out = promote_swap(pool, ["i3", "i10"], promote_k=1, protect_n=9)
    assert out == pool


def test_check_guardrail_passes_small_reorder_within_head() -> None:
    stage1 = ["a", "b", "c", "d", "e"]
    merged = ["b", "a", "c", "d", "e"]
    assert check_guardrail(stage1, merged, top_n=5, max_drop_rank=10) is True
