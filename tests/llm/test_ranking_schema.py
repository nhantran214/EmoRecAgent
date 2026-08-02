"""Tests for listwise ranking schema coercion."""

from __future__ import annotations

from emorecagent.llm.prompts import REASONING_RANK_V1, format_prompt
from emorecagent.llm.schemas import ReasoningRankingVerdict, coerce_ranking_verdict


def test_coerce_ranking_happy_path() -> None:
    pool = ["i1", "i2", "i3"]
    out = coerce_ranking_verdict(
        ReasoningRankingVerdict(ranked_item_ids=["i3", "i1", "i2"]),
        pool_ids=pool,
    )
    assert out == ["i3", "i1", "i2"]


def test_coerce_ranking_dedupes_and_backfills() -> None:
    pool = ["i1", "i2", "i3"]
    out = coerce_ranking_verdict(
        {"ranked_item_ids": ["i2", "i2", "i9", "i1"]},
        pool_ids=pool,
    )
    assert out == ["i2", "i1", "i3"]


def test_coerce_ranking_empty_llm_list_backfills_pool() -> None:
    pool = ["a", "b"]
    out = coerce_ranking_verdict({"ranked_item_ids": []}, pool_ids=pool)
    assert out == pool


def test_reasoning_rank_prompt_renders() -> None:
    text = format_prompt(
        REASONING_RANK_V1,
        top_aspects="comfort:0.8, scent:0.2",
        candidate_cards="i1 | S=0.9 | comfort:0.7 | base=0.5",
    )
    assert "comfort:0.8" in text
    assert "i1 | S=0.9" in text
    assert "ReasoningRankingVerdict" in text
