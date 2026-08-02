"""Batch ranking schema coercion tests."""

from __future__ import annotations

from emorecagent.llm.schemas import (
    BatchReasoningRankingVerdict,
    BatchReasoningRow,
    coerce_batch_ranking_verdict,
    parse_ranking_json,
    ranking_max_tokens,
)


def test_coerce_batch_ranking_verdict_happy_path() -> None:
    pools = {
        "r1": ["a", "b", "c"],
        "r2": ["x", "y"],
    }
    verdict = BatchReasoningRankingVerdict(
        rows=[
            BatchReasoningRow(row_id="r1", ranked_item_ids=["c", "a", "b"]),
            BatchReasoningRow(row_id="r2", ranked_item_ids=["y", "x"]),
        ]
    )
    out = coerce_batch_ranking_verdict(verdict, pools_by_row=pools)
    assert out["r1"] == ["c", "a", "b"]
    assert out["r2"] == ["y", "x"]


def test_coerce_batch_missing_row_omitted() -> None:
    pools = {"r1": ["a", "b"]}
    verdict = BatchReasoningRankingVerdict(
        rows=[BatchReasoningRow(row_id="r2", ranked_item_ids=["a", "b"])]
    )
    out = coerce_batch_ranking_verdict(verdict, pools_by_row=pools)
    assert out == {}


def test_parse_ranking_json_salvages_truncated_output() -> None:
    pool = ["B0001", "B0002", "B0003", "B0004"]
    partial = '{"ranked_item_ids":["B0003","B0001","B0002"'
    ranked = parse_ranking_json(partial, pool_ids=pool)
    assert ranked[:3] == ["B0003", "B0001", "B0002"]
    assert ranked[3] == "B0004"


def test_ranking_max_tokens_scales_with_pool() -> None:
    assert ranking_max_tokens([50]) >= 2048
    assert ranking_max_tokens([50, 50], cap=4096) <= 4096


def test_coerce_batch_repairs_duplicate_ids() -> None:
    pools = {"r1": ["a", "b", "c"]}
    verdict = BatchReasoningRankingVerdict(
        rows=[BatchReasoningRow(row_id="r1", ranked_item_ids=["a", "a", "b"])]
    )
    out = coerce_batch_ranking_verdict(verdict, pools_by_row=pools)
    assert set(out["r1"]) == {"a", "b", "c"}
    assert len(out["r1"]) == 3
