"""Tests for Stage 2 LLM rerank helper."""

from __future__ import annotations

from unittest.mock import MagicMock

from emorecagent.llm.client import LLMError
from emorecagent.tisasrec_align.stage2_llm_rerank import (
    build_lookup_hints,
    llm_rerank_pool,
)


def test_build_lookup_hints_only_pool_items() -> None:
    lookup = {"A": {"X": 5, "Z": 1}}
    hints = build_lookup_hints(["A"], {"X"}, lookup)
    assert "X" in hints
    assert "Z" not in hints


def test_llm_rerank_pool_uses_mock_llm() -> None:
    llm = MagicMock()
    llm.invoke_ranking_json.return_value = ["b", "a"]
    pool = ["a", "b"]
    scores = {"a": 1.0, "b": 0.5}
    out = llm_rerank_pool(
        llm,
        t_u="likes skincare",
        reviewed_items=["A"],
        lookup={},
        pool=pool,
        scores=scores,
        numeric_fallback=["a", "b"],
    )
    assert out == ["b", "a"]


def test_llm_rerank_pool_unknown_ids_coerced_via_client() -> None:
    llm = MagicMock()
    llm.invoke_ranking_json.return_value = ["b", "a"]
    out = llm_rerank_pool(
        llm,
        t_u="t",
        reviewed_items=[],
        lookup={},
        pool=["a", "b", "c"],
        scores={"a": 1.0, "b": 0.5, "c": 0.1},
        numeric_fallback=["a", "b", "c"],
    )
    assert out == ["b", "a"]


def test_llm_rerank_pool_none_returns_numeric_fallback() -> None:
    out = llm_rerank_pool(
        None,
        t_u="t",
        reviewed_items=[],
        lookup={},
        pool=["a", "b"],
        scores={"a": 1.0, "b": 0.2},
        numeric_fallback=["a", "b"],
    )
    assert out == ["a", "b"]


def test_llm_rerank_pool_llm_error_falls_back() -> None:
    llm = MagicMock()
    llm.invoke_ranking_json.side_effect = LLMError("fail")
    out = llm_rerank_pool(
        llm,
        t_u="t",
        reviewed_items=[],
        lookup={},
        pool=["a", "b"],
        scores={"a": 1.0, "b": 0.2},
        numeric_fallback=["a", "b"],
    )
    assert out == ["a", "b"]
