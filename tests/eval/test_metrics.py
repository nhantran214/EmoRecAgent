"""Ranking-metric tests validated against hand-computed values."""

from __future__ import annotations

import math

import pytest

from emorecagent.eval import metrics as M


@pytest.fixture
def ranked() -> list[str]:
    # held-out relevant item "x" sits at rank 3
    return ["a", "b", "x", "c", "d"]


def test_hr_at_k_is_one_iff_relevant_in_topk(ranked: list[str]) -> None:
    assert M.hr_at_k(ranked, {"x"}, 2) == 0.0
    assert M.hr_at_k(ranked, {"x"}, 3) == 1.0
    assert M.hr_at_k(ranked, {"x"}, 5) == 1.0


def test_recall_single_relevant_equals_hit(ranked: list[str]) -> None:
    assert M.recall_at_k(ranked, {"x"}, 2) == 0.0
    assert M.recall_at_k(ranked, {"x"}, 3) == 1.0


def test_recall_multiple_relevant(ranked: list[str]) -> None:
    # "a"(1) and "x"(3) relevant; @2 retrieves only "a" -> 1/2
    assert M.recall_at_k(ranked, {"a", "x"}, 2) == 0.5
    assert M.recall_at_k(ranked, {"a", "x"}, 3) == 1.0


def test_ndcg_matches_hand_computation(ranked: list[str]) -> None:
    # single relevant at rank 3 -> DCG = 1/log2(4) = 0.5, IDCG = 1 -> 0.5
    assert M.ndcg_at_k(ranked, {"x"}, 3) == pytest.approx(0.5)
    assert M.ndcg_at_k(ranked, {"x"}, 2) == 0.0


def test_ndcg_two_relevant_uses_ideal_dcg(ranked: list[str]) -> None:
    # relevant "a"(rank1), "x"(rank3); DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
    # IDCG (2 hits) = 1/log2(2) + 1/log2(3) = 1 + 0.6309 = 1.6309
    expected = 1.5 / (1.0 + 1.0 / math.log2(3))
    assert M.ndcg_at_k(ranked, {"a", "x"}, 5) == pytest.approx(expected)


def test_mrr_is_reciprocal_of_first_hit_rank(ranked: list[str]) -> None:
    assert M.mrr_at_k(ranked, {"x"}, 5) == pytest.approx(1.0 / 3.0)
    assert M.mrr_at_k(ranked, {"x"}, 2) == 0.0  # beyond cutoff
    assert M.mrr_at_k(ranked, {"a", "x"}, 5) == pytest.approx(1.0)  # first hit rank 1


def test_evaluate_ranking_returns_all_metrics(ranked: list[str]) -> None:
    out = M.evaluate_ranking(ranked, {"x"}, 3)
    assert set(out) == {"recall", "ndcg", "hr", "mrr"}


def test_hr_equals_recall_single_relevant(ranked: list[str]) -> None:
    assert M.hr_at_k(ranked, {"x"}, 3) == M.recall_at_k(ranked, {"x"}, 3)


def test_empty_relevant_is_zero(ranked: list[str]) -> None:
    assert M.ndcg_at_k(ranked, set(), 5) == 0.0
    assert M.recall_at_k(ranked, set(), 5) == 0.0
