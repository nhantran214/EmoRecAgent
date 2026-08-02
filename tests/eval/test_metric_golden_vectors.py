"""Golden-vector tests for standard ranking metrics."""

from __future__ import annotations

import pytest

from emorecagent.eval import metrics as M


@pytest.fixture
def ranked() -> list[str]:
    return ["a", "b", "x", "c", "d"]


def test_k_zero_raises() -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        M.evaluate_ranking(["a"], {"a"}, 0)


def test_ndcg_relevant_at_rank_one_is_one(ranked: list[str]) -> None:
    assert M.ndcg_at_k(["x", "a", "b"], {"x"}, 5) == pytest.approx(1.0)


def test_avg_hr_relevant_at_rank_four() -> None:
    ranked = ["a", "b", "c", "x", "d"]
    # x at rank 4: HR@1=0, HR@3=0, HR@5=1 -> avg = 1/3
    assert M.avg_hr_at_k_list(ranked, {"x"}, (1, 3, 5)) == pytest.approx(1 / 3)


def test_avg_hr_relevant_at_rank_one(ranked: list[str]) -> None:
    assert M.avg_hr_at_k_list(["x", "a"], {"x"}, (1, 3, 5)) == pytest.approx(1.0)


def test_avg_hr_short_ranked_list() -> None:
    assert M.avg_hr_at_k_list(["a"], {"a"}, (1, 3, 5)) == pytest.approx(1.0)


def test_evaluate_hr_avg_keys() -> None:
    ranked = ["a", "b", "c", "x", "d"]
    out = M.evaluate_hr_avg(ranked, {"x"})
    assert out[M.AVG_HR_KEY] == pytest.approx(1 / 3)
    assert out["hr@1"] == 0.0
    assert out["hr@5"] == 1.0
