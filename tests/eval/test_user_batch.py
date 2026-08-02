"""Tests for user-batch (paper-aligned) evaluation protocol."""

from __future__ import annotations

from emorecagent.baselines.popularity import PopularityRecommender
from emorecagent.data.types import Interaction
from emorecagent.eval.runner import evaluate

DAY = 86_400_000


def _train() -> list[Interaction]:
    return [
        Interaction("u1", "i_a", 5.0, 1 * DAY),
        Interaction("u1", "i_b", 5.0, 2 * DAY),
        Interaction("u2", "i_a", 5.0, 1 * DAY),
        Interaction("u2", "i_c", 5.0, 2 * DAY),
    ]


def test_user_batch_one_rank_per_user() -> None:
    test = [
        Interaction("u1", "i_x", 5.0, 3 * DAY, verified_purchase=True),
        Interaction("u1", "i_y", 5.0, 4 * DAY, verified_purchase=False),
        Interaction("u2", "i_z", 5.0, 3 * DAY, verified_purchase=True),
    ]
    res = evaluate(
        PopularityRecommender(),
        _train(),
        test,
        k_values=[5],
        method="pop",
        eval_protocol="user_batch",
        aggregation="user_mean",
        verified_only=False,
    )
    assert res.eval_protocol == "user_batch"
    assert res.aggregation == "user_mean"
    assert res.n_test_users == 2
    assert res.n_test_rows == 3
    assert len(res.user_ids) == 2
    assert len(res.per_user["recall@5"]) == 2
    assert res.means == res.means_per_user


def test_user_batch_multi_relevant_recall() -> None:
    """User with two test items: recall@3 counts both in ground truth."""
    train = [Interaction("u1", "i_train", 5.0, 1 * DAY)]
    test = [
        Interaction("u1", "i_hit", 5.0, 2 * DAY),
        Interaction("u1", "i_miss", 5.0, 3 * DAY),
    ]

    class _FixedRank:
        name = "fixed"

        def fit(self, interactions):
            return self

        def rank(self, user_id, candidates):
            return ["i_hit", "i_other", "i_tail", "i_miss", "i_train"]

    res = evaluate(
        _FixedRank(),
        train,
        test,
        k_values=[3],
        method="fixed",
        eval_protocol="user_batch",
        aggregation="user_mean",
    )
    assert res.means["recall@3"] == 0.5


def test_user_batch_verified_only_excludes_unverified_items() -> None:
    test = [
        Interaction("u1", "i_v", 5.0, 3 * DAY, verified_purchase=True),
        Interaction("u1", "i_u", 5.0, 4 * DAY, verified_purchase=False),
    ]
    res = evaluate(
        PopularityRecommender(),
        _train(),
        test,
        k_values=[5],
        method="pop",
        eval_protocol="user_batch",
        verified_only=True,
    )
    assert res.n_test_users == 1
    assert res.n_test_rows == 1


def test_per_row_protocol_unchanged() -> None:
    test = [Interaction("u1", "i_x", 5.0, 3 * DAY)]
    res = evaluate(
        PopularityRecommender(),
        _train(),
        test,
        k_values=[5],
        method="pop",
        eval_protocol="per_row",
        aggregation="row_mean",
        sampled_n_negatives=None,
    )
    assert res.eval_protocol == "per_row"
    assert res.aggregation == "row_mean"
    assert res.n_test_rows == 1


def test_user_batch_sampled_negatives() -> None:
    test = [
        Interaction("u1", "i_x", 5.0, 3 * DAY),
        Interaction("u1", "i_y", 5.0, 4 * DAY),
    ]
    res = evaluate(
        PopularityRecommender(),
        _train(),
        test,
        k_values=[5, 10],
        method="pop",
        eval_protocol="user_batch",
        aggregation="user_mean",
        n_negatives=2,
    )
    assert res.protocol == "sampled_negatives"
    assert res.n_negatives == 2
    assert len(res.per_user["hr@10"]) == 2
    assert "hr@10" in res.means


def test_dual_eval_includes_sampled_block() -> None:
    test = [Interaction("u0", "i_held", 5.0, 3 * DAY)]
    res = evaluate(
        PopularityRecommender(),
        _train(),
        test,
        k_values=[10],
        method="pop",
        sampled_n_negatives=2,
        sampled_k_values=[1, 3, 5],
    )
    assert res.protocol == "full_catalog"
    assert res.sampled is not None
    assert res.sampled["protocol"] == "sampled_negatives"
    assert res.sampled["n_negatives"] == 2
    assert res.sampled["k_values"] == [1, 3, 5, 10]
    for k in (1, 3, 5, 10):
        for m in ("hr", "mrr", "ndcg", "recall"):
            assert f"{m}@{k}" in res.sampled["means"]
    # full-catalog pass: no recall/mrr/ndcg @1,3,5 (hr@1,3,5 still from hr_avg_k)
    assert "recall@1" not in res.means
    assert "mrr@3" not in res.means
    assert "ndcg@5" not in res.means
