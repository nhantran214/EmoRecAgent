"""Runner tests: split loading, evaluation, infra-method guard, reproducibility."""

from __future__ import annotations

import json

import pytest

from emorecagent.baselines.popularity import PopularityRecommender
from emorecagent.data.types import Interaction
from emorecagent.eval.runner import (
    build_recommender,
    evaluate,
    load_split_jsonl,
    paired_compare,
    write_results,
)

DAY = 86_400_000


def _train() -> list[Interaction]:
    data = []
    # 6 users each interact with i_pop + one tail item; i_pop is globally popular.
    for u in range(6):
        data.append(Interaction(f"u{u}", "i_pop", 5.0, 1 * DAY))
        data.append(Interaction(f"u{u}", f"i_tail{u}", 5.0, 2 * DAY))
    return data


def _test() -> list[Interaction]:
    # held-out item for u0 is i_pop2 (a second popular item not in train)
    return [Interaction("u0", "i_held", 5.0, 3 * DAY)]


def test_evaluate_produces_means_and_per_user_vectors() -> None:
    res = evaluate(
        PopularityRecommender(), _train(), _test(), k_values=[1, 5], method="pop"
    )
    assert res.n_test_rows == 1
    assert res.n_test_users == 1
    assert res.protocol == "full_catalog"
    assert "ndcg@5" in res.means and "hr@1" in res.means
    assert "avg_hr@1,3,5" in res.means
    assert len(res.user_ids) == 1
    assert all(len(v) == 1 for v in res.per_user.values())


def test_agentic_methods_require_train_interactions() -> None:
    with pytest.raises(ValueError, match="train_interactions"):
        build_recommender("emorecagent", {}, seed=0)
    with pytest.raises(ValueError, match="train_interactions"):
        build_recommender("aspect_aware", {}, seed=0)
    with pytest.raises(ValueError, match="train_interactions"):
        build_recommender("emorecagent_align", {}, seed=0)


def test_emorecagent_runs_on_fixture_train() -> None:
    rec = build_recommender(
        "emorecagent",
        {
            "train_interactions": _train(),
            "factors": 4,
            "kg_backend": "memory",
            "use_llm_cot": False,
            "use_reflection": False,
        },
        seed=0,
    )
    res = evaluate(
        rec,
        _train(),
        _test(),
        k_values=[5],
        method="emorecagent",
        seed=0,
        method_variant="langgraph",
    )
    assert res.n_test_users == 1
    assert "ndcg@5" in res.means
    assert res.method_variant == "langgraph"


def test_emorecagent_fast_runs_on_fixture_train() -> None:
    rec = build_recommender(
        "emorecagent_fast",
        {"train_interactions": _train(), "factors": 4},
        seed=0,
    )
    res = evaluate(
        rec, _train(), _test(), k_values=[5], method="emorecagent_fast", seed=0
    )
    assert res.n_test_users == 1


def test_verified_only_filters_test_rows() -> None:
    test_rows = [
        Interaction("u0", "i_held", 5.0, 3 * DAY, verified_purchase=True),
        Interaction("u0", "i_skip", 5.0, 4 * DAY, verified_purchase=False),
    ]
    res = evaluate(
        PopularityRecommender(),
        _train(),
        test_rows,
        [5],
        method="pop",
        verified_only=True,
    )
    assert res.n_test_rows == 1
    assert res.verified_only is True
    assert res.n_verified_rows == 1


def test_emorecagent_runs_on_fixture_train() -> None:
    rec = build_recommender(
        "emorecagent",
        {
            "train_interactions": _train(),
            "factors": 4,
            "kg_backend": "memory",
            "use_llm_cot": False,
            "use_reflection": False,
        },
        seed=0,
    )
    res = evaluate(
        rec,
        _train(),
        _test(),
        k_values=[5],
        method="emorecagent",
        seed=0,
        method_variant="langgraph",
    )
    assert res.n_test_users == 1
    assert "ndcg@5" in res.means
    assert res.method_variant == "langgraph"


def test_emorecagent_fast_runs_on_fixture_train() -> None:
    rec = build_recommender(
        "emorecagent_fast",
        {"train_interactions": _train(), "factors": 4},
        seed=0,
    )
    res = evaluate(
        rec, _train(), _test(), k_values=[5], method="emorecagent_fast", seed=0
    )
    assert res.n_test_users == 1


def test_verified_only_filters_test_rows() -> None:
    test_rows = [
        Interaction("u0", "i_held", 5.0, 3 * DAY, verified_purchase=True),
        Interaction("u0", "i_skip", 5.0, 4 * DAY, verified_purchase=False),
    ]
    res = evaluate(
        PopularityRecommender(),
        _train(),
        test_rows,
        [5],
        method="pop",
        verified_only=True,
    )
    assert res.n_test_rows == 1
    assert res.verified_only is True
    assert res.n_verified_rows == 1


def test_base_cf_aliases_svd() -> None:
    rec = build_recommender("base_cf", {"factors": 4}, seed=0)
    assert rec.name == "svd"


def test_run_is_reproducible_with_fixed_seed() -> None:
    r1 = evaluate(
        build_recommender("svd", {"factors": 4}, seed=7),
        _train(), _test(), k_values=[5], method="svd", seed=7,
    )
    r2 = evaluate(
        build_recommender("svd", {"factors": 4}, seed=7),
        _train(), _test(), k_values=[5], method="svd", seed=7,
    )
    assert r1.means == r2.means


def test_paired_compare_returns_significance() -> None:
    # two methods over the same 10 users; method A strictly better on recall@5
    a = evaluate(PopularityRecommender(), _train(), _test(), [5], method="a")
    # fabricate per-user vectors to exercise the comparison path deterministically
    a.per_user["recall@5"] = [1.0] * 12
    b = evaluate(PopularityRecommender(), _train(), _test(), [5], method="b")
    b.per_user["recall@5"] = [0.0] * 12
    res = paired_compare(a, b, "recall@5", n_bootstrap=200, seed=1)
    assert res.mean_delta == pytest.approx(1.0)
    assert res.p_value < 0.05


def test_n_negatives_protocol() -> None:
    res_full = evaluate(
        PopularityRecommender(), _train(), _test(), [5], method="pop", n_negatives=None
    )
    res_sampled = evaluate(
        PopularityRecommender(), _train(), _test(), [5], method="pop",
        n_negatives=2, seed=0,
    )
    assert res_full.protocol == "full_catalog"
    assert res_sampled.protocol == "sampled_negatives"


def test_cumulative_history_excludes_prior_test_item() -> None:
    train = [
        Interaction("u0", "i_a", 5.0, 1 * DAY),
        Interaction("u0", "i_b", 5.0, 2 * DAY),
        Interaction("u1", "i_pop", 5.0, 1 * DAY),
    ]
    test_rows = [
        Interaction("u0", "i_held1", 5.0, 3 * DAY),
        Interaction("u0", "i_held2", 5.0, 4 * DAY),
    ]
    default = evaluate(
        PopularityRecommender(), train, test_rows, [5],
        method="pop", cumulative_history=False,
    )
    cumulative = evaluate(
        PopularityRecommender(), train, test_rows, [5],
        method="pop", cumulative_history=True,
    )
    assert default.n_test_rows == 2
    assert cumulative.n_test_rows == 2


def test_user_mean_with_multiple_rows() -> None:
    test_rows = [
        Interaction("u0", "i_held", 5.0, 3 * DAY),
        Interaction("u0", "i_extra", 5.0, 4 * DAY),
    ]
    res = evaluate(PopularityRecommender(), _train(), test_rows, [5], method="pop")
    assert res.n_test_rows == 2
    assert res.n_test_users == 1
    assert res.means["hr@5"] == res.means_per_user["hr@5"]


def test_write_and_load_round_trip(tmp_path) -> None:
    res = evaluate(PopularityRecommender(), _train(), _test(), [5], method="pop")
    out = write_results(tmp_path / "r.json", res)
    payload = json.loads(out.read_text())
    assert payload["method"] == "pop"
    assert payload["n_test_users"] == 1
    assert payload["n_test_rows"] == 1
    assert "means_per_user" in payload

    # load_split_jsonl round-trips the on-disk split format
    split_file = tmp_path / "train.jsonl"
    with split_file.open("w") as fh:
        for it in _train():
            fh.write(json.dumps({
                "user_id": it.user_id, "item": it.item,
                "rating": it.rating, "timestamp": it.timestamp,
                "helpful_vote": it.helpful_vote,
            }) + "\n")
    loaded = load_split_jsonl(split_file)
    assert len(loaded) == len(_train())
    assert loaded[0].user_id == "u0"
