"""U12 runner tests: split loading, evaluation, infra-method guard, reproducibility."""

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
    assert res.n_test_users == 1
    assert "ndcg@5" in res.means and "hr@1" in res.means
    assert all(len(v) == 1 for v in res.per_user.values())


def test_infra_methods_raise_until_pipeline_exists() -> None:
    with pytest.raises(NotImplementedError):
        build_recommender("emorecagent", {}, seed=0)
    with pytest.raises(NotImplementedError):
        build_recommender("aspect_aware", {}, seed=0)


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


def test_write_and_load_round_trip(tmp_path) -> None:
    res = evaluate(PopularityRecommender(), _train(), _test(), [5], method="pop")
    out = write_results(tmp_path / "r.json", res)
    payload = json.loads(out.read_text())
    assert payload["method"] == "pop"
    assert payload["n_test_users"] == 1

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
