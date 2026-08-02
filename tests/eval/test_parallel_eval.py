"""Parallel eval runner smoke tests."""

from __future__ import annotations

import pytest

from emorecagent.baselines.popularity import PopularityRecommender
from emorecagent.data.types import Interaction
from emorecagent.eval.runner import evaluate

DAY = 86_400_000


def _train() -> list[Interaction]:
    data: list[Interaction] = []
    for u in range(6):
        data.append(Interaction(f"u{u}", "i_pop", 5.0, 1 * DAY))
        data.append(Interaction(f"u{u}", f"i_tail{u}", 5.0, 2 * DAY))
    return data


def _test_rows(n: int = 8) -> list[Interaction]:
    return [
        Interaction(f"u{i % 3}", f"i_held{i}", 5.0, (i + 3) * DAY, verified_purchase=True)
        for i in range(n)
    ]


def test_parallel_eval_matches_serial_popularity() -> None:
    train = _train()
    test = _test_rows()
    kwargs = dict(
        train=train,
        test=test,
        k_values=[1, 5],
        method="pop",
        seed=0,
        verified_only=False,
    )
    serial = evaluate(PopularityRecommender(), **kwargs, parallel_workers=1)
    parallel = evaluate(PopularityRecommender(), **kwargs, parallel_workers=4)
    assert serial.user_ids == parallel.user_ids
    assert serial.means == parallel.means
    assert serial.per_user == parallel.per_user


def test_parallel_user_batch_matches_serial() -> None:
    """user_batch is the Yelp_AC / paper protocol; parity must hold there too."""
    train = _train()
    test = _test_rows()
    kwargs = dict(
        train=train,
        test=test,
        k_values=[1, 5],
        method="pop",
        seed=0,
        verified_only=False,
        eval_protocol="user_batch",
        aggregation="user_mean",
    )
    serial = evaluate(PopularityRecommender(), **kwargs, parallel_workers=1)
    parallel = evaluate(PopularityRecommender(), **kwargs, parallel_workers=4)
    assert serial.user_ids == parallel.user_ids
    assert serial.means == parallel.means
    assert serial.per_user == parallel.per_user
    assert serial.n_test_rows == parallel.n_test_rows


def test_parallel_user_batch_uses_query_ts_per_call() -> None:
    """Threads must not rely on prepare_user_query state set by another thread."""

    class _QueryTsRecommender(PopularityRecommender):
        def __init__(self) -> None:
            super().__init__()
            self.seen_query_ts: dict[str, int] = {}

        def rank(
            self,
            user_id: str,
            candidates: list[str],
            *,
            query_ts_ms: int | None = None,
        ) -> list[str]:
            assert query_ts_ms is not None
            self.seen_query_ts[user_id] = query_ts_ms
            return super().rank(user_id, candidates)

    rec = _QueryTsRecommender()
    result = evaluate(
        rec,
        _train(),
        _test_rows(),
        k_values=[5],
        method="pop",
        seed=0,
        verified_only=False,
        eval_protocol="user_batch",
        aggregation="user_mean",
        parallel_workers=3,
    )
    assert set(rec.seen_query_ts) == set(result.user_ids)


def test_parallel_eval_rejects_cumulative_history() -> None:
    with pytest.raises(ValueError, match="cumulative_history"):
        evaluate(
            PopularityRecommender(),
            _train(),
            _test_rows(2),
            k_values=[5],
            method="pop",
            cumulative_history=True,
            parallel_workers=2,
        )
