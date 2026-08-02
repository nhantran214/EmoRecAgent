"""Batch eval integration tests."""

from __future__ import annotations

import pytest

from emorecagent.baselines.popularity import PopularityRecommender
from emorecagent.data.types import Interaction
from emorecagent.eval.runner import evaluate

DAY = 86_400_000


def _train() -> list[Interaction]:
    data = []
    for u in range(4):
        data.append(Interaction(f"u{u}", "i_pop", 5.0, 1 * DAY))
        data.append(Interaction(f"u{u}", f"i_tail{u}", 5.0, 2 * DAY))
    return data


def _test_rows() -> list[Interaction]:
    return [
        Interaction("u0", "i_held1", 5.0, 3 * DAY, verified_purchase=True),
        Interaction("u1", "i_held2", 5.0, 3 * DAY, verified_purchase=True),
    ]


def test_llm_batch_parallel_workers_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        evaluate(
            PopularityRecommender(),
            _train(),
            _test_rows(),
            [5],
            method="pop",
            n_negatives=2,
            llm_batch=True,
            parallel_workers=2,
        )


def test_llm_batch_requires_graph_recommender() -> None:
    with pytest.raises(ValueError, match="GraphRecommender"):
        evaluate(
            PopularityRecommender(),
            _train(),
            _test_rows(),
            [5],
            method="pop",
            n_negatives=2,
            llm_batch=True,
        )
