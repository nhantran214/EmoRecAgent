"""Integration tests for emorecagent_hgt wiring."""

from __future__ import annotations

import numpy as np

from emorecagent.data.types import Interaction
from emorecagent.eval.runner import build_recommender, evaluate
from emorecagent.hgt.embeddings import EmbeddingStore
from emorecagent.hgt.retriever import HGTRetriever
from emorecagent.hgt.schema import OTHER_ASPECT

DAY = 86_400_000


def _train() -> list[Interaction]:
    data = []
    for u in range(3):
        data.append(Interaction(f"u{u}", "i_pop", 5.0, 1 * DAY))
        data.append(Interaction(f"u{u}", f"i_tail{u}", 5.0, 2 * DAY))
    return data


def _test() -> list[Interaction]:
    return [Interaction("u0", "i_held", 5.0, 3 * DAY)]


def _mock_hgt_retriever() -> HGTRetriever:
    users = [f"u{u}" for u in range(3)] + ["u0"]
    items = ["i_pop", "i_tail0", "i_tail1", "i_tail2", "i_held"]
    rng = np.random.RandomState(0)
    store = EmbeddingStore(
        user_ids=sorted(set(users)),
        item_ids=items,
        aspect_ids=["scent", OTHER_ASPECT],
        user_embeddings=rng.randn(len(set(users)), 8).astype(np.float32),
        item_embeddings=rng.randn(len(items), 8).astype(np.float32),
        aspect_embeddings=rng.randn(2, 8).astype(np.float32),
        meta={},
    )
    return HGTRetriever(store, pool_size=2)


def test_build_recommender_emorecagent_hgt_with_mock(monkeypatch):
    train = _train()

    class _Hgt:
        pool_size = 2

    class _Cfg:
        hgt = _Hgt()

    def _fake_from_config(config, *, seed=0):
        del config, seed
        return _mock_hgt_retriever()

    monkeypatch.setattr(
        "emorecagent.hgt.retriever.HGTRetriever.from_config",
        _fake_from_config,
    )
    rec = build_recommender(
        "emorecagent_hgt",
        {
            "train_interactions": train,
            "cf_backend": "hgt",
            "factors": 4,
            "use_llm_cot": False,
            "use_reflection": False,
            "app_config": _Cfg(),
        },
        seed=0,
    )
    assert rec.name == "emorecagent_hgt"
    res = evaluate(
        rec,
        train,
        _test(),
        k_values=[5],
        method="emorecagent_hgt",
        seed=0,
    )
    assert res.n_test_users == 1
