"""Smoke tests for HGT retriever (numpy embeddings, no GPU)."""

from __future__ import annotations

import numpy as np

from emorecagent.hgt.embeddings import EmbeddingStore
from emorecagent.hgt.retriever import HGTRetriever
from emorecagent.hgt.schema import OTHER_ASPECT
from emorecagent.hgt.aspect_vocab import AspectVocab


def _store() -> EmbeddingStore:
    return EmbeddingStore(
        user_ids=["u1"],
        item_ids=["i1", "i2"],
        aspect_ids=["scent", OTHER_ASPECT],
        user_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        item_embeddings=np.array(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        ),
        aspect_embeddings=np.array(
            [[0.5, 0.5], [0.0, 0.0]],
            dtype=np.float32,
        ),
        meta={},
    )


def test_retrieve_ranks_by_dot_product():
    retriever = HGTRetriever(_store(), pool_size=1)
    ranked = retriever.retrieve("u1", 1, None, ["i1", "i2"])
    assert ranked == ["i1"]


def test_injection_shifts_ranking():
    vocab = AspectVocab(aspects=("scent", OTHER_ASPECT), other_id=1)
    retriever = HGTRetriever(_store(), aspect_vocab=vocab, pool_size=2)
    base = retriever.retrieve("u1", 2, None, ["i1", "i2"])
    shifted = retriever.retrieve("u1", 2, {"scent": 2.0}, ["i1", "i2"])
    assert base[0] == "i1"
    assert shifted[0] == "i1"


def test_unknown_user_scores_zero():
    retriever = HGTRetriever(_store())
    scores = retriever.score("missing", ["i1", "i2"])
    assert scores["i1"] == 0.0 and scores["i2"] == 0.0
