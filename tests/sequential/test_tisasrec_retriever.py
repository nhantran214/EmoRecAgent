"""Tests for TiSASRec retriever chronological protocol."""

from __future__ import annotations

import numpy as np
import torch

from emorecagent.data.types import Interaction
from emorecagent.sequential.id_maps import IdMaps
from emorecagent.sequential.seq_utils import TiSASRecArgs
from emorecagent.sequential.tisasrec_retriever import TiSASRecRetriever


class _RecordingModel:
    last_seq: np.ndarray | None = None

    def predict(self, users, seq, time_matrix, item_indices):  # noqa: ANN001
        _RecordingModel.last_seq = seq[0]
        del users, time_matrix
        return torch.zeros(1, len(item_indices[0]))


def _retriever() -> TiSASRecRetriever:
    id_maps = IdMaps(
        user_to_idx={"u1": 1},
        item_to_idx={"i1": 1, "i2": 2, "i3": 3},
    )
    args = TiSASRecArgs({"maxlen": 5, "time_span": 8}, "cpu")
    return TiSASRecRetriever(
        _RecordingModel(),
        id_maps,
        torch.device("cpu"),
        args,
        item_ids=["i1", "i2", "i3"],
    )


def test_prepare_user_query_excludes_future_interactions():
    rec = _retriever()
    train = [
        Interaction("u1", "i1", 5.0, 1000),
        Interaction("u1", "i2", 5.0, 2000),
        Interaction("u1", "i3", 5.0, 3000),
    ]
    rec.fit(train)
    rec.prepare_user_query("u1", 2500)
    rec.score("u1", ["i3"])
    seq = _RecordingModel.last_seq
    assert seq is not None
    assert 2 in seq  # i2 index
    assert 3 not in seq  # i3 at query_ts excluded


def test_unknown_user_returns_zero_scores():
    rec = _retriever()
    rec.fit([])
    scores = rec.score("unknown", ["i1", "i2"])
    assert scores == {"i1": 0.0, "i2": 0.0}
