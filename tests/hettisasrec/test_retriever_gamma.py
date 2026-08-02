"""Gamma bias tests for HetTiSASRec retriever."""

from __future__ import annotations

import numpy as np
import torch

from emorecagent.data.types import Interaction
from emorecagent.hettisasrec.aspect_graph import AspectGraphBundle
from emorecagent.hettisasrec.aspect_vocab import AspectVocab
from emorecagent.hettisasrec.model import HetTiSASRecArgs, HetTiSASRecModel
from emorecagent.hettisasrec.retriever import HetTiSASRecRetriever
from emorecagent.sequential.id_maps import IdMaps


class _SpyModel(HetTiSASRecModel):
    last_gamma: torch.Tensor | None = None

    def predict(self, log_seqs, time_matrices, item_indices, *, aspect_gamma=None):
        _SpyModel.last_gamma = aspect_gamma
        return torch.zeros(log_seqs.shape[0], item_indices.shape[1])


def _bundle() -> AspectGraphBundle:
    return AspectGraphBundle(
        item_to_aspect_idx=torch.tensor([[0, 0], [1, 0], [0, 0]]),
        item_to_aspect_w=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]),
        aspect_to_item_idx=torch.tensor([[0, 0], [1, 0], [0, 0]]),
        aspect_to_item_w=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]),
        n_items=2,
        n_aspects=1,
        item_ids=("i1", "i2"),
        aspect_ids=("aspect:quality",),
    )


def test_gammas_passed_to_predict():
    graph = _bundle()
    args = HetTiSASRecArgs(maxlen=4, hidden_units=8, num_blocks=1, num_heads=1)
    model = _SpyModel(1, 2, args, graph)
    id_maps = IdMaps(user_to_idx={"u1": 1}, item_to_idx={"i1": 1, "i2": 2})
    vocab = AspectVocab(aspects=("aspect:quality",), other_id=0)
    rec = HetTiSASRecRetriever(
        model,
        id_maps,
        graph,
        vocab,
        torch.device("cpu"),
        args,
        pool_size=50,
    )
    rec.fit([Interaction("u1", "i1", 5.0, 1000)])
    rec.score("u1", ["i2"], gammas={"quality": 0.8})
    assert _SpyModel.last_gamma is not None
    assert float(_SpyModel.last_gamma.abs().sum()) > 0
