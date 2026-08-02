"""Smoke tests for HetTiSASRec model forward pass."""

from __future__ import annotations

import torch

from emorecagent.hettisasrec.aspect_graph import AspectGraphBundle
from emorecagent.hettisasrec.model import HetTiSASRecArgs, HetTiSASRecModel


def _tiny_graph() -> AspectGraphBundle:
    return AspectGraphBundle(
        item_to_aspect_idx=torch.tensor([[0, 0], [1, 2], [0, 0]]),
        item_to_aspect_w=torch.tensor([[0.0, 0.0], [1.0, 0.5], [0.0, 0.0]]),
        aspect_to_item_idx=torch.tensor([[0, 0], [1, 0], [2, 0]]),
        aspect_to_item_w=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 0.0]]),
        n_items=2,
        n_aspects=2,
        item_ids=("i1", "i2"),
        aspect_ids=("a1", "a2"),
    )


def test_forward_and_predict_shapes():
    graph = _tiny_graph()
    args = HetTiSASRecArgs(
        maxlen=5,
        hidden_units=16,
        num_blocks=1,
        num_heads=1,
        time_span=8,
    )
    model = HetTiSASRecModel(2, 2, args, graph)
    b = 2
    seq = torch.tensor([[0, 0, 1, 2, 0], [0, 1, 0, 0, 0]], dtype=torch.long)
    time_mat = torch.zeros(b, 5, 5, dtype=torch.long)
    pos = torch.tensor([[0, 0, 2, 0, 0], [0, 2, 0, 0, 0]], dtype=torch.long)
    neg = torch.tensor([[0, 0, 1, 0, 0], [0, 1, 0, 0, 0]], dtype=torch.long)

    pos_logits, neg_logits = model(seq, time_mat, pos, neg)
    assert pos_logits.shape == (b, 5)
    assert neg_logits.shape == (b, 5)

    items = torch.tensor([[1, 2]], dtype=torch.long)
    scores = model.predict(seq[:1], time_mat[:1], items)
    assert scores.shape == (1, 2)


def test_pure_tisasrec_ablation_disables_graph():
    graph = _tiny_graph()
    args = HetTiSASRecArgs(
        maxlen=4,
        hidden_units=8,
        num_blocks=1,
        num_heads=1,
        time_span=4,
        use_aspect_enrichment=False,
    )
    model = HetTiSASRecModel(1, 2, args, graph)
    emb = model.item_encoder(torch.tensor([1, 2]))
    assert emb.shape == (2, 8)
