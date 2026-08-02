"""Unit tests for pure TiSASRec model."""

from __future__ import annotations

import torch

from emorecagent.tisasrec_align.model import TiSASRecModel
from emorecagent.tisasrec_align.schema import TiSASRecArgs


def test_forward_shapes_and_grad():
    args = TiSASRecArgs(
        maxlen=5,
        hidden_units=16,
        num_blocks=1,
        num_heads=1,
        time_span=8,
    )
    model = TiSASRecModel(3, args)
    b = 2
    seq = torch.tensor([[0, 0, 1, 2, 0], [0, 1, 0, 0, 0]], dtype=torch.long)
    time_mat = torch.zeros(b, 5, 5, dtype=torch.long)
    pos = torch.tensor([[0, 0, 2, 0, 0], [0, 2, 0, 0, 0]], dtype=torch.long)
    neg = torch.tensor([[0, 0, 1, 0, 0], [0, 1, 0, 0, 0]], dtype=torch.long)

    pos_logits, neg_logits = model(seq, time_mat, pos, neg)
    assert pos_logits.shape == (b, 5)
    assert neg_logits.shape == (b, 5)

    neg_k = torch.zeros(b, 5, 2, dtype=torch.long)
    neg_k[:, 2, 0] = 1
    neg_k[:, 2, 1] = 2
    neg_k[1, 1, 0] = 2
    neg_k[1, 1, 1] = 1
    _, neg_logits_k = model(seq, time_mat, pos, neg_k)
    assert neg_logits_k.shape == (b, 5, 2)

    loss = pos_logits.sum() + neg_logits.sum()
    loss.backward()
    assert model.item_emb.weight.grad is not None

    hu = model.user_repr(seq[:1], time_mat[:1])
    assert hu.shape == (1, args.hidden_units)

    table = model.all_item_embeddings()
    assert table[1:].shape[0] == 3
