"""Alignment MLP and InfoNCE tests."""

from __future__ import annotations

import torch

from emorecagent.tisasrec_align.alignment_mlp import AlignmentMLP
from emorecagent.tisasrec_align.infonce import infonce_loss
from emorecagent.tisasrec_align.text_encoder import HashEncoder


def test_alignment_mlp_shape():
    mlp = AlignmentMLP(768, 64, activation="elu")
    x = torch.randn(4, 768)
    out = mlp(x)
    assert out.shape == (4, 64)


def test_infonce_loss_ordering():
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    pos = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    neg = torch.tensor([[[0.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]])
    low = float(infonce_loss(q, pos, neg, temperature=0.07))
    random_neg = torch.randn(2, 8, 2)
    high = float(infonce_loss(q, pos, random_neg, temperature=0.07))
    assert low < high


def test_hash_encoder_deterministic():
    enc = HashEncoder(dim=32)
    a = enc.encode(["hello"], device=torch.device("cpu"))
    b = enc.encode(["hello"], device=torch.device("cpu"))
    assert torch.allclose(a, b)
