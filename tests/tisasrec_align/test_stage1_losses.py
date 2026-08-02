"""Tests for Stage 1 ranking losses."""

from __future__ import annotations

import torch

from emorecagent.tisasrec_align.train_stage1 import compute_stage1_loss


def test_multi_bce_finite_and_differentiable():
    pos = torch.tensor([[1.0, 2.0, 0.0]], requires_grad=True)
    neg = torch.tensor([[[0.0, -1.0], [0.5, -0.5], [0.0, 0.0]]], requires_grad=True)
    loss = compute_stage1_loss("multi_bce", pos, neg)
    assert torch.isfinite(loss)
    loss.backward()
    assert pos.grad is not None


def test_bpr_finite():
    pos = torch.tensor([[1.0, 2.0, 0.0]])
    neg = torch.tensor([[[0.0, -1.0], [0.5, -0.5], [0.0, 0.0]]])
    loss = compute_stage1_loss("bpr", pos, neg)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_ce_loss_finite_and_differentiable() -> None:
    from emorecagent.tisasrec_align.model import TiSASRecModel, init_model_weights
    from emorecagent.tisasrec_align.schema import TiSASRecArgs
    from emorecagent.tisasrec_align.train_stage1 import compute_ce_loss

    args = TiSASRecArgs(
        hidden_units=8,
        maxlen=4,
        num_blocks=1,
        num_heads=1,
        dropout_rate=0.0,
        time_span=8,
        l2_emb=0.0,
        inner_size=16,
    )
    model = TiSASRecModel(5, args)
    init_model_weights(model)
    # Left-padded row: last non-pad at index 2 → target pos[2]=4.
    seq = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
    time = torch.zeros(1, 4, 4, dtype=torch.long)
    pos = torch.tensor([[2, 3, 4, 0]], dtype=torch.long)
    loss = compute_ce_loss(model, seq, time, pos)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.item_emb.weight.grad is not None


def test_ce_loss_supervises_every_nonpad_target() -> None:
    """Kang packing: CE must cover all timestep targets, not just the last one.

    Supervising only the last position drops ~8x of the training signal versus
    RecBole's prefix-augmented dataset (same total targets, different packing).
    """
    from emorecagent.tisasrec_align.model import TiSASRecModel, init_model_weights
    from emorecagent.tisasrec_align.schema import TiSASRecArgs
    from emorecagent.tisasrec_align.train_stage1 import compute_ce_loss

    args = TiSASRecArgs(
        hidden_units=8,
        maxlen=4,
        num_blocks=1,
        num_heads=1,
        dropout_rate=0.0,
        time_span=8,
        l2_emb=0.0,
    )
    model = TiSASRecModel(5, args)
    init_model_weights(model)
    seq = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    time = torch.zeros(1, 4, 4, dtype=torch.long)
    # Three non-pad targets; gradient must reflect all of them.
    pos_all = torch.tensor([[0, 2, 3, 4]], dtype=torch.long)
    pos_last = torch.tensor([[0, 0, 0, 4]], dtype=torch.long)
    loss_all = compute_ce_loss(model, seq, time, pos_all)
    loss_last = compute_ce_loss(model, seq, time, pos_last)
    assert torch.isfinite(loss_all) and torch.isfinite(loss_last)
    # Different target counts must produce different losses.
    assert not torch.isclose(loss_all, loss_last)


def test_ffn_inner_size_expands() -> None:
    from emorecagent.tisasrec_align.tisasrec_layers import PointWiseFeedForward

    ffn = PointWiseFeedForward(8, 0.0, inner_size=32)
    assert ffn.conv1.out_channels == 32
    assert ffn.conv2.in_channels == 32
    assert ffn.conv2.out_channels == 8
    x = torch.randn(2, 5, 8)
    y = ffn(x)
    assert y.shape == x.shape


def test_bce_backward_compatible():
    pos = torch.tensor([[1.0, 2.0, 0.0]])
    neg = torch.tensor([[0.0, -1.0, 0.0]])
    loss = compute_stage1_loss("bce", pos, neg)
    assert torch.isfinite(loss)
