"""Smoke test Stage 1 training on toy data."""

from __future__ import annotations

from pathlib import Path

import torch

from emorecagent.data.types import Interaction
from emorecagent.tisasrec_align.schema import TiSASRecArgs
from emorecagent.tisasrec_align.train_stage1 import train_stage1


def test_train_stage1_smoke(tmp_path: Path):
    train = [
        Interaction("u1", "i1", 5.0, 1000),
        Interaction("u1", "i2", 5.0, 2000),
        Interaction("u2", "i1", 4.0, 1500),
        Interaction("u2", "i3", 4.0, 2500),
    ]
    valid = [Interaction("u1", "i3", 5.0, 3000), Interaction("u2", "i4", 5.0, 3500)]
    test = [Interaction("u1", "i4", 5.0, 4000, verified_purchase=True)]
    args = TiSASRecArgs(maxlen=5, hidden_units=8, num_blocks=1, time_span=4)
    ckpt = tmp_path / "ckpt.pt"
    e_i = tmp_path / "e_i.pt"
    result = train_stage1(
        train=train,
        valid=valid,
        args=args,
        checkpoint_path=ckpt,
        e_i_matrix_path=e_i,
        device=torch.device("cpu"),
        epochs=2,
        batch_size=2,
        steps_per_epoch=2,
        early_stop_patience=5,
        require_valid=True,
        valid_eval_all=True,
        valid_eval_max_pairs=10,
        valid_eval_batch_size=4,
        lr_scheduler_enabled=False,
        stage1_loss="multi_bce",
        num_train_negatives=2,
        test=test,
        verified_only=False,
        seed=0,
    )
    assert ckpt.exists()
    assert e_i.exists()
    assert result.best_epoch >= 1
    assert result.post_train_test is not None


def test_train_stage1_fits_train_only_not_valid(tmp_path: Path, monkeypatch):
    """Valid must stay out of next-item loss sequences (early-stop holdout)."""
    from emorecagent.tisasrec_align import train_stage1 as mod

    train = [
        Interaction("u1", "i1", 5.0, 1000),
        Interaction("u1", "i2", 5.0, 2000),
        Interaction("u2", "i1", 4.0, 1500),
        Interaction("u2", "i3", 4.0, 2500),
    ]
    valid = [Interaction("u1", "i3", 5.0, 3000), Interaction("u2", "i4", 5.0, 3500)]
    captured: dict[str, set[tuple[str, str]]] = {}
    real = mod.build_user_sequences

    def _capture(interactions, id_maps, **kwargs):
        captured["pairs"] = {(it.user_id, it.item) for it in interactions}
        return real(interactions, id_maps, **kwargs)

    monkeypatch.setattr(mod, "build_user_sequences", _capture)
    args = TiSASRecArgs(maxlen=5, hidden_units=8, num_blocks=1, time_span=4)
    train_stage1(
        train=train,
        valid=valid,
        args=args,
        checkpoint_path=tmp_path / "ckpt.pt",
        e_i_matrix_path=tmp_path / "e_i.pt",
        device=torch.device("cpu"),
        epochs=1,
        batch_size=2,
        steps_per_epoch=1,
        early_stop_patience=1,
        require_valid=True,
        valid_eval_all=True,
        lr_scheduler_enabled=False,
        stage1_loss="bce",
        num_train_negatives=1,
        test=None,
        seed=0,
    )
    assert captured["pairs"] == {("u1", "i1"), ("u1", "i2"), ("u2", "i1"), ("u2", "i3")}
    assert ("u1", "i3") not in captured["pairs"]
    assert ("u2", "i4") not in captured["pairs"]
