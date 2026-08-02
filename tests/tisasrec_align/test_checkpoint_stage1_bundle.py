"""Stage-1-only bundle load skips alignment checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch

from emorecagent.tisasrec_align.checkpoint import load_align_bundle


def test_load_align_bundle_without_alignment(tmp_path) -> None:
    hidden = 8
    args_dict = {
        "hidden_units": hidden,
        "maxlen": 5,
        "num_blocks": 1,
        "num_heads": 1,
        "dropout_rate": 0.0,
        "time_span": 16,
        "l2_emb": 0.0,
    }
    from emorecagent.tisasrec_align.model import TiSASRecArgs, TiSASRecModel

    args = TiSASRecArgs(**args_dict)
    model = TiSASRecModel(4, args)
    stage1 = tmp_path / "stage1.pt"
    e_i = tmp_path / "e_i.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "meta": {
                "args": args_dict,
                "item_num": 4,
                "item_ids": ["i1", "i2", "i3"],
                "id_maps": {
                    "user_to_idx": {"u1": 1},
                    "item_to_idx": {"i1": 1, "i2": 2, "i3": 3},
                },
            },
        },
        stage1,
    )
    torch.save(torch.randn(4, hidden), e_i)

    bundle = load_align_bundle(
        stage1_ckpt=stage1,
        e_i_path=e_i,
        device=torch.device("cpu"),
        alignment_ckpt=None,
    )
    assert bundle.alignment_mlp is None
    assert bundle.e_i_matrix.shape == (4, hidden)
