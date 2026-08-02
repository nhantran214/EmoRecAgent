"""Alignment checkpoint encoder metadata."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from emorecagent.data.types import Interaction
from emorecagent.tisasrec_align.train_stage2 import train_stage2
from emorecagent.tisasrec_align.tu_cache import TuCacheRow, cache_key


def test_train_stage2_saves_use_hash_encoder_meta(tmp_path: Path) -> None:
    train = [
        Interaction(user_id="u1", item="i1", rating=5.0, timestamp=1000, verified_purchase=True),
        Interaction(user_id="u1", item="i2", rating=5.0, timestamp=2000, verified_purchase=True),
        Interaction(user_id="u2", item="i1", rating=4.0, timestamp=1500, verified_purchase=True),
        Interaction(user_id="u2", item="i3", rating=4.0, timestamp=2500, verified_purchase=True),
    ]
    tu_path = tmp_path / "tu.jsonl"
    rows = []
    for it in train:
        rows.append(
            {
                "key": cache_key(it.user_id, it.timestamp),
                "user_id": it.user_id,
                "query_ts_ms": it.timestamp,
                "T_u": f"prefs for {it.user_id}",
                "has_reviews": True,
            }
        )
    tu_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    item_num = 4
    e_i = torch.randn(item_num + 1, 8)
    ckpt = tmp_path / "align.pt"
    train_stage2(
        train=train,
        tu_cache_path=tu_path,
        e_i_matrix=e_i,
        hidden_dim=8,
        alignment_ckpt_path=ckpt,
        device=torch.device("cpu"),
        epochs=1,
        batch_size=2,
        tau_grid=(0.1,),
        seed=0,
    )
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert payload["meta"]["use_hash_encoder"] is True
