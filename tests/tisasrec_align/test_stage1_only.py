"""stage1_only disables fusion even when T_u is present."""

from __future__ import annotations

from collections import defaultdict

import torch

from emorecagent.sequential.id_maps import IdMaps
from emorecagent.tisasrec_align.alignment_mlp import AlignmentMLP
from emorecagent.tisasrec_align.checkpoint import AlignBundle
from emorecagent.tisasrec_align.full_rank_recommender import AlignFullRankRecommender
from emorecagent.tisasrec_align.model import TiSASRecArgs, TiSASRecModel
from emorecagent.tisasrec_align.tu_cache import TuCacheRow


def _minimal_bundle(hidden: int = 8) -> AlignBundle:
    args = TiSASRecArgs(
        hidden_units=hidden,
        maxlen=5,
        num_blocks=1,
        num_heads=1,
        dropout_rate=0.0,
        time_span=16,
        l2_emb=0.0,
    )
    model = TiSASRecModel(3, args)
    mlp = AlignmentMLP(768, hidden)
    e_i = torch.randn(4, hidden)
    return AlignBundle(
        tisasrec=model,
        alignment_mlp=mlp,
        item_ids=["i1", "i2", "i3"],
        e_i_matrix=e_i,
        args=args,
        tau=0.1,
    )


def test_stage1_only_forces_alpha_one() -> None:
    bundle = _minimal_bundle()
    id_maps = IdMaps(
        user_to_idx={"u1": 1},
        item_to_idx={"i1": 1, "i2": 2, "i3": 3},
    )
    tu_cache = {
        "u1|1000": TuCacheRow(
            user_id="u1",
            query_ts_ms=1000,
            T_u="user likes skincare",
            has_reviews=True,
        )
    }
    rec = AlignFullRankRecommender(
        bundle,
        id_maps,
        tu_cache,
        fusion_alpha=0.7,
        device=torch.device("cpu"),
        use_hash_encoder=True,
        stage1_only=True,
    )
    rec._user_events = defaultdict(list, {"u1": [(500, "i1")]})
    assert rec._alpha_eff("u1", 1000) == 1.0

    rec_fused = AlignFullRankRecommender(
        bundle,
        id_maps,
        tu_cache,
        fusion_alpha=0.7,
        device=torch.device("cpu"),
        use_hash_encoder=True,
        stage1_only=False,
    )
    rec_fused._user_events = rec._user_events
    assert rec_fused._alpha_eff("u1", 1000) == 0.7
