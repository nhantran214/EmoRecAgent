"""AlignFullRankRecommender must use checkpoint id_maps for correct E_I indexing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from emorecagent.sequential.id_maps import IdMaps
from emorecagent.tisasrec_align.full_rank_recommender import AlignFullRankRecommender


def test_from_config_uses_checkpoint_id_maps() -> None:
    ckpt_id_maps = IdMaps(
        user_to_idx={"u_train": 1, "u_valid": 2},
        item_to_idx={"i_train": 1, "i_valid": 2},
    )
    bundle = MagicMock()
    bundle.item_ids = ["i_train", "i_valid"]
    bundle.text_encoder_dim = 768

    with (
        patch(
            "emorecagent.tisasrec_align.full_rank_recommender.load_align_bundle",
            return_value=bundle,
        ),
        patch(
            "emorecagent.tisasrec_align.full_rank_recommender.load_tu_cache",
            return_value={},
        ),
        patch(
            "emorecagent.tisasrec_align.full_rank_recommender.load_stage1_id_maps",
            return_value=ckpt_id_maps,
        ) as mock_load_maps,
    ):
        from emorecagent.config import load_config

        cfg = load_config("configs/emorecagent_stage1_baseline.yaml")
        rec = AlignFullRankRecommender.from_config(cfg, train=[], seed=0)
        mock_load_maps.assert_called_once_with(cfg.tisasrec_align.stage1_checkpoint_path)
        assert rec._id_maps.item_to_idx == ckpt_id_maps.item_to_idx
        assert rec.catalog_items() == ["i_train", "i_valid"]
