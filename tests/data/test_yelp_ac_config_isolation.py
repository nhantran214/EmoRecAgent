"""Yelp_AC category configs must not collide with review-track Yelp."""

from __future__ import annotations

from emorecagent.config import load_config, validate_category_path_isolation


def test_yelp_ac_configs_load_and_isolate() -> None:
    review = load_config("configs/categories/Yelp.yaml")
    ac = load_config("configs/categories/Yelp_AC.yaml")
    align = load_config("configs/categories/Yelp_AC_emorecagent_align.yaml")
    s1 = load_config("configs/categories/Yelp_AC_emorecagent_stage1_baseline.yaml")

    assert review.data.category == "Yelp"
    assert ac.data.category == "Yelp_AC"
    assert align.data.category == "Yelp_AC"
    assert s1.data.category == "Yelp_AC"

    assert ac.data.out_dir != review.data.out_dir
    assert "Yelp_AC" in ac.data.out_dir
    assert "Yelp_AC" not in review.data.out_dir.replace("Yelp_AC", "")

    assert validate_category_path_isolation(ac) == []
    assert validate_category_path_isolation(align) == []
    assert validate_category_path_isolation(s1) == []
    assert validate_category_path_isolation(review) == []


def test_yelp_ac_protocol_knobs() -> None:
    ac = load_config("configs/categories/Yelp_AC.yaml")
    align = load_config("configs/categories/Yelp_AC_emorecagent_align.yaml")

    assert ac.data.split_method == "leave_last_out"
    assert ac.data.k_core == 5
    assert ac.data.min_timestamp_s == 1_546_264_800
    assert ac.data.max_timestamp_s == 1_577_714_400
    assert ac.absa.enabled is False
    assert ac.tisasrec_align.test_history == "train_valid"
    assert ac.tisasrec_align.stage1_epochs == 200
    assert ac.tisasrec_align.batch_size == 256
    assert ac.tisasrec_align.lr == 0.0001

    assert align.tisasrec_align.preference_source == "item_metadata"
    assert align.tisasrec_align.cross_user_mode == "id_only"
    assert align.tisasrec_align.guardrail_mode == "reorder_head"
    assert align.absa.enabled is False


def test_review_yelp_unchanged_defaults() -> None:
    review = load_config("configs/categories/Yelp.yaml")
    assert review.absa.enabled is True
    assert review.data.split_method == "chronological_ratio"
    assert review.tisasrec_align.preference_source == "absa"
    assert review.tisasrec_align.cross_user_mode == "review_text"
    assert review.tisasrec_align.test_history == "train"


def test_yelp_ac_tisasrec_paper_config_uses_ce() -> None:
    paper = load_config("configs/categories/Yelp_AC_tisasrec_paper.yaml")
    assert paper.data.category == "Yelp_AC"
    assert paper.data.dedup_user_item is False
    # Paper track Stage-1 is RecBole TiSASRec (not in-repo ERA trainer).
    assert paper.tisasrec_align.stage1_backend == "recbole"
    assert "recbole_stage1_bundle" in paper.tisasrec_align.recbole_bundle_path
    assert paper.tisasrec_align.test_history == "train_valid"
    # Must not collide with EmoRecAgent BCE checkpoint path.
    emorec = load_config("configs/categories/Yelp_AC.yaml")
    assert (
        paper.tisasrec_align.stage1_checkpoint_path
        != emorec.tisasrec_align.stage1_checkpoint_path
    )
    # Amazon / BCE / default Yelp_AC keep ERA Stage-1 backend.
    assert emorec.tisasrec_align.stage1_backend == "era"
    amazon = load_config("configs/default.yaml")
    assert amazon.tisasrec_align.stage1_backend == "era"
    review = load_config("configs/categories/Yelp.yaml")
    assert review.tisasrec_align.stage1_backend == "era"


def test_yelp_ac_align_paper_uses_recbole_backbone() -> None:
    """Stage-2 on the RecBole Stage-1 bundle (not the BCE align backbone)."""
    align_paper = load_config(
        "configs/categories/Yelp_AC_emorecagent_align_paper.yaml"
    )
    assert align_paper.data.category == "Yelp_AC"
    assert align_paper.tisasrec_align.stage1_backend == "recbole"
    assert "recbole_stage1_bundle" in align_paper.tisasrec_align.recbole_bundle_path
    # Full Stage-2 rerank on the no-review adaptations.
    assert align_paper.tisasrec_align.stage1_only is False
    assert align_paper.tisasrec_align.stage2_mode == "rerank"
    assert align_paper.tisasrec_align.preference_source == "item_metadata"
    assert align_paper.tisasrec_align.cross_user_mode == "id_only"
    # Stage-2 must not require ABSA on this track.
    assert align_paper.absa.enabled is False
    # Must not reuse the Amazon-style BCE align checkpoint path.
    bce_align = load_config("configs/categories/Yelp_AC_emorecagent_align.yaml")
    assert bce_align.tisasrec_align.stage1_backend == "era"
    assert (
        align_paper.tisasrec_align.recbole_bundle_path
        != bce_align.tisasrec_align.stage1_checkpoint_path
    )
    assert validate_category_path_isolation(align_paper) == []


def test_review_and_amazon_tracks_unchanged_by_align_paper() -> None:
    """Scope isolation: Yelp review + Amazon defaults keep their conventions."""
    review = load_config("configs/categories/Yelp.yaml")
    assert review.absa.enabled is True
    assert review.tisasrec_align.stage1_backend == "era"
    amazon = load_config("configs/default.yaml")
    assert amazon.data.dedup_user_item is True
    assert amazon.absa.enabled is True
    assert amazon.tisasrec_align.stage1_backend == "era"
