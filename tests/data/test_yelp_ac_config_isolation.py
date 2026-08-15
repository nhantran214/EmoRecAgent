"""Yelp_AC category configs must not collide with review-track Yelp."""

from __future__ import annotations

from emorecagent.config import load_config, validate_category_path_isolation


def test_yelp_ac_configs_load_and_isolate() -> None:
    review = load_config("configs/categories/Yelp.yaml")
    ac = load_config("configs/categories/Yelp_AC.yaml")
    align_a = load_config("configs/legacy/categories/Yelp_AC_emorecagent_align.yaml")
    s1_a = load_config(
        "configs/legacy/categories/Yelp_AC_emorecagent_stage1_baseline.yaml"
    )

    assert review.data.category == "Yelp"
    assert ac.data.category == "Yelp_AC"
    assert align_a.data.category == "Yelp_AC"
    assert s1_a.data.category == "Yelp_AC"

    assert ac.data.out_dir != review.data.out_dir
    assert "Yelp_AC" in ac.data.out_dir
    assert "Yelp_AC" not in review.data.out_dir.replace("Yelp_AC", "")

    assert validate_category_path_isolation(ac) == []
    assert validate_category_path_isolation(align_a) == []
    assert validate_category_path_isolation(s1_a) == []
    assert validate_category_path_isolation(review) == []


def test_yelp_ac_method_knobs() -> None:
    ac = load_config("configs/categories/Yelp_AC.yaml")

    assert ac.data.split_method == "leave_last_out"
    assert ac.data.k_core == 5
    assert ac.data.min_timestamp_s == 1_546_264_800
    assert ac.data.max_timestamp_s == 1_577_714_400
    assert ac.absa.enabled is False
    assert ac.tisasrec_align.test_history == "train_valid"
    assert ac.tisasrec_align.stage1_backend == "recbole"
    assert "tisasrec_option_b" in ac.tisasrec_align.recbole_bundle_path
    assert ac.tisasrec_align.batch_size == 256
    assert ac.tisasrec_align.lr == 0.0001
    assert ac.tisasrec_align.preference_source == "item_metadata"
    assert ac.tisasrec_align.cross_user_mode == "id_only"
    assert ac.tisasrec_align.guardrail_mode == "context_dependent"
    assert ac.tisasrec_align.llm_rerank_mode == "listwise"
    assert ac.tisasrec_align.stage2_score == "ltr_llm"
    assert ac.tisasrec_align.llm_pool_cap == 100
    assert ac.tisasrec_align.llm_blend_beta == 0.0
    assert ac.tisasrec_align.llm_gate_enabled is True
    assert ac.tisasrec_align.stage1_only is False
    assert "Yelp_AC" in ac.tisasrec_align.item_potential_ltr_path


def test_review_yelp_defaults() -> None:
    review = load_config("configs/categories/Yelp.yaml")
    assert review.absa.enabled is True
    assert review.data.split_method == "chronological_ratio"
    assert review.tisasrec_align.preference_source == "absa"
    assert review.tisasrec_align.cross_user_mode == "review_text"
    assert review.tisasrec_align.test_history == "train"
    assert review.tisasrec_align.stage1_backend == "recbole"
    assert review.tisasrec_align.llm_rerank_mode == "listwise"
    assert review.tisasrec_align.stage2_score == "ltr_llm"
    assert "tisasrec_option_b" in review.tisasrec_align.recbole_bundle_path


def test_yelp_ac_legacy_paper_config_uses_ce() -> None:
    paper = load_config("configs/legacy/categories/Yelp_AC_tisasrec_paper.yaml")
    assert paper.data.category == "Yelp_AC"
    assert paper.data.dedup_user_item is False
    assert paper.tisasrec_align.stage1_backend == "recbole"
    assert "recbole_stage1_bundle" in paper.tisasrec_align.recbole_bundle_path
    assert "tisasrec_paper" in paper.tisasrec_align.recbole_bundle_path
    assert paper.tisasrec_align.test_history == "train_valid"
    # Category artifact root differs from the archived paper RecBole bundle.
    current = load_config("configs/categories/Yelp_AC.yaml")
    assert (
        paper.tisasrec_align.recbole_bundle_path
        != current.tisasrec_align.recbole_bundle_path
    )
    amazon = load_config("configs/default.yaml")
    assert amazon.tisasrec_align.stage1_backend == "era"


def test_yelp_ac_legacy_align_paper_uses_recbole_backbone() -> None:
    """Legacy Stage-2 paper overlay on the RecBole Stage-1 bundle."""
    align_paper = load_config(
        "configs/legacy/categories/Yelp_AC_emorecagent_align_paper.yaml"
    )
    assert align_paper.data.category == "Yelp_AC"
    assert align_paper.tisasrec_align.stage1_backend == "recbole"
    assert "recbole_stage1_bundle" in align_paper.tisasrec_align.recbole_bundle_path
    assert align_paper.tisasrec_align.stage1_only is False
    assert align_paper.tisasrec_align.stage2_mode == "rerank"
    assert align_paper.tisasrec_align.preference_source == "item_metadata"
    assert align_paper.tisasrec_align.cross_user_mode == "id_only"
    assert align_paper.absa.enabled is False
    bce_align = load_config(
        "configs/legacy/categories/Yelp_AC_emorecagent_align.yaml"
    )
    assert bce_align.tisasrec_align.stage1_backend == "era"
    assert (
        align_paper.tisasrec_align.recbole_bundle_path
        != bce_align.tisasrec_align.stage1_checkpoint_path
    )
    assert validate_category_path_isolation(align_paper) == []


def test_amazon_default_keeps_era_backend() -> None:
    """``configs/default.yaml`` remains ERA; Beauty method lives in the category file."""
    amazon = load_config("configs/default.yaml")
    assert amazon.data.dedup_user_item is True
    assert amazon.absa.enabled is True
    assert amazon.tisasrec_align.stage1_backend == "era"
    beauty_b = load_config("configs/categories/Beauty_and_Personal_Care.yaml")
    assert beauty_b.tisasrec_align.stage1_backend == "recbole"
