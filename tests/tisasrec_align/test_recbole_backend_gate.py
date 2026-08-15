"""Category gate for RecBole Stage-1 backend (all five benchmarks)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from emorecagent.config import load_config
from emorecagent.tisasrec_align.stage1_factory import build_stage1_recommender


BENCHMARK_CATS = [
    "Beauty_and_Personal_Care",
    "Sports_and_Outdoors",
    "Toys_and_Games",
    "Yelp",
    "Yelp_AC",
]


def test_recbole_backend_rejected_for_unknown_category() -> None:
    cfg = load_config("configs/default.yaml")
    ta = cfg.tisasrec_align.model_copy(update={"stage1_backend": "recbole"})
    data = cfg.data.model_copy(update={"category": "Unknown_Cat"})
    cfg = cfg.model_copy(update={"tisasrec_align": ta, "data": data})
    with pytest.raises(ValueError, match="only supported for"):
        build_stage1_recommender(cfg, train=[])


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/categories/Beauty_and_Personal_Care.yaml",
        "configs/categories/Sports_and_Outdoors.yaml",
        "configs/categories/Toys_and_Games.yaml",
        "configs/categories/Yelp.yaml",
        "configs/categories/Yelp_AC.yaml",
    ],
)
def test_category_configs_use_recbole(config_path: str) -> None:
    cfg = load_config(config_path)
    assert cfg.data.category in BENCHMARK_CATS
    assert cfg.tisasrec_align.stage1_backend == "recbole"
    assert "tisasrec_option_b" in cfg.tisasrec_align.recbole_bundle_path
    assert cfg.data.category in cfg.tisasrec_align.item_potential_ltr_path
    assert cfg.tisasrec_align.stage1_only is False
    assert cfg.tisasrec_align.guardrail_mode == "context_dependent"
    assert cfg.tisasrec_align.fusion_alpha == 0.7
    assert cfg.tisasrec_align.llm_blend_beta == 0.0
    assert cfg.tisasrec_align.llm_gate_enabled is True
    assert cfg.tisasrec_align.rerank_pool_k == 300
    assert cfg.tisasrec_align.llm_card_review_snippets is True
    assert cfg.tisasrec_align.llm_card_review_candidates == 5
    assert cfg.tisasrec_align.llm_rerank_mode == "listwise"
    assert cfg.tisasrec_align.stage2_score == "ltr_llm"
    assert cfg.tisasrec_align.llm_pool_cap == 100
    assert cfg.tisasrec_align.llm_overlap_inject == 0
    assert cfg.tisasrec_align.llm_w_phi == 1.0
    assert cfg.tisasrec_align.llm_w_tu == 0.2
    assert cfg.tisasrec_align.llm_w_co == 0.1
    assert cfg.tisasrec_align.llm_w_llm == 0.25
    assert cfg.tisasrec_align.llm_protect_n == 0
    assert cfg.tisasrec_align.llm_promote_k == 20
    assert cfg.tisasrec_align.llm_reason_then_pick is False
    assert cfg.tisasrec_align.llm_constraint_override is True
    assert cfg.tisasrec_align.llm_hybrid_gate_enabled is False
    assert cfg.tisasrec_align.llm_reason_depth == "deep"
    assert cfg.tisasrec_align.guardrail_m0 == 15
    assert cfg.tisasrec_align.guardrail_m_max == 30
    assert cfg.tisasrec_align.recbole_train_config
    assert cfg.tisasrec_align.hidden_units == 64


def test_legacy_yelp_keeps_era() -> None:
    legacy = load_config("configs/legacy/categories/Yelp_emorecagent_align.yaml")
    assert legacy.tisasrec_align.stage1_backend == "era"
    assert legacy.tisasrec_align.guardrail_mode == "reorder_head"


def test_stage1_only_cli_override() -> None:
    """``--stage1-only`` flips stage1_only without a separate baseline YAML."""
    import importlib.util
    from pathlib import Path

    path = Path("scripts/run_experiment.py")
    spec = importlib.util.spec_from_file_location("run_experiment_cli", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cfg = load_config("configs/categories/Yelp.yaml")
    assert cfg.tisasrec_align.stage1_only is False
    args = SimpleNamespace(
        method="emorecagent_align", stage1_only=True, use_hash_encoder=False
    )
    out = mod.apply_emorecagent_align_cli_overrides(cfg, args)
    assert out.tisasrec_align.stage1_only is True
    assert out.tisasrec_align.use_hash_encoder is False

    args2 = SimpleNamespace(
        method="emorecagent_align", stage1_only=True, use_hash_encoder=True
    )
    out2 = mod.apply_emorecagent_align_cli_overrides(cfg, args2)
    assert out2.tisasrec_align.stage1_only is True
    assert out2.tisasrec_align.use_hash_encoder is True


def test_stage2_ltr_cli_override() -> None:
    """``--stage2-ltr`` sets stage2_score without a separate YAML."""
    import importlib.util
    from pathlib import Path

    path = Path("scripts/run_experiment.py")
    spec = importlib.util.spec_from_file_location("run_experiment_cli", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cfg = load_config("configs/categories/Beauty_and_Personal_Care.yaml")
    assert cfg.tisasrec_align.stage2_score == "ltr_llm"
    args = SimpleNamespace(
        method="emorecagent_align",
        stage1_only=False,
        stage2_ltr=True,
        stage2_ltr_llm=False,
        use_hash_encoder=False,
    )
    out = mod.apply_emorecagent_align_cli_overrides(cfg, args)
    assert out.tisasrec_align.stage2_score == "ltr"
    assert out.tisasrec_align.stage1_only is False


def test_stage2_ltr_llm_cli_override() -> None:
    import importlib.util
    from pathlib import Path

    path = Path("scripts/run_experiment.py")
    spec = importlib.util.spec_from_file_location("run_experiment_cli", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cfg = load_config("configs/default.yaml")
    assert cfg.tisasrec_align.stage2_score == "llm"
    args = SimpleNamespace(
        method="emorecagent_align",
        stage1_only=False,
        stage2_ltr=False,
        stage2_ltr_llm=True,
        use_hash_encoder=False,
    )
    out = mod.apply_emorecagent_align_cli_overrides(cfg, args)
    assert out.tisasrec_align.stage2_score == "ltr_llm"


def test_eval_fit_interactions_uses_history_for_align() -> None:
    import importlib.util
    from pathlib import Path

    path = Path("scripts/run_experiment.py")
    spec = importlib.util.spec_from_file_location("run_experiment_cli", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    train = ["t"]
    history = ["t", "v"]
    assert mod.eval_fit_interactions(
        train, history, method="emorecagent_align"
    ) == history
    assert mod.eval_fit_interactions(train, history, method="popularity") == train
