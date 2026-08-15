"""Tests for config loading."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from emorecagent.config import ConfigError, load_config

_VALID_YAML = """
experiment:
  name: test
  seed: 7
data:
  category: Beauty_and_Personal_Care
  review_path: r.jsonl
  meta_path: m.jsonl
  out_dir: out
  k_core: 5
  max_users: 100
  max_items: 200
  min_history: 5
  min_distinct_aspects: 2
scoring:
  alpha: 0.5
  lambda_decay: 0.01
  affective_rescaled: true
  helpful_vote_cap: 10
cf:
  backend: svd
  factors: 64
absa:
  targets_path: targets.jsonl
  cache_path: c.sqlite
  gold_path: g.jsonl
  backend: llm_only
  pipeline_version: hybrid-v1
  classical_checkpoint: multilingual
  classical_checkpoint_path: null
  classical_device: auto
  classical_min_confidence: 0.85
  repair_on_gap: true
  quality_gate_max_f1_drop: 0.02
  min_confidence: 0.5
agents:
  max_reflection_iters: 2
  candidate_pool_size: 200
  min_aspect_edges: 1
  top_k_aspects: 5
  default_budget: null
eval:
  k_values: [5, 10, 20]
  n_bootstrap: 1000
  n_seeds: 3
llm:
  model: Qwen/Qwen2.5-7B-Instruct
  temperature: 0.0
  request_timeout_s: 120
  max_retries: 3
tgi:
  base_url: http://localhost:8080
"""


def _set_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("emorecagent.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    monkeypatch.setenv("TGI_BASE_URL", "http://localhost:8080")
    # Repo .env may set LLM_MODEL*; tests assert yaml/env overlay behavior explicitly.
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_MODEL_SMALL", raising=False)


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def test_loads_valid_config_and_overlays_env(tmp_path, monkeypatch):
    _set_env(monkeypatch)
    cfg = load_config(_write(tmp_path, _VALID_YAML))
    assert cfg.data.category == "Beauty_and_Personal_Care"
    assert cfg.scoring.alpha == 0.5
    assert cfg.neo4j.uri == "bolt://localhost:7687"
    assert cfg.tgi.base_url == "http://localhost:8080"


def test_env_model_overrides_yaml(tmp_path, monkeypatch):
    _set_env(monkeypatch)
    monkeypatch.setenv("LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct")
    cfg = load_config(_write(tmp_path, _VALID_YAML))
    assert cfg.llm.model == "Qwen/Qwen2.5-14B-Instruct"


def test_env_model_small_overrides_yaml(tmp_path, monkeypatch):
    _set_env(monkeypatch)
    monkeypatch.setenv("LLM_MODEL_SMALL", "Qwen/Qwen2.5-3B-Instruct")
    cfg = load_config(_write(tmp_path, _VALID_YAML))
    assert cfg.llm.model_small == "Qwen/Qwen2.5-3B-Instruct"
    assert cfg.llm.model == "Qwen/Qwen2.5-7B-Instruct"


def test_resolve_llm_model_for_absa():
    from emorecagent.config import LlmCfg, resolve_llm_model

    llm = LlmCfg(
        model="Qwen/Qwen2.5-7B-Instruct",
        model_small="Qwen/Qwen2.5-3B-Instruct",
    )
    assert resolve_llm_model(llm, for_absa=True) == "Qwen/Qwen2.5-3B-Instruct"
    assert resolve_llm_model(llm, for_absa=False) == "Qwen/Qwen2.5-7B-Instruct"
    assert (
        resolve_llm_model(
            LlmCfg(model="Qwen/Qwen2.5-7B-Instruct", model_small=None),
            for_absa=True,
        )
        == "Qwen/Qwen2.5-7B-Instruct"
    )


def test_missing_env_raises_clear_error(tmp_path, monkeypatch):
    for var in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("TGI_BASE_URL", raising=False)
    # Repo .env would repopulate vars; disable for this negative test.
    monkeypatch.setattr("emorecagent.config.load_dotenv", lambda *a, **k: None)
    with pytest.raises(ConfigError, match="NEO4J_URI"):
        load_config(_write(tmp_path, _VALID_YAML))


def test_missing_tgi_base_url_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    monkeypatch.delenv("TGI_BASE_URL", raising=False)
    monkeypatch.setattr("emorecagent.config.load_dotenv", lambda *a, **k: None)
    yaml = _VALID_YAML.replace(
        "tgi:\n  base_url: http://localhost:8080\n",
        "",
    )
    with pytest.raises(ConfigError, match="TGI base URL"):
        load_config(_write(tmp_path, yaml))


def test_tgi_env_overlays(tmp_path, monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    monkeypatch.setenv("TGI_BASE_URL", "http://gpu:9000")
    monkeypatch.setenv("TGI_BASE_URL_SMALL", "http://gpu:9001")
    monkeypatch.setenv("TGI_API_KEY", "secret-key")
    monkeypatch.setattr("emorecagent.config.load_dotenv", lambda *a, **k: None)
    cfg = load_config(_write(tmp_path, _VALID_YAML))
    assert cfg.tgi.base_url == "http://gpu:9000"
    assert cfg.tgi.base_url_small == "http://gpu:9001"
    assert cfg.tgi.api_key == "secret-key"


def test_resolve_tgi_base_url_small_endpoint():
    from emorecagent.config import LlmCfg, TgiCfg, resolve_tgi_base_url

    llm = LlmCfg(model="big", model_small="small")
    tgi = TgiCfg(
        base_url="http://localhost:8080",
        base_url_small="http://localhost:8081",
    )
    assert resolve_tgi_base_url(tgi, llm, for_absa=True) == "http://localhost:8081"
    assert resolve_tgi_base_url(tgi, llm, for_absa=False) == "http://localhost:8080"


def test_unknown_key_is_rejected(tmp_path, monkeypatch):
    _set_env(monkeypatch)
    bad = _VALID_YAML + "\nbogus_section:\n  foo: 1\n"
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, bad))


def test_missing_file_raises(monkeypatch):
    _set_env(monkeypatch)
    with pytest.raises(ConfigError, match="not found"):
        load_config("does/not/exist.yaml")


def test_default_config_has_full_ablation(monkeypatch):
    _set_env(monkeypatch)
    cfg = load_config("configs/default.yaml")
    assert cfg.ablation.reflection
    assert cfg.ablation.dynamic_weights
    assert cfg.ablation.aspect_term


def test_default_config_has_tisasrec_align_section(monkeypatch):
    _set_env(monkeypatch)
    cfg = load_config("configs/default.yaml")
    assert cfg.tisasrec_align.fusion_alpha == 0.7
    assert cfg.tisasrec_align.early_stop_metric == "valid_link_hr@10"
    assert cfg.tisasrec_align.tu_mode == "cache"
    assert cfg.tisasrec_align.valid_eval_all is True
    assert cfg.tisasrec_align.stage1_loss == "bce"
    assert cfg.tisasrec_align.stage2_mode == "fusion"
    assert cfg.tisasrec_align.stage2_score == "llm"
    assert cfg.tisasrec_align.rerank_pool_k == 100
    assert cfg.tisasrec_align.num_train_negatives == 1
    assert cfg.tisasrec_align.batch_size == 512
    assert cfg.tisasrec_align.alignment_batch_size == 2048
    # YAML ``steps_per_epoch: auto`` coerces to None (compute at train time).
    assert cfg.tisasrec_align.steps_per_epoch is None
    assert cfg.tisasrec_align.lr_scheduler_enabled is True


def test_default_config_batch_eval_defaults(monkeypatch):
    _set_env(monkeypatch)
    cfg = load_config("configs/default.yaml")
    assert cfg.eval.llm_batch is False
    assert cfg.eval.batch_size == 12
    assert cfg.eval.batch_token_budget == 28_000


def test_paper_baseline_config(monkeypatch):
    _set_env(monkeypatch)
    cfg = load_config("configs/legacy/paper_baseline.yaml")
    assert cfg.eval.verified_only is False
    assert cfg.eval.protocol == "user_batch"
    assert cfg.eval.aggregation == "user_mean"
    assert cfg.eval.k_values == [10, 20]
    assert cfg.eval.n_negatives == 100


def test_emorecagent_align_config(monkeypatch):
    _set_env(monkeypatch)
    cfg = load_config("configs/legacy/emorecagent_align.yaml")
    assert cfg.eval.protocol == "user_batch"
    assert cfg.eval.aggregation == "user_mean"
    assert cfg.tisasrec_align.use_hash_encoder is True
    assert cfg.tisasrec_align.stage1_only is False
    assert cfg.tisasrec_align.stage2_mode == "rerank"
    assert cfg.tisasrec_align.rerank_pool_k == 100
    assert cfg.tisasrec_align.llm_pool_cap == 40


def test_emorecagent_stage1_baseline_config(monkeypatch):
    _set_env(monkeypatch)
    cfg = load_config("configs/legacy/emorecagent_stage1_baseline.yaml")
    assert cfg.tisasrec_align.stage1_only is True
    assert cfg.eval.protocol == "user_batch"


def test_beauty_category(monkeypatch):
    _set_env(monkeypatch)
    cfg = load_config("configs/categories/Beauty_and_Personal_Care.yaml")
    assert cfg.data.category == "Beauty_and_Personal_Care"
    assert cfg.tisasrec_align.stage1_backend == "recbole"
    assert "tisasrec_option_b" in cfg.tisasrec_align.stage1_checkpoint_path
    assert cfg.tisasrec_align.hidden_units == 64
    assert cfg.tisasrec_align.guardrail_mode == "context_dependent"
    assert cfg.tisasrec_align.llm_rerank_mode == "listwise"
    assert cfg.tisasrec_align.fusion_alpha == 0.7
    assert cfg.tisasrec_align.llm_blend_beta == 0.0
    assert cfg.tisasrec_align.llm_gate_enabled is True
    assert cfg.tisasrec_align.rerank_pool_k == 300
    assert cfg.tisasrec_align.llm_pool_cap == 100
    assert cfg.tisasrec_align.llm_overlap_inject == 0
    assert cfg.tisasrec_align.llm_w_phi == 1.0
    assert cfg.tisasrec_align.llm_w_tu == 0.2
    assert cfg.tisasrec_align.llm_w_co == 0.1
    assert cfg.tisasrec_align.llm_w_llm == 0.25
    assert cfg.tisasrec_align.llm_promote_k == 20
    assert cfg.tisasrec_align.llm_protect_n == 0
    assert cfg.tisasrec_align.llm_card_review_snippets is True
    assert cfg.tisasrec_align.llm_card_review_candidates == 5
    assert cfg.tisasrec_align.llm_narrow_cap == 0
    assert cfg.tisasrec_align.llm_reason_then_pick is False
    assert cfg.tisasrec_align.llm_reason_depth == "deep"
    assert cfg.tisasrec_align.stage2_score == "ltr_llm"
    assert cfg.tisasrec_align.llm_constraint_override is True
    assert cfg.tisasrec_align.llm_hybrid_gate_enabled is False
    assert "Beauty_and_Personal_Care" in cfg.tisasrec_align.item_potential_ltr_path
    assert cfg.tisasrec_align.llm_lexical_first_enabled is False
    assert cfg.tisasrec_align.llm_lexical_first_rank_lo == 11
    assert cfg.tisasrec_align.llm_lexical_first_rank_hi == 20
    assert cfg.tisasrec_align.llm_lexical_first_overlap_delta == 1
    # Eq. 21 window relaxed vs §IV.D (smaller N_u, larger M_u).
    assert cfg.tisasrec_align.guardrail_n0 == 3
    assert cfg.tisasrec_align.guardrail_m0 == 15
    assert cfg.tisasrec_align.guardrail_m_max == 30
    assert cfg.eval.k_values == [10, 20, 50, 100]
    # Stage-1 hparams match the already-trained RecBole CE bundle.
    assert cfg.tisasrec_align.num_heads == 2
    assert cfg.tisasrec_align.dropout_rate == 0.5
    assert cfg.tisasrec_align.stage1_only is False


def test_sports_category_overlay(monkeypatch):
    _set_env(monkeypatch)
    cfg = load_config("configs/categories/Sports_and_Outdoors.yaml")
    assert cfg.data.category == "Sports_and_Outdoors"
    assert cfg.data.out_dir == "data/processed/Sports_and_Outdoors"
    assert "Sports_and_Outdoors.jsonl" in cfg.data.review_path
    assert "meta_Sports_and_Outdoors.jsonl" in cfg.data.meta_path
    assert cfg.absa.cache_path.endswith("Sports_and_Outdoors/absa_cache.sqlite")
    assert "Sports_and_Outdoors/tisasrec_option_b" in cfg.tisasrec_align.stage1_checkpoint_path
    assert cfg.data.k_core == 10
    assert cfg.tisasrec_align.hidden_units == 64
    assert cfg.tisasrec_align.stage1_backend == "recbole"
    assert cfg.tisasrec_align.guardrail_mode == "context_dependent"
    assert cfg.tisasrec_align.llm_rerank_mode == "listwise"
    assert cfg.tisasrec_align.stage2_score == "ltr_llm"
    assert cfg.tisasrec_align.llm_gate_enabled is True
    assert cfg.tisasrec_align.rerank_pool_k == 300
    assert cfg.tisasrec_align.llm_pool_cap == 100
    assert "Sports_and_Outdoors" in cfg.tisasrec_align.item_potential_ltr_path
    assert cfg.eval.k_values == [10, 20, 50, 100]


def test_sports_legacy_align_overlay(monkeypatch):
    _set_env(monkeypatch)
    cfg = load_config(
        "configs/legacy/categories/Sports_and_Outdoors_emorecagent_align.yaml"
    )
    assert cfg.data.category == "Sports_and_Outdoors"
    assert cfg.eval.protocol == "user_batch"
    assert cfg.eval.aggregation == "user_mean"
    assert cfg.tisasrec_align.stage2_mode == "rerank"
    assert cfg.tisasrec_align.use_hash_encoder is True
    assert cfg.tisasrec_align.stage1_only is False
    assert cfg.tisasrec_align.stage1_backend == "era"
    assert cfg.tisasrec_align.rerank_pool_k == 100
    assert cfg.tisasrec_align.llm_pool_cap == 40


def test_sports_legacy_stage1_baseline_overlay(monkeypatch):
    _set_env(monkeypatch)
    cfg = load_config(
        "configs/legacy/categories/Sports_and_Outdoors_emorecagent_stage1_baseline.yaml"
    )
    assert cfg.data.category == "Sports_and_Outdoors"
    assert cfg.tisasrec_align.stage1_only is True
    assert cfg.eval.protocol == "user_batch"
    beauty = load_config("configs/legacy/emorecagent_stage1_baseline.yaml")
    assert cfg.tisasrec_align.stage2_mode == beauty.tisasrec_align.stage2_mode
    assert cfg.tisasrec_align.hidden_units == beauty.tisasrec_align.hidden_units


def test_sports_paths_do_not_touch_beauty(monkeypatch):
    _set_env(monkeypatch)
    from emorecagent.config import (
        category_scoped_artifact_paths,
        validate_category_path_isolation,
    )

    sports = load_config("configs/categories/Sports_and_Outdoors.yaml")
    beauty = load_config("configs/categories/Beauty_and_Personal_Care.yaml")
    assert validate_category_path_isolation(sports) == []
    assert validate_category_path_isolation(beauty) == []

    sports_paths = {p for _, p in category_scoped_artifact_paths(sports)}
    beauty_paths = {p for _, p in category_scoped_artifact_paths(beauty)}
    assert sports_paths.isdisjoint(beauty_paths)
    for p in sports_paths:
        assert "Beauty_and_Personal_Care" not in p
    for p in beauty_paths:
        assert "Sports_and_Outdoors" not in p


def test_toys_category_overlay(monkeypatch):
    _set_env(monkeypatch)
    cfg = load_config("configs/categories/Toys_and_Games.yaml")
    assert cfg.data.category == "Toys_and_Games"
    assert cfg.data.out_dir == "data/processed/Toys_and_Games"
    assert "Toys_and_Games.jsonl" in cfg.data.review_path
    assert "meta_Toys_and_Games.jsonl" in cfg.data.meta_path
    assert cfg.absa.cache_path.endswith("Toys_and_Games/absa_cache.sqlite")
    assert "Toys_and_Games/tisasrec_option_b" in cfg.tisasrec_align.stage1_checkpoint_path
    assert cfg.data.k_core == 10
    assert cfg.tisasrec_align.hidden_units == 64
    assert cfg.tisasrec_align.stage1_backend == "recbole"
    assert cfg.tisasrec_align.llm_rerank_mode == "listwise"
    assert cfg.tisasrec_align.stage2_score == "ltr_llm"
    assert cfg.tisasrec_align.rerank_pool_k == 300
    assert cfg.tisasrec_align.llm_pool_cap == 100
    assert "Toys_and_Games" in cfg.tisasrec_align.item_potential_ltr_path
    assert cfg.eval.k_values == [10, 20, 50, 100]


def test_toys_legacy_align_overlay(monkeypatch):
    _set_env(monkeypatch)
    cfg = load_config(
        "configs/legacy/categories/Toys_and_Games_emorecagent_align.yaml"
    )
    assert cfg.data.category == "Toys_and_Games"
    assert cfg.eval.protocol == "user_batch"
    assert cfg.eval.aggregation == "user_mean"
    assert cfg.tisasrec_align.stage2_mode == "rerank"
    assert cfg.tisasrec_align.use_hash_encoder is True
    assert cfg.tisasrec_align.stage1_only is False
    assert cfg.tisasrec_align.stage1_backend == "era"
    assert cfg.tisasrec_align.rerank_pool_k == 100
    assert cfg.tisasrec_align.llm_pool_cap == 40


def test_toys_legacy_stage1_baseline_overlay(monkeypatch):
    _set_env(monkeypatch)
    cfg = load_config(
        "configs/legacy/categories/Toys_and_Games_emorecagent_stage1_baseline.yaml"
    )
    assert cfg.data.category == "Toys_and_Games"
    assert cfg.tisasrec_align.stage1_only is True
    assert cfg.eval.protocol == "user_batch"
    beauty = load_config("configs/legacy/emorecagent_stage1_baseline.yaml")
    assert cfg.tisasrec_align.stage2_mode == beauty.tisasrec_align.stage2_mode
    assert cfg.tisasrec_align.hidden_units == beauty.tisasrec_align.hidden_units


def test_toys_paths_do_not_touch_beauty_or_sports(monkeypatch):
    _set_env(monkeypatch)
    from emorecagent.config import (
        category_scoped_artifact_paths,
        validate_category_path_isolation,
    )

    toys = load_config("configs/categories/Toys_and_Games.yaml")
    sports = load_config("configs/categories/Sports_and_Outdoors.yaml")
    beauty = load_config("configs/categories/Beauty_and_Personal_Care.yaml")
    assert validate_category_path_isolation(toys) == []

    toys_paths = {p for _, p in category_scoped_artifact_paths(toys)}
    sports_paths = {p for _, p in category_scoped_artifact_paths(sports)}
    beauty_paths = {p for _, p in category_scoped_artifact_paths(beauty)}
    assert toys_paths.isdisjoint(beauty_paths)
    assert toys_paths.isdisjoint(sports_paths)
    for p in toys_paths:
        assert "Beauty_and_Personal_Care" not in p
        assert "Sports_and_Outdoors" not in p


def test_yelp_category_overlay(monkeypatch):
    _set_env(monkeypatch)
    cfg = load_config("configs/categories/Yelp.yaml")
    assert cfg.data.category == "Yelp"
    assert cfg.data.out_dir == "data/processed/Yelp"
    assert cfg.data.review_path.endswith("yelp_dataset") or "yelp_dataset" in cfg.data.review_path
    assert "yelp_dataset" in cfg.data.meta_path
    assert cfg.absa.cache_path.endswith("Yelp/absa_cache.sqlite")
    assert "Yelp/tisasrec_option_b" in cfg.tisasrec_align.stage1_checkpoint_path
    assert cfg.data.k_core == 20
    assert cfg.tisasrec_align.hidden_units == 64
    assert cfg.tisasrec_align.stage1_backend == "recbole"
    assert cfg.tisasrec_align.guardrail_mode == "context_dependent"
    assert cfg.tisasrec_align.llm_rerank_mode == "listwise"
    assert cfg.tisasrec_align.stage2_score == "ltr_llm"
    assert cfg.tisasrec_align.llm_gate_enabled is True
    assert cfg.tisasrec_align.rerank_pool_k == 300
    assert cfg.tisasrec_align.llm_pool_cap == 100
    assert cfg.tisasrec_align.item_potential_ltr_path.endswith(
        "Yelp/tisasrec_option_b/item_potential_ltr.npz"
    )
    assert cfg.eval.k_values == [10, 20, 50, 100]


def test_yelp_legacy_align_overlay(monkeypatch):
    _set_env(monkeypatch)
    cfg = load_config("configs/legacy/categories/Yelp_emorecagent_align.yaml")
    assert cfg.data.category == "Yelp"
    assert cfg.eval.protocol == "user_batch"
    assert cfg.eval.aggregation == "user_mean"
    assert cfg.tisasrec_align.stage2_mode == "rerank"
    assert cfg.tisasrec_align.use_hash_encoder is True
    assert cfg.tisasrec_align.stage1_only is False
    assert cfg.tisasrec_align.stage1_backend == "era"
    beauty = load_config("configs/legacy/emorecagent_align.yaml")
    assert cfg.tisasrec_align.rerank_pool_k == beauty.tisasrec_align.rerank_pool_k
    assert cfg.tisasrec_align.llm_pool_cap == beauty.tisasrec_align.llm_pool_cap


def test_yelp_paths_isolated_from_amazon(monkeypatch):
    _set_env(monkeypatch)
    from emorecagent.config import (
        category_scoped_artifact_paths,
        validate_category_path_isolation,
    )

    yelp = load_config("configs/categories/Yelp.yaml")
    beauty = load_config("configs/categories/Beauty_and_Personal_Care.yaml")
    assert validate_category_path_isolation(yelp) == []
    yelp_paths = {p for _, p in category_scoped_artifact_paths(yelp)}
    beauty_paths = {p for _, p in category_scoped_artifact_paths(beauty)}
    assert yelp_paths.isdisjoint(beauty_paths)
    for p in yelp_paths:
        assert "Beauty_and_Personal_Care" not in p
        assert "amazon-reviews-2023" not in p

def test_recursive_extends(monkeypatch, tmp_path):
    _set_env(monkeypatch)
    base = tmp_path / "base.yaml"
    mid = tmp_path / "mid.yaml"
    top = tmp_path / "top.yaml"
    base.write_text(
        textwrap.dedent(
            """
            experiment: {name: base, seed: 1}
            data:
              category: X
              review_path: r
              meta_path: m
              out_dir: o
              k_core: 5
              max_users: 10
              max_items: 20
              min_history: 5
              min_distinct_aspects: 2
            scoring: {alpha: 0.5, lambda_decay: 0.01, affective_rescaled: true, helpful_vote_cap: 10}
            cf: {backend: svd, factors: 64}
            absa:
              targets_path: t
              cache_path: c
              gold_path: g
              backend: llm_only
              pipeline_version: hybrid-v1
              classical_checkpoint: multilingual
              classical_checkpoint_path: null
              classical_device: auto
              classical_min_confidence: 0.85
              repair_on_gap: true
              quality_gate_max_f1_drop: 0.02
              min_confidence: 0.5
            agents:
              max_reflection_iters: 2
              candidate_pool_size: 200
              min_aspect_edges: 1
              top_k_aspects: 5
              default_budget: null
            eval: {k_values: [5], n_bootstrap: 10, n_seeds: 1}
            llm: {model: m, temperature: 0.0, request_timeout_s: 1, max_retries: 1}
            tgi: {base_url: http://localhost:8080}
            """
        ),
        encoding="utf-8",
    )
    mid.write_text(
        "extends: base.yaml\ndata:\n  category: Y\n  out_dir: oy\n",
        encoding="utf-8",
    )
    top.write_text(
        "extends: mid.yaml\nexperiment:\n  name: top\n",
        encoding="utf-8",
    )
    cfg = load_config(top)
    assert cfg.experiment.name == "top"
    assert cfg.data.category == "Y"
    assert cfg.data.out_dir == "oy"
    assert cfg.data.k_core == 5

