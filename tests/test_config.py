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
  model: qwen2.5:7b
  temperature: 0.0
  request_timeout_s: 120
  max_retries: 3
"""


def _set_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")


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
    assert cfg.ollama.host == "http://localhost:11434"


def test_env_model_overrides_yaml(tmp_path, monkeypatch):
    _set_env(monkeypatch)
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:14b")
    cfg = load_config(_write(tmp_path, _VALID_YAML))
    assert cfg.llm.model == "qwen2.5:14b"


def test_env_model_small_overrides_yaml(tmp_path, monkeypatch):
    _set_env(monkeypatch)
    monkeypatch.setenv("LLM_MODEL_SMALL", "qwen2.5:3b")
    cfg = load_config(_write(tmp_path, _VALID_YAML))
    assert cfg.llm.model_small == "qwen2.5:3b"
    assert cfg.llm.model == "qwen2.5:7b"


def test_resolve_llm_model_for_absa():
    from emorecagent.config import LlmCfg, resolve_llm_model

    llm = LlmCfg(model="qwen2.5:7b", model_small="qwen2.5:3b")
    assert resolve_llm_model(llm, for_absa=True) == "qwen2.5:3b"
    assert resolve_llm_model(llm, for_absa=False) == "qwen2.5:7b"
    assert resolve_llm_model(LlmCfg(model="qwen2.5:7b"), for_absa=True) == "qwen2.5:7b"


def test_missing_env_raises_clear_error(tmp_path, monkeypatch):
    for var in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "OLLAMA_HOST"):
        monkeypatch.delenv(var, raising=False)
    # Repo .env would repopulate vars; disable for this negative test.
    monkeypatch.setattr("emorecagent.config.load_dotenv", lambda *a, **k: None)
    with pytest.raises(ConfigError, match="NEO4J_URI"):
        load_config(_write(tmp_path, _VALID_YAML))


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


def test_default_config_has_hgt_section(monkeypatch):
    _set_env(monkeypatch)
    cfg = load_config("configs/default.yaml")
    assert cfg.hgt.pool_size == 50
    assert cfg.hgt.text_encoder == "hash"


def test_paper_baseline_config(monkeypatch):
    _set_env(monkeypatch)
    cfg = load_config("configs/paper_baseline.yaml")
    assert cfg.eval.verified_only is False
    assert cfg.eval.protocol == "user_batch"
    assert cfg.eval.aggregation == "user_mean"
    assert cfg.eval.k_values == [10, 20]
    assert cfg.eval.n_negatives == 100


def test_ablation_overlay_extends_default(monkeypatch):
    _set_env(monkeypatch)
    cfg = load_config("configs/ablations/base_cf.yaml")
    # overlay-only keys change; everything else inherits from default.yaml
    assert cfg.experiment.name == "base_cf"
    assert cfg.scoring.alpha == 1.0
    assert cfg.ablation.aspect_term is False
    assert cfg.data.category == "Beauty_and_Personal_Care"  # inherited


def test_no_dynamic_weights_overlay(monkeypatch):
    _set_env(monkeypatch)
    cfg = load_config("configs/ablations/no_dynamic_weights.yaml")
    assert cfg.ablation.dynamic_weights is False
    assert cfg.ablation.reflection is True
    assert cfg.ablation.aspect_term is True
