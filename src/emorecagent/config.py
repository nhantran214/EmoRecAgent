"""Typed configuration loading: YAML experiment config + environment secrets.

Unknown YAML keys are rejected (extra="forbid") so config drift surfaces early.
Connection/secret values come from the environment (.env), never from the YAML.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentCfg(_Strict):
    name: str = "default"
    seed: int = 42
    max_test_rows: int | None = None
    use_llm_cot: bool = True


class DataCfg(_Strict):
    category: str
    review_path: str
    meta_path: str
    out_dir: str
    k_core: int = 10
    max_users: int | None = None
    max_items: int | None = None
    min_history: int = 5
    min_distinct_aspects: int = 2
    # Chronological split ratios (must sum to 1.0; never random).
    split_train_ratio: float = 0.8
    split_valid_ratio: float = 0.1
    split_test_ratio: float = 0.1


class ScoringCfg(_Strict):
    alpha: float = 0.5
    lambda_decay: float = 0.01
    affective_rescaled: bool = True
    helpful_vote_cap: int = 10


class CfCfg(_Strict):
    backend: str = "svd"
    factors: int = 64


class AbsaCfg(_Strict):
    targets_path: str
    cache_path: str
    gold_path: str
    summary_path: str = "results/absa_summary.json"
    preview_html_path: str = "results/absa_preview.html"
    backend: Literal["llm_only", "hybrid"] = "llm_only"
    pipeline_version: str = "hybrid-v1"
    classical_checkpoint: str = "multilingual"
    classical_checkpoint_path: str | None = None
    classical_device: Literal["auto", "cpu", "cuda"] = "auto"
    classical_min_confidence: float = 0.85
    repair_on_gap: bool = True
    min_confidence: float = 0.5
    gold_n_samples: int = 750
    gold_train_only: bool = True
    gold_stratify_length: bool = True
    min_aspect_support: int = 5
    min_sentiment_support: int = 10
    quality_gate_max_f1_drop: float = 0.02


class AgentsCfg(_Strict):
    max_reflection_iters: int = 2
    candidate_pool_size: int = 200
    min_aspect_edges: int = 1
    top_k_aspects: int = 5
    default_budget: float | None = None


class HgtCfg(_Strict):
    graph_path: str = "data/processed/Beauty_and_Personal_Care/hgt/hgt_graph.pt"
    aspect_vocab_path: str = (
        "data/processed/Beauty_and_Personal_Care/hgt/aspect_vocab.json"
    )
    checkpoint_path: str = (
        "data/processed/Beauty_and_Personal_Care/hgt/checkpoint.pt"
    )
    embeddings_dir: str = "data/processed/Beauty_and_Personal_Care/hgt/embeddings"
    aspect_top_k: int = 100
    text_encoder: str = "hash"
    feature_dim: int = 64
    pool_size: int = 50
    n_layers: int = 2
    n_heads: int = 8
    n_hid: int = 256
    dropout: float = 0.2
    use_RTE: bool = True
    epochs: int = 50
    lr: float = 0.001
    batch_size: int = 1024
    neg_samples: int = 1
    early_stop_patience: int = 5
    device: str = "auto"


class EvalCfg(_Strict):
    k_values: list[int] = Field(default_factory=lambda: [5, 10, 20])
    hr_avg_k: list[int] = Field(default_factory=lambda: [1, 3, 5])
    cumulative_history: bool = False
    n_bootstrap: int = 1000
    n_seeds: int = 3
    verified_only: bool = True
    # per_row: one ranking per test interaction (EmoRecAgent ablation default).
    # user_batch: one ranking per user, multi-relevant ground truth (LightGCN-style).
    protocol: Literal["per_row", "user_batch"] = "per_row"
    # row_mean: average over test rows; user_mean: macro-average over users (paper table).
    aggregation: Literal["row_mean", "user_mean"] = "row_mean"
    # Additional sampled eval (1 positive + N negatives) alongside full-catalog pass.
    n_negatives: int | None = 100
    # Extra @K metrics for the sampled pass only (hr/mrr/ndcg/recall).
    sampled_k_values: list[int] = Field(default_factory=lambda: [1, 3, 5])


class LlmCfg(_Strict):
    model: str = "qwen2.5:7b"
    # Offline ABSA (validate/repair); falls back to `model` when unset.
    model_small: str | None = None
    temperature: float = 0.0
    request_timeout_s: int = 120
    max_retries: int = 3


def resolve_llm_model(llm: LlmCfg, *, for_absa: bool = False) -> str:
    """Pick agent model vs ABSA model (``LLM_MODEL`` / ``LLM_MODEL_SMALL``)."""
    if for_absa and llm.model_small:
        return llm.model_small
    return llm.model


class AblationCfg(_Strict):
    """Factorial ablation toggles. Full system = all enabled.

    - reflection: run the Reflection Agent loop (off = single forward pass)
    - dynamic_weights: time-decayed w_u(a,t) (off = static aspect weights)
    - aspect_term: include the affective term (off = base CF only, i.e. alpha=1)
    """

    reflection: bool = True
    dynamic_weights: bool = True
    aspect_term: bool = True


class Neo4jCfg(_Strict):
    uri: str
    user: str
    password: str


class OllamaCfg(_Strict):
    host: str


class Config(_Strict):
    experiment: ExperimentCfg
    data: DataCfg
    scoring: ScoringCfg
    cf: CfCfg
    absa: AbsaCfg
    agents: AgentsCfg
    eval: EvalCfg
    llm: LlmCfg
    ablation: AblationCfg = Field(default_factory=AblationCfg)
    hgt: HgtCfg = Field(default_factory=HgtCfg)
    neo4j: Neo4jCfg
    ollama: OllamaCfg


class ConfigError(RuntimeError):
    """Raised when configuration is missing required values or is malformed."""


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively overlay `overlay` onto `base` (overlay wins on conflicts)."""
    out = dict(base)
    for key, val in overlay.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ConfigError(
            f"Required environment variable '{name}' is not set. "
            f"Copy .env.example to .env and fill it in."
        )
    return value


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    """Load and validate the experiment config plus environment-derived settings.

    Raises ConfigError on missing env vars; raises pydantic ValidationError on
    unknown/malformed YAML keys.
    """
    load_dotenv()

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise ConfigError(f"Config file not found: {cfg_path}")

    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    # Overlay configs (e.g. ablation grids) declare `extends: <relative path>`
    # and override only the keys that differ from the base config.
    extends = raw.pop("extends", None)
    if extends:
        base_path = (cfg_path.parent / extends).resolve()
        if not base_path.exists():
            raise ConfigError(f"Base config referenced by 'extends' not found: {base_path}")
        base_raw = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
        base_raw.pop("extends", None)
        raw = _deep_merge(base_raw, raw)

    # Environment overlays for LLM model names (.env is source of truth).
    if isinstance(raw.get("llm"), dict):
        env_model = os.environ.get("LLM_MODEL")
        if env_model:
            raw["llm"]["model"] = env_model
        env_model_small = os.environ.get("LLM_MODEL_SMALL")
        if env_model_small:
            raw["llm"]["model_small"] = env_model_small

    raw["neo4j"] = {
        "uri": _require_env("NEO4J_URI"),
        "user": _require_env("NEO4J_USER"),
        "password": _require_env("NEO4J_PASSWORD"),
    }
    raw["ollama"] = {"host": _require_env("OLLAMA_HOST")}

    return Config(**raw)
