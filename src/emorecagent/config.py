"""Typed configuration loading: YAML experiment config + environment secrets.

Unknown YAML keys are rejected (extra="forbid") so config drift surfaces early.
Connection/secret values come from the environment (.env), never from the YAML.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentCfg(_Strict):
    name: str = "default"
    seed: int = 42


class DataCfg(_Strict):
    category: str
    review_path: str
    meta_path: str
    out_dir: str
    k_core: int = 5
    max_users: int | None = None
    max_items: int | None = None
    min_history: int = 5
    min_distinct_aspects: int = 2


class ScoringCfg(_Strict):
    alpha: float = 0.5
    lambda_decay: float = 0.01
    affective_rescaled: bool = True
    helpful_vote_cap: int = 10


class CfCfg(_Strict):
    backend: str = "svd"
    factors: int = 64


class AbsaCfg(_Strict):
    cache_path: str
    gold_path: str
    min_confidence: float = 0.5


class AgentsCfg(_Strict):
    max_reflection_iters: int = 2
    candidate_pool_size: int = 200
    min_aspect_edges: int = 1
    top_k_aspects: int = 5
    default_budget: float | None = None


class EvalCfg(_Strict):
    k_values: list[int] = Field(default_factory=lambda: [5, 10, 20])
    n_bootstrap: int = 1000
    n_seeds: int = 3


class LlmCfg(_Strict):
    model: str = "qwen2.5:7b"
    temperature: float = 0.0
    request_timeout_s: int = 120
    max_retries: int = 3


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
    neo4j: Neo4jCfg
    ollama: OllamaCfg


class ConfigError(RuntimeError):
    """Raised when configuration is missing required values or is malformed."""


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

    # Environment overlays for the LLM model name (keeps one source of truth).
    env_model = os.environ.get("LLM_MODEL")
    if env_model and isinstance(raw.get("llm"), dict):
        raw["llm"]["model"] = env_model

    raw["neo4j"] = {
        "uri": _require_env("NEO4J_URI"),
        "user": _require_env("NEO4J_USER"),
        "password": _require_env("NEO4J_PASSWORD"),
    }
    raw["ollama"] = {"host": _require_env("OLLAMA_HOST")}

    return Config(**raw)
