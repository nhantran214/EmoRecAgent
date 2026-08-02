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

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    # ``chronological_ratio`` = per-user 80/10/10; ``leave_last_out`` = LOO.
    split_method: Literal["chronological_ratio", "leave_last_out"] = (
        "chronological_ratio"
    )
    # Optional RecBole ``*.inter`` source (ID-only). When set, build_dataset
    # loads this instead of streaming review JSONL from ``review_path``.
    inter_path: str | None = None
    # Inclusive unix-second bounds applied to raw RecBole / interaction times
    # before k-core (AC-TSR Yelp closed-2019: 1546264800–1577714400).
    min_timestamp_s: int | None = None
    max_timestamp_s: int | None = None
    # Collapse duplicate (user, item) to the earliest timestamp. Amazon review
    # tracks keep this True. RecBole / AC-TSR Yelp keeps multi-visit reviews
    # (``review_id`` rows) so set False for paper-parity cohorts.
    dedup_user_item: bool = True


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
    # When False, dataset build skips ABSA target export and prereq checks
    # skip the ABSA cache (ID-only / no-review tracks).
    enabled: bool = True
    backend: Literal["llm_only", "hybrid", "classical"] = "llm_only"
    pipeline_version: str = "hybrid-v1"
    classical_checkpoint: str = "multilingual"
    classical_checkpoint_path: str | None = None
    classical_device: Literal["auto", "cpu", "cuda"] = "auto"
    classical_batch_size: int = Field(default=32, ge=1, le=256)
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
    llm_rank_prefix: int = 20
    aspect_recall_tau: float = 0.65
    aspect_recall_max: int = 15
    ranking_num_predict: int = 4096


class TiSASRecAlignCfg(_Strict):
    stage1_checkpoint_path: str = (
        "data/processed/Beauty_and_Personal_Care/tisasrec_align/stage1_checkpoint.pt"
    )
    e_i_matrix_path: str = (
        "data/processed/Beauty_and_Personal_Care/tisasrec_align/e_i_matrix.pt"
    )
    alignment_checkpoint_path: str = (
        "data/processed/Beauty_and_Personal_Care/tisasrec_align/alignment_mlp.pt"
    )
    tu_cache_path: str = (
        "data/processed/Beauty_and_Personal_Care/tisasrec_align/tu_cache.jsonl"
    )
    hidden_units: int = 64
    maxlen: int = 50
    num_blocks: int = 2
    num_heads: int = 1
    dropout_rate: float = 0.2
    l2_emb: float = 1e-4
    time_span: int = 256
    # RecBole FFN width. ``None`` keeps Kang hidden→hidden (Amazon default).
    # Paper Yelp_AC sets 256 to match RecBole ``inner_size``.
    inner_size: int | None = None
    # Relative-time unit for the TiSASRec interval matrix. ``None`` keeps the
    # per-user min-gap rule (Amazon default). Yelp needs a fixed unit (86400 =
    # days): its min-gap is sub-hour for most users, which overflows
    # ``time_span`` and makes the shared interval embedding user-dependent.
    time_unit_seconds: int | None = None
    fusion_alpha: float = 0.7
    text_encoder_dim: int = 768
    tu_mode: Literal["cache", "live"] = "cache"
    use_hash_encoder: bool = False
    stage1_only: bool = False
    stage2_mode: Literal["fusion", "rerank"] = "fusion"
    rerank_pool_k: int = 100
    llm_pool_cap: int = 40
    cross_user_boost: float = 0.05
    cross_user_lookup_path: str = (
        "data/processed/Beauty_and_Personal_Care/tisasrec_align/cross_user_lookup.json"
    )
    guardrail_top_n: int = 5
    guardrail_max_drop_rank: int = 10
    # Stage-2 merge policy. "position": legacy full-revert guardrail (any Stage-1
    # top-N item dropping past max_drop_rank reverts the whole ranking).
    # "reorder_head": constrain the LLM to permute only Stage-1's top
    # `reorder_head_n` items (membership at k >= reorder_head_n preserved by
    # construction, so hr@k/recall@k never regress vs Stage-1). "off": keep the
    # merged ranking unconditionally.
    guardrail_mode: Literal["position", "reorder_head", "off"] = "position"
    reorder_head_n: int = 10
    # LOO / Stage-1 test history: ``train`` = train-only (legacy); ``train_valid``
    # = train+valid prefix when ranking the held-out test item (RecBole LOO).
    test_history: Literal["train", "train_valid"] = "train"
    # Stage-1 backbone. ``era`` = in-repo TiSASRec (Amazon / Yelp-review default).
    # ``recbole`` = AC-TSR RecBole TiSASRec bundle — **Yelp_AC only**.
    stage1_backend: Literal["era", "recbole"] = "era"
    recbole_bundle_path: str = (
        "data/processed/Yelp_AC/tisasrec_paper/recbole_stage1_bundle.pt"
    )
    recbole_vendor_root: str = "baseline/RecBole-TiSASRec/vendor"
    # Stage-2 preference text source. ``absa`` = review/ABSA pipeline (default);
    # ``item_metadata`` = RecBole ``.item`` categories/names (Yelp_AC).
    preference_source: Literal["absa", "item_metadata"] = "absa"
    # Cross-user lookup build mode. ``review_text`` requires non-empty reviews;
    # ``id_only`` builds co-visit counts from train sequences.
    cross_user_mode: Literal["review_text", "id_only"] = "review_text"
    infonce_tau_grid: list[float] = Field(default_factory=lambda: [0.05, 0.07, 0.1])
    alignment_activation: Literal["elu", "gelu"] = "elu"
    stage1_epochs: int = 1000
    stage2_epochs: int = 10
    lr: float = 0.001
    alignment_lr: float = 0.001
    batch_size: int = 2048
    alignment_batch_size: int = 64
    steps_per_epoch: int | None = None
    early_stop_patience: int = 10
    early_stop_metric: str = "valid_link_hr@10"
    valid_mask_train_seen: bool = True
    require_valid: bool = True
    valid_eval_all: bool = True
    device: str = "auto"
    valid_eval_max_pairs: int = 2048
    valid_eval_batch_size: int = 64
    pool_size: int = 50
    lr_scheduler_enabled: bool = True
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    min_lr: float = 1e-5
    optimizer: Literal["adam", "adamw"] = "adam"
    weight_decay: float = 0.0
    # ``ce`` = RecBole / AC-TSR full-catalog cross-entropy (paper TiSASRec).
    # ``bce`` / ``multi_bce`` / ``bpr`` = sampled pairwise (Amazon EmoRecAgent track).
    stage1_loss: Literal["bce", "multi_bce", "bpr", "ce"] = "bce"
    num_train_negatives: int = 1
    lambda_decay: float = 0.01
    top_k_aspects: int = 5

    @field_validator("steps_per_epoch", mode="before")
    @classmethod
    def _coerce_steps_per_epoch(cls, value: object) -> int | None:
        if value is None or value == "auto":
            return None
        return int(value)  # type: ignore[arg-type]


class HetTiSASRecCfg(_Strict):
    aspect_vocab_path: str = (
        "data/processed/Beauty_and_Personal_Care/hettisasrec/aspect_vocab.json"
    )
    aspect_graph_path: str = (
        "data/processed/Beauty_and_Personal_Care/hettisasrec/aspect_graph.pt"
    )
    checkpoint_path: str = (
        "data/processed/Beauty_and_Personal_Care/hettisasrec/checkpoint.pt"
    )
    aspect_top_k: int = 100
    pool_size: int = 50
    hidden_units: int = 64
    maxlen: int = 50
    num_blocks: int = 2
    num_heads: int = 1
    dropout_rate: float = 0.2
    l2_emb: float = 1e-4
    time_span: int = 256
    use_aspect_enrichment: bool = True
    aspect_mp_layers: int = 1
    aspect_loss_weight: float = 0.2
    epochs: int = 50
    lr: float = 0.001
    batch_size: int = 256
    steps_per_epoch: int = 800
    early_stop_patience: int = 10
    early_stop_metric: str = "valid_pool_recall@50"
    valid_mask_train_seen: bool = True
    require_valid: bool = True
    device: str = "auto"
    valid_eval_max_pairs: int = 2048
    valid_eval_batch_size: int = 64


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
    parallel_workers: int = 1
    # Batched LLM eval: group per-row ranking tasks into one reasoning call.
    llm_batch: bool = False
    batch_size: int = Field(default=12, ge=8, le=16)
    batch_token_budget: int = Field(default=28_000, ge=4_000)
    # Additional sampled eval (1 positive + N negatives) alongside full-catalog pass.
    n_negatives: int | None = 100
    # Extra @K metrics for the sampled pass only (hr/mrr/ndcg/recall).
    sampled_k_values: list[int] = Field(default_factory=lambda: [1, 3, 5])


class LlmCfg(_Strict):
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    # Offline ABSA (validate/repair); falls back to `model` when unset.
    model_small: str | None = "Qwen/Qwen2.5-3B-Instruct"
    temperature: float = 0.0
    request_timeout_s: int = 120
    max_retries: int = 3


class TgiCfg(_Strict):
    base_url: str = "http://localhost:8080"
    base_url_small: str | None = None
    api_key: str | None = None


def resolve_llm_model(llm: LlmCfg, *, for_absa: bool = False) -> str:
    """Pick agent model vs ABSA model (``LLM_MODEL`` / ``LLM_MODEL_SMALL``)."""
    if for_absa and llm.model_small:
        return llm.model_small
    return llm.model


def resolve_tgi_base_url(tgi: TgiCfg, llm: LlmCfg, *, for_absa: bool = False) -> str:
    """Pick primary vs small-model TGI endpoint."""
    if for_absa and llm.model_small and tgi.base_url_small:
        return tgi.base_url_small
    return tgi.base_url


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
    hettisasrec: HetTiSASRecCfg = Field(default_factory=HetTiSASRecCfg)
    tisasrec_align: TiSASRecAlignCfg = Field(default_factory=TiSASRecAlignCfg)
    neo4j: Neo4jCfg
    tgi: TgiCfg = Field(default_factory=TgiCfg)


class ConfigError(RuntimeError):
    """Raised when configuration is missing required values or is malformed."""


KNOWN_CATEGORIES: tuple[str, ...] = (
    "Beauty_and_Personal_Care",
    "Sports_and_Outdoors",
    "Toys_and_Games",
    "Yelp",
    "Yelp_AC",
)


def category_scoped_artifact_paths(cfg: Config) -> list[tuple[str, str]]:
    """Writable / category-specific artifact paths that must stay isolated."""
    ta = cfg.tisasrec_align
    absa = cfg.absa
    het = cfg.hettisasrec
    return [
        ("data.out_dir", cfg.data.out_dir),
        ("absa.targets_path", absa.targets_path),
        ("absa.cache_path", absa.cache_path),
        ("absa.gold_path", absa.gold_path),
        ("absa.summary_path", absa.summary_path),
        ("absa.preview_html_path", absa.preview_html_path),
        ("tisasrec_align.stage1_checkpoint_path", ta.stage1_checkpoint_path),
        ("tisasrec_align.e_i_matrix_path", ta.e_i_matrix_path),
        ("tisasrec_align.alignment_checkpoint_path", ta.alignment_checkpoint_path),
        ("tisasrec_align.tu_cache_path", ta.tu_cache_path),
        ("tisasrec_align.cross_user_lookup_path", ta.cross_user_lookup_path),
        ("hettisasrec.aspect_vocab_path", het.aspect_vocab_path),
        ("hettisasrec.aspect_graph_path", het.aspect_graph_path),
        ("hettisasrec.checkpoint_path", het.checkpoint_path),
    ]


def validate_category_path_isolation(cfg: Config) -> list[str]:
    """Return errors if any artifact path points at another category's tree.

    Keeps Beauty caches intact when running Sports (and the reverse).
    """
    cat = cfg.data.category
    errors: list[str] = []
    if Path(cfg.data.out_dir).name != cat:
        errors.append(
            f"data.out_dir={cfg.data.out_dir!r} does not match data.category={cat!r}"
        )

    for label, path in category_scoped_artifact_paths(cfg):
        norm = Path(path).as_posix()
        for other in KNOWN_CATEGORIES:
            if other == cat:
                continue
            if f"/{other}/" in f"/{norm}/" or norm.rstrip("/").endswith(f"/{other}"):
                errors.append(
                    f"{label}={path!r} references other category {other!r} "
                    f"(active category is {cat!r})"
                )
                break
        # Processed / model artifacts must live under the active category name.
        needs_token = (
            "/processed/" in norm
            or "/tisasrec_align/" in norm
            or "/hettisasrec/" in norm
            or norm.endswith("absa_cache.sqlite")
            or "absa_targets" in Path(norm).name
        )
        if needs_token and cat not in norm:
            errors.append(
                f"{label}={path!r} is not scoped under category {cat!r}"
            )
    return errors


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


def _load_yaml_with_extends(
    cfg_path: Path,
    *,
    _stack: tuple[Path, ...] = (),
) -> dict:
    """Load YAML and recursively resolve ``extends`` (overlay wins on conflicts)."""
    resolved = cfg_path.resolve()
    if resolved in _stack:
        cycle = " -> ".join(str(p) for p in (*_stack, resolved))
        raise ConfigError(f"Config extends cycle detected: {cycle}")
    if not resolved.exists():
        raise ConfigError(f"Config file not found: {resolved}")

    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping: {resolved}")

    extends = raw.pop("extends", None)
    if not extends:
        return raw

    base_path = (resolved.parent / str(extends)).resolve()
    base_raw = _load_yaml_with_extends(base_path, _stack=(*_stack, resolved))
    return _deep_merge(base_raw, raw)


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    """Load and validate the experiment config plus environment-derived settings.

    Raises ConfigError on missing env vars; raises pydantic ValidationError on
    unknown/malformed YAML keys.
    """
    load_dotenv()

    cfg_path = Path(path)
    raw = _load_yaml_with_extends(cfg_path)

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

    tgi_raw = raw.get("tgi")
    if not isinstance(tgi_raw, dict):
        tgi_raw = {}
        raw["tgi"] = tgi_raw
    if env_base := os.environ.get("TGI_BASE_URL"):
        tgi_raw["base_url"] = env_base
    if env_base_small := os.environ.get("TGI_BASE_URL_SMALL"):
        tgi_raw["base_url_small"] = env_base_small
    if env_api_key := os.environ.get("TGI_API_KEY"):
        tgi_raw["api_key"] = env_api_key

    if not tgi_raw.get("base_url"):
        raise ConfigError(
            "TGI base URL is required. Set tgi.base_url in YAML or TGI_BASE_URL in .env."
        )

    return Config(**raw)
