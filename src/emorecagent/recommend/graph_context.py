"""Build LangGraph dependencies from config + KG backend."""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import Any, Callable

from ..agents.profiling_agent import DynamicUserProfilingAgent
from ..agents.reasoning_agent import ReasoningAgent
from ..agents.reflection_agent import ReflectionAgent
from ..config import Config
from ..graph.build import GraphDeps
from ..graph.timing import GraphStageTimer
from ..kg.memory import InMemoryKG
from ..kg.neo4j_adapter import GraphKGBackend, InMemoryGraphKG, Neo4jGraphKG
from ..kg.repository import KGRepository
from ..llm.client import LLMClient
from ..llm.schemas import AbsaTriple
from ..scoring.cf_base import CFBase
from ..scoring.dynamic_weights import AspectSignal

_active_query_ts: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "graph_active_query_ts", default=None
)


def set_active_query_ts(ts: int | None) -> contextvars.Token[int | None]:
    return _active_query_ts.set(ts)


def reset_active_query_ts(token: contextvars.Token[int | None]) -> None:
    _active_query_ts.reset(token)


class _QuerySignalSource:
    def __init__(
        self,
        backend: GraphKGBackend,
        query_ts: Callable[[str], int],
    ) -> None:
        self._backend = backend
        self._query_ts = query_ts

    def get_user_aspect_signals(self, user_id: str) -> list[AspectSignal]:
        t_query = self._query_ts(user_id)
        return self._backend.get_user_signals(user_id, t_query)

    def upsert_user_preferences(self, user_id, weights, updated_ts):  # noqa: ANN001
        pass


class _ItemAspectSource:
    def __init__(self, backend: GraphKGBackend) -> None:
        self._backend = backend

    def get_item_aspects(self, item_id: str) -> dict[str, float]:
        return self._backend.get_item_aspects_rescaled(item_id)


class _AspectStrongSource:
    def __init__(
        self,
        backend: GraphKGBackend,
        pool_size: int,
        *,
        tau: float = 0.65,
        min_support: int = 3,
    ) -> None:
        self._backend = backend
        self._pool_size = pool_size
        self._tau = tau
        self._min_support = min_support

    def items_strong_on(
        self, aspects: list[str], limit: int, exclude: set[str]
    ) -> list[str]:
        if hasattr(self._backend, "items_strong_on"):
            return self._backend.items_strong_on(  # type: ignore[attr-defined]
                aspects,
                limit,
                exclude,
                tau=self._tau,
                min_support=self._min_support,
            )
        return []


@dataclass
class GraphContext:
    graph_deps: GraphDeps
    cf: CFBase
    kg_backend: GraphKGBackend
    alpha: float
    top_k_aspects: int
    use_dynamic_weights: bool
    use_llm_cot: bool
    llm: LLMClient | None
    query_ts: dict[str, int]
    stage_timer: GraphStageTimer
    hgt_pool_size: int | None = None
    lambda_decay: float = 0.01
    llm_rank_prefix: int = 20
    enable_stage_timing: bool = True


def build_graph_context(
    cfg: dict,
    *,
    config: Config | None = None,
    driver: Any | None = None,
    memory_kg: InMemoryKG | None = None,
) -> GraphContext:
    """Construct graph dependencies for experiment or tests."""
    config = config or cfg.get("app_config")
    train = cfg["train_interactions"]
    seed = int(cfg.get("seed", 42))
    alpha = float(cfg.get("alpha", 0.5))
    if not cfg.get("use_aspect_term", True):
        alpha = 1.0
    lambda_decay = float(cfg.get("lambda_decay", 0.01))
    pool_size = int(cfg.get("pool_size", 200))
    top_k_aspects = int(cfg.get("top_k_aspects", 5))
    max_reflection_iters = int(cfg.get("max_reflection_iters", 2))
    if not cfg.get("use_reflection", True):
        max_reflection_iters = 0
    use_dynamic_weights = bool(cfg.get("use_dynamic_weights", True))
    use_llm_cot = bool(cfg.get("use_llm_cot", True))
    affective_rescaled = bool(cfg.get("affective_rescaled", True))
    kg_backend_name = str(cfg.get("kg_backend", "neo4j"))
    llm_rank_prefix = int(cfg.get("llm_rank_prefix", 20))
    aspect_recall_tau = float(cfg.get("aspect_recall_tau", 0.65))
    aspect_recall_max = int(cfg.get("aspect_recall_max", 15))
    ranking_num_predict = int(cfg.get("ranking_num_predict", 4096))
    if config is not None:
        llm_rank_prefix = int(config.agents.llm_rank_prefix)
        aspect_recall_tau = float(config.agents.aspect_recall_tau)
        aspect_recall_max = int(config.agents.aspect_recall_max)
        ranking_num_predict = int(config.agents.ranking_num_predict)
    parallel_workers = int(cfg.get("parallel_workers", 1))
    enable_stage_timing = parallel_workers <= 1

    cf_backend = str(cfg.get("cf_backend", "svd"))
    hgt_pool_size: int | None = None
    cf = CFBase(
        backend=cf_backend,
        factors=int(cfg.get("factors", 64)),
        seed=seed,
    ).fit(train)

    query_ts: dict[str, int] = {}

    def _query_ts(user_id: str) -> int:
        active = _active_query_ts.get()
        if active is not None:
            return active
        return query_ts.get(user_id, 0)

    if kg_backend_name == "memory":
        if memory_kg is None:
            from .context import build_recommend_context

            memory_cf_backend = cf_backend
            ctx = build_recommend_context(
                train,
                seed=seed,
                cf_backend=memory_cf_backend,
                cf_factors=int(cfg.get("factors", 64)),
                alpha=alpha,
                lambda_decay=lambda_decay,
                helpful_cap=int(cfg.get("helpful_cap", 10)),
                affective_rescaled=affective_rescaled,
                absa_cache_path=cfg.get("absa_cache_path"),
                review_path=cfg.get("review_path"),
                max_reflection_iters=max_reflection_iters,
                pool_size=pool_size,
                top_k_aspects=top_k_aspects,
                use_reflection=cfg.get("use_reflection", True),
                use_dynamic_weights=use_dynamic_weights,
                use_aspect_term=cfg.get("use_aspect_term", True),
            )
            memory_kg = ctx.kg
        backend: GraphKGBackend = InMemoryGraphKG(
            memory_kg, affective_rescaled=affective_rescaled
        )
    else:
        if driver is None:
            if config is None:
                raise ValueError("neo4j kg_backend requires config or driver")
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(
                config.neo4j.uri,
                auth=(config.neo4j.user, config.neo4j.password),
            )
        backend = Neo4jGraphKG(
            KGRepository(driver), affective_rescaled=affective_rescaled
        )

    signal_source = _QuerySignalSource(backend, _query_ts)
    profiling = DynamicUserProfilingAgent(signal_source, lambda_per_day=lambda_decay)
    aspect_src = _ItemAspectSource(backend)
    strong_src = _AspectStrongSource(
        backend,
        pool_size,
        tau=aspect_recall_tau,
        min_support=3,
    )

    llm: LLMClient | None = None
    if use_llm_cot and config is not None:
        llm = LLMClient.from_config(config)

    reasoning = ReasoningAgent(
        cf,
        aspect_src,
        strong_src,
        llm,
        alpha=alpha,
        pool_size=pool_size,
        aspect_recall_max=aspect_recall_max,
        ranking_num_predict=ranking_num_predict,
    )
    reflection = ReflectionAgent(
        llm=llm,
        use_llm_judge=llm is not None and cfg.get("use_reflection", True),
    )

    def load_triples(user_id: str) -> list[AbsaTriple]:
        return backend.load_user_triples(user_id, _query_ts(user_id))

    def user_signals(user_id: str, before_ts: int) -> list[AspectSignal]:
        return backend.get_user_signals(user_id, before_ts)

    def aspect_support(item_id: str) -> dict[str, int]:
        return backend.get_aspect_support(item_id)

    stage_timer = GraphStageTimer()
    deps = GraphDeps(
        profiling=profiling,
        reasoning=reasoning,
        reflection=reflection,
        load_triples=load_triples,
        user_signals=user_signals,
        aspect_support=aspect_support,
        max_reflection_iters=max_reflection_iters,
        top_k=top_k_aspects,
        stage_timer=stage_timer,
    )

    return GraphContext(
        graph_deps=deps,
        cf=cf,
        kg_backend=backend,
        alpha=alpha,
        top_k_aspects=top_k_aspects,
        use_dynamic_weights=use_dynamic_weights,
        use_llm_cot=use_llm_cot,
        llm=llm,
        query_ts=query_ts,
        stage_timer=stage_timer,
        hgt_pool_size=hgt_pool_size,
        lambda_decay=lambda_decay,
        llm_rank_prefix=llm_rank_prefix,
        enable_stage_timing=enable_stage_timing,
    )
