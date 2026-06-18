"""LangGraph-backed recommender for the eval harness."""

from __future__ import annotations

import logging
import time

from ..baselines.base import Recommender
from ..baselines.aspect_aware import static_weights
from ..data.types import Interaction
from ..graph.build import build_emorec_graph
from ..hgt.retriever import HGTRetriever
from ..scoring.dynamic_weights import aspect_gammas
from ..scoring.score import rank_items
from .graph_context import GraphContext, build_graph_context

MS_PER_DAY = 86_400_000
COMPLAINT_WINDOW_MS = 30 * MS_PER_DAY


class GraphRecommender(Recommender):
    """Runs the four-agent LangGraph pipeline; full-catalog rank via graph head + numeric tail."""

    name = "emorecagent"

    def __init__(self, context: GraphContext) -> None:
        self._ctx = context
        self._graph = build_emorec_graph(context.graph_deps)
        self._default_query_ts: dict[str, int] = {}
        self._numeric_tail_total_s: float = 0.0
        self._numeric_tail_count: int = 0

    @classmethod
    def from_runner_cfg(
        cls,
        cfg: dict,
        *,
        config=None,
        driver=None,
        memory_kg=None,
    ) -> "GraphRecommender":
        ctx = build_graph_context(
            cfg, config=config, driver=driver, memory_kg=memory_kg
        )
        rec = cls(ctx)
        if ctx.hgt_pool_size is not None:
            rec.name = "emorecagent_hgt"
        return rec

    def fit(self, interactions: list[Interaction]) -> "GraphRecommender":
        per_user: dict[str, int] = {}
        for it in interactions:
            per_user[it.user_id] = max(per_user.get(it.user_id, 0), it.timestamp)
        self._default_query_ts = {u: ts + 1 for u, ts in per_user.items()}
        return self

    def prepare_user_query(self, user_id: str, timestamp_ms: int) -> None:
        self._ctx.query_ts[user_id] = timestamp_ms

    def _query_time(self, user_id: str) -> int:
        return self._ctx.query_ts.get(
            user_id, self._default_query_ts.get(user_id, 0)
        )

    def _aspect_gammas(self, user_id: str, t_query: int) -> dict[str, float]:
        if hasattr(self._ctx.graph_deps.profiling, "profile_gammas"):
            return self._ctx.graph_deps.profiling.profile_gammas(user_id, t_query)
        signals = self._ctx.kg_backend.get_user_signals(user_id, t_query)
        return aspect_gammas(signals, t_query, self._ctx.lambda_decay)

    def _recent_complaints(self, user_id: str, t_query: int) -> list[str]:
        signals = self._ctx.kg_backend.get_user_signals(user_id, t_query)
        out: list[str] = []
        for s in signals:
            if s.polarity >= -0.3:
                continue
            if t_query - s.timestamp_ms > COMPLAINT_WINDOW_MS:
                continue
            if s.aspect not in out:
                out.append(s.aspect)
        return out

    def _select_graph_pool(
        self, user_id: str, candidates: list[str], weights: dict[str, float]
    ) -> list[str]:
        pool_size = self._ctx.graph_deps.reasoning._pool_size
        if self._ctx.hgt_pool_size is not None:
            pool_size = self._ctx.hgt_pool_size
        if len(candidates) <= pool_size:
            return list(candidates)
        if isinstance(self._ctx.cf, HGTRetriever):
            gammas = self._aspect_gammas(user_id, self._query_time(user_id))
            return self._ctx.cf.retrieve(
                user_id,
                pool_size,
                gammas,
                candidates,
            )
        aspect_maps = {
            item: self._ctx.kg_backend.get_item_aspects_rescaled(item)
            for item in candidates
        }
        s_base = self._ctx.cf.score(user_id, candidates)
        ranked = rank_items(self._ctx.alpha, s_base, weights, aspect_maps)
        return [item for item, _ in ranked[:pool_size]]

    def _build_state(
        self,
        user_id: str,
        candidates: list[str],
        exclude: set[str],
        *,
        eval_pool: list[str],
    ) -> dict:
        t_query = self._query_time(user_id)
        item_e_hat = {
            item: self._ctx.kg_backend.get_item_aspects_rescaled(item)
            for item in eval_pool
        }
        aspect_support = {
            item: self._ctx.kg_backend.get_aspect_support(item)
            for item in eval_pool
        }
        return {
            "user_id": user_id,
            "t_query_ms": t_query,
            "exclude_items": set(exclude),
            "eval_candidates": eval_pool,
            "item_e_hat": item_e_hat,
            "aspect_support": aspect_support,
            "item_prices": {},
            "recent_complaint_aspects": self._recent_complaints(user_id, t_query),
        }

    def _graph_top(
        self,
        user_id: str,
        candidates: list[str],
        exclude: set[str],
        weights: dict[str, float],
    ) -> list[str]:
        if not candidates:
            return []
        eval_pool = self._select_graph_pool(user_id, candidates, weights)
        timer = self._ctx.stage_timer
        if timer is not None:
            timer.record_graph_invoke()
        state = self._graph.invoke(
            self._build_state(
                user_id, candidates, exclude, eval_pool=eval_pool
            ),
            config={"recursion_limit": 25},
        )
        recs = list(state.get("recommendations") or [])
        self._last_breakdowns = state.get("breakdowns") or {}
        self._last_weights = state.get("weights") or {}
        return [r for r in recs if r in candidates]

    def _numeric_rank(
        self,
        user_id: str,
        candidates: list[str],
        weights: dict[str, float],
    ) -> list[str]:
        if not candidates:
            return []
        aspect_maps = {
            item: self._ctx.kg_backend.get_item_aspects_rescaled(item)
            for item in candidates
        }
        gammas = None
        if isinstance(self._ctx.cf, HGTRetriever):
            gammas = self._aspect_gammas(user_id, self._query_time(user_id))
            s_base = self._ctx.cf.score(user_id, candidates, gammas=gammas)
        else:
            s_base = self._ctx.cf.score(user_id, candidates)
        ranked = rank_items(self._ctx.alpha, s_base, weights, aspect_maps)
        return [item for item, _ in ranked]

    def _weights(self, user_id: str, t_query: int) -> dict[str, float]:
        if self._ctx.use_dynamic_weights:
            return self._ctx.graph_deps.profiling.profile(
                user_id, t_query, self._ctx.top_k_aspects, persist=False
            )
        signals = self._ctx.kg_backend.get_user_signals(user_id, t_query)
        return static_weights(signals)

    def rank(self, user_id: str, candidates: list[str]) -> list[str]:
        if not candidates:
            return []
        t_query = self._query_time(user_id)
        all_items = set(candidates)
        exclude: set[str] = set()
        weights = self._weights(user_id, t_query)

        graph_top = self._graph_top(user_id, candidates, exclude, weights)
        weights = getattr(self, "_last_weights", None) or weights

        seen: set[str] = set()
        ordered: list[str] = []
        for item in graph_top:
            if item in all_items and item not in seen:
                ordered.append(item)
                seen.add(item)

        tail = [c for c in candidates if c not in seen]
        t_tail = time.monotonic()
        tail_ranked = self._numeric_rank(user_id, tail, weights)
        self._numeric_tail_total_s += time.monotonic() - t_tail
        self._numeric_tail_count += 1
        for item in tail_ranked:
            if item not in seen:
                ordered.append(item)
                seen.add(item)
        return ordered

    def log_stage_timing(self, logger: logging.Logger, *, prefix: str = "graph") -> None:
        timer = self._ctx.stage_timer
        if timer is not None:
            for line in timer.summary_lines():
                logger.info("[%s] %s", prefix, line)
        if self._numeric_tail_count:
            mean = self._numeric_tail_total_s / self._numeric_tail_count
            logger.info(
                "[%s] numeric_tail: n=%s total=%.2fs mean=%.3fs",
                prefix,
                self._numeric_tail_count,
                self._numeric_tail_total_s,
                mean,
            )

    def score(self, user_id: str, candidates: list[str]) -> dict[str, float]:
        ranked = self.rank(user_id, candidates)
        n = len(ranked)
        if n == 0:
            return {}
        return {item: float(n - i) for i, item in enumerate(ranked)}
