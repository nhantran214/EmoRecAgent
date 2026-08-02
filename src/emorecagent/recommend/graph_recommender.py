"""LangGraph-backed recommender for the eval harness."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from ..baselines.base import Recommender
from ..baselines.aspect_aware import static_weights
from ..data.types import Interaction
from ..graph.build import build_emorec_graph
from ..scoring.dynamic_weights import aspect_gammas
from ..scoring.score import ScoreBreakdown, rank_items
from .graph_context import GraphContext, build_graph_context, reset_active_query_ts, set_active_query_ts

MS_PER_DAY = 86_400_000
COMPLAINT_WINDOW_MS = 30 * MS_PER_DAY


def _is_neural_pool_retriever(cf: object) -> bool:
    return hasattr(cf, "retrieve") and hasattr(cf, "pool_size")


class GraphRecommender(Recommender):
    """Runs the four-agent LangGraph pipeline; full-catalog rank via graph head + numeric tail."""

    name = "emorecagent"

    def __init__(self, context: GraphContext) -> None:
        self._ctx = context
        self._graph = build_emorec_graph(context.graph_deps)
        self._default_query_ts: dict[str, int] = {}
        self._numeric_tail_total_s: float = 0.0
        self._numeric_tail_count: int = 0
        self._timing_lock = threading.Lock()

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
        return rec

    def fit(self, interactions: list[Interaction]) -> "GraphRecommender":
        per_user: dict[str, int] = {}
        for it in interactions:
            per_user[it.user_id] = max(per_user.get(it.user_id, 0), it.timestamp)
        self._default_query_ts = {u: ts + 1 for u, ts in per_user.items()}
        return self

    def prepare_user_query(self, user_id: str, timestamp_ms: int) -> None:
        self._ctx.query_ts[user_id] = timestamp_ms
        if hasattr(self._ctx.cf, "prepare_user_query"):
            self._ctx.cf.prepare_user_query(user_id, timestamp_ms)

    def _resolve_query_ts(self, user_id: str, query_ts_ms: int | None) -> int:
        if query_ts_ms is not None:
            return query_ts_ms
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
        self, user_id: str, candidates: list[str], weights: dict[str, float], t_query: int
    ) -> list[str]:
        pool_size = self._ctx.graph_deps.reasoning._pool_size
        if self._ctx.hgt_pool_size is not None:
            pool_size = self._ctx.hgt_pool_size
        if len(candidates) <= pool_size:
            return list(candidates)
        if _is_neural_pool_retriever(self._ctx.cf):
            gammas = self._aspect_gammas(user_id, t_query)
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
        t_query: int,
    ) -> dict[str, Any]:
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
        t_query: int,
    ) -> tuple[list[str], dict[str, float]]:
        if not candidates:
            return [], weights
        eval_pool = self._select_graph_pool(user_id, candidates, weights, t_query)
        timer = self._ctx.stage_timer
        if self._ctx.enable_stage_timing and timer is not None:
            timer.record_graph_invoke()
        state = self._graph.invoke(
            self._build_state(
                user_id, candidates, exclude, eval_pool=eval_pool, t_query=t_query
            ),
            config={"recursion_limit": 25},
        )
        pool_order = list(state.get("ranked_pool_order") or [])
        prefix = self._ctx.llm_rank_prefix
        candidate_set = set(candidates)
        if pool_order:
            graph_top = [
                item for item in pool_order[:prefix] if item in candidate_set
            ]
        else:
            recs = list(state.get("recommendations") or [])
            graph_top = [r for r in recs if r in candidate_set][:prefix]
        out_weights = dict(state.get("weights") or weights)
        return graph_top, out_weights

    def _numeric_rank(
        self,
        user_id: str,
        candidates: list[str],
        weights: dict[str, float],
        t_query: int,
    ) -> list[str]:
        if not candidates:
            return []
        aspect_maps = {
            item: self._ctx.kg_backend.get_item_aspects_rescaled(item)
            for item in candidates
        }
        gammas = None
        if _is_neural_pool_retriever(self._ctx.cf):
            gammas = self._aspect_gammas(user_id, t_query)
            s_base = self._ctx.cf.score(user_id, candidates, gammas=gammas)
        else:
            s_base = self._ctx.cf.score(user_id, candidates)
        ranked = rank_items(self._ctx.alpha, s_base, weights, aspect_maps)
        return [item for item, _ in ranked]

    def batch_precompute(
        self,
        user_id: str,
        candidates: list[str],
        query_ts_ms: int,
    ) -> tuple[
        int,
        dict[str, float],
        list[str],
        dict[str, ScoreBreakdown],
        list[str],
        dict[str, dict[str, float]],
        list[str],
    ]:
        """Pre-LLM eval context: weights, filtered pool, breakdowns, numeric order."""
        from ..agents.reasoning_agent import expand_pool_with_sar

        t_query = self._resolve_query_ts(user_id, query_ts_ms)
        weights = self._weights(user_id, t_query)
        reasoning = self._ctx.graph_deps.reasoning
        eval_pool = self._select_graph_pool(user_id, candidates, weights, t_query)
        pool = expand_pool_with_sar(
            list(eval_pool),
            reasoning._strong,
            weights,
            exclude=set(),
            pool_size=reasoning._pool_size,
            aspect_recall_max=reasoning._aspect_recall_max,
        )
        filtered, aspect_maps = reasoning._filter_pool(pool, None, {})
        ranked = reasoning._numeric_rank(user_id, filtered, weights, aspect_maps)
        numeric_order = [item for item, _ in ranked]
        breakdowns: dict[str, ScoreBreakdown] = {
            item: bd for item, bd in ranked
        }
        item_e_hat = {
            item: self._ctx.kg_backend.get_item_aspects_rescaled(item)
            for item in filtered
        }
        complaints = self._recent_complaints(user_id, t_query)
        return (
            t_query,
            weights,
            filtered,
            breakdowns,
            numeric_order,
            item_e_hat,
            complaints,
        )

    def batch_merge_rank(
        self,
        user_id: str,
        candidates: list[str],
        weights: dict[str, float],
        query_ts_ms: int,
        pool_order: list[str],
    ) -> list[str]:
        """Merge LLM pool order prefix with numeric tail over full candidates."""
        t_query = self._resolve_query_ts(user_id, query_ts_ms)
        all_items = set(candidates)
        prefix = self._ctx.llm_rank_prefix
        seen: set[str] = set()
        ordered: list[str] = []
        for item in pool_order[:prefix]:
            if item in all_items and item not in seen:
                ordered.append(item)
                seen.add(item)
        tail = [c for c in candidates if c not in seen]
        tail_ranked = self._numeric_rank(user_id, tail, weights, t_query)
        for item in tail_ranked:
            if item not in seen:
                ordered.append(item)
                seen.add(item)
        return ordered

    def _weights(self, user_id: str, t_query: int) -> dict[str, float]:
        if self._ctx.use_dynamic_weights:
            return self._ctx.graph_deps.profiling.profile(
                user_id, t_query, self._ctx.top_k_aspects, persist=False
            )
        signals = self._ctx.kg_backend.get_user_signals(user_id, t_query)
        return static_weights(signals)

    def rank(
        self,
        user_id: str,
        candidates: list[str],
        *,
        query_ts_ms: int | None = None,
    ) -> list[str]:
        if not candidates:
            return []
        ts_token = None
        if query_ts_ms is not None:
            ts_token = set_active_query_ts(query_ts_ms)
        try:
            return self._rank_impl(user_id, candidates, query_ts_ms)
        finally:
            if ts_token is not None:
                reset_active_query_ts(ts_token)

    def _rank_impl(
        self,
        user_id: str,
        candidates: list[str],
        query_ts_ms: int | None,
    ) -> list[str]:
        if not candidates:
            return []
        t_query = self._resolve_query_ts(user_id, query_ts_ms)
        all_items = set(candidates)
        exclude: set[str] = set()
        weights = self._weights(user_id, t_query)

        graph_top, weights = self._graph_top(
            user_id, candidates, exclude, weights, t_query
        )

        seen: set[str] = set()
        ordered: list[str] = []
        for item in graph_top:
            if item in all_items and item not in seen:
                ordered.append(item)
                seen.add(item)

        tail = [c for c in candidates if c not in seen]
        t_tail = time.monotonic()
        tail_ranked = self._numeric_rank(user_id, tail, weights, t_query)
        if self._ctx.enable_stage_timing:
            with self._timing_lock:
                self._numeric_tail_total_s += time.monotonic() - t_tail
                self._numeric_tail_count += 1
        for item in tail_ranked:
            if item not in seen:
                ordered.append(item)
                seen.add(item)
        return ordered

    def log_stage_timing(self, logger: logging.Logger, *, prefix: str = "graph") -> None:
        if not self._ctx.enable_stage_timing:
            return
        timer = self._ctx.stage_timer
        if timer is not None:
            for line in timer.summary_lines():
                logger.info("[%s] %s", prefix, line)
        with self._timing_lock:
            n_tail = self._numeric_tail_count
            total_tail = self._numeric_tail_total_s
        if n_tail:
            mean = total_tail / n_tail
            logger.info(
                "[%s] numeric_tail: n=%s total=%.2fs mean=%.3fs",
                prefix,
                n_tail,
                total_tail,
                mean,
            )

    def score(self, user_id: str, candidates: list[str]) -> dict[str, float]:
        ranked = self.rank(user_id, candidates)
        n = len(ranked)
        if n == 0:
            return {}
        return {item: float(n - i) for i, item in enumerate(ranked)}
