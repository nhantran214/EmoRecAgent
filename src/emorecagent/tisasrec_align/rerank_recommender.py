"""Stage 2 rerank recommender: Stage-1-anchored pool + LLM + guardrail."""

from __future__ import annotations

import logging
import os
from collections import defaultdict

from ..baselines.base import Recommender
from ..config import Config
from ..data.types import Interaction
from ..llm.client import LLMClient
from .cross_user_lookup import CrossUserLookup, load_lookup, lookup_co_items
from .item_metadata import ItemMeta, load_item_metadata
from .review_context import load_review_text_index, prefix_reviews_for_user
from .stage1_factory import build_stage1_recommender
from .stage1_protocol import Stage1Scorer
from .stage2_llm_rerank import llm_rerank_pool
from .stage2_rerank import (
    apply_cross_user_boosts,
    build_pool,
    check_guardrail,
    merge_ranking,
    reorder_within_head,
)
from .tu_cache import TuCacheRow, cache_key, load_tu_cache

logger = logging.getLogger(__name__)


class RerankAlignRecommender(Recommender):
    """Stage 1 full rank + bounded rerank pool with optional LLM and guardrail."""

    name = "emorecagent_align"

    def __init__(
        self,
        stage1: Stage1Scorer,
        tu_cache: dict[str, TuCacheRow],
        lookup: CrossUserLookup,
        review_index: dict[tuple[str, str, int], str],
        train: list[Interaction],
        *,
        rerank_pool_k: int = 100,
        llm_pool_cap: int = 40,
        cross_user_boost: float = 0.05,
        guardrail_top_n: int = 5,
        guardrail_max_drop_rank: int = 10,
        guardrail_mode: str = "position",
        reorder_head_n: int = 10,
        llm: LLMClient | None = None,
        skip_llm: bool = False,
        item_meta: dict[str, ItemMeta] | None = None,
        cross_user_mode: str = "review_text",
    ) -> None:
        self._stage1 = stage1
        self._tu_cache = tu_cache
        self._lookup = lookup
        self._review_index = review_index
        self._train_by_user: dict[str, list[Interaction]] = defaultdict(list)
        for it in train:
            self._train_by_user[it.user_id].append(it)
        self._rerank_pool_k = rerank_pool_k
        self._llm_pool_cap = llm_pool_cap
        self._cross_user_boost = cross_user_boost
        self._guardrail_top_n = guardrail_top_n
        self._guardrail_max_drop_rank = guardrail_max_drop_rank
        self._guardrail_mode = guardrail_mode
        self._reorder_head_n = reorder_head_n
        self._llm = None if skip_llm else llm
        self._item_meta = item_meta or {}
        self._cross_user_mode = cross_user_mode
        self.n_fallback = 0
        self.n_llm_calls = 0
        self.n_stage1_only = 0
        self._query_ts: dict[str, int] = {}

    @classmethod
    def from_config(
        cls,
        config: Config,
        train: list[Interaction],
        *,
        seed: int = 42,
    ) -> RerankAlignRecommender:
        del seed
        cfg = config.tisasrec_align
        stage1 = build_stage1_recommender(
            config, train, force_stage1_only=True
        )
        tu_cache = load_tu_cache(cfg.tu_cache_path)
        lookup = load_lookup(cfg.cross_user_lookup_path)
        review_index: dict[tuple[str, str, int], str] = {}
        item_meta: dict[str, ItemMeta] = {}
        if cfg.preference_source == "item_metadata" or cfg.cross_user_mode == "id_only":
            meta_root = (
                config.data.meta_path
                or config.data.inter_path
                or config.data.review_path
            )
            try:
                item_meta = load_item_metadata(meta_root)
            except FileNotFoundError as exc:
                logger.warning("item metadata unavailable: %s", exc)
        if cfg.cross_user_mode != "id_only":
            review_index = load_review_text_index(config.data.review_path)
        skip_llm = os.environ.get("NO_LLM", "").strip() in ("1", "true", "yes")
        llm = None if skip_llm else LLMClient.from_config(config)
        return cls(
            stage1,
            tu_cache,
            lookup,
            review_index,
            train,
            rerank_pool_k=cfg.rerank_pool_k,
            llm_pool_cap=cfg.llm_pool_cap,
            cross_user_boost=cfg.cross_user_boost,
            guardrail_top_n=cfg.guardrail_top_n,
            guardrail_max_drop_rank=cfg.guardrail_max_drop_rank,
            guardrail_mode=cfg.guardrail_mode,
            reorder_head_n=cfg.reorder_head_n,
            llm=llm,
            skip_llm=skip_llm,
            item_meta=item_meta,
            cross_user_mode=cfg.cross_user_mode,
        )

    def fit(self, interactions: list[Interaction]) -> RerankAlignRecommender:
        self._stage1.fit(interactions)
        self._train_by_user.clear()
        for it in interactions:
            self._train_by_user[it.user_id].append(it)
        return self

    def prepare_user_query(self, user_id: str, timestamp_ms: int) -> None:
        self._stage1.prepare_user_query(user_id, timestamp_ms)
        self._query_ts[user_id] = timestamp_ms

    def catalog_items(self) -> list[str]:
        return self._stage1.catalog_items()

    def _tu_row(self, user_id: str, query_ts_ms: int) -> TuCacheRow | None:
        return self._tu_cache.get(cache_key(user_id, query_ts_ms))

    def _has_reviews(self, user_id: str, query_ts_ms: int) -> bool:
        """Stage-2 eligibility (review text or metadata-derived preference)."""
        row = self._tu_row(user_id, query_ts_ms)
        return row is not None and row.has_reviews and bool(row.T_u.strip())

    def _prefix_anchor_items(self, user_id: str, query_ts_ms: int) -> list[str]:
        events = self._train_by_user.get(user_id, [])
        if self._cross_user_mode == "id_only":
            items = [
                it.item for it in events if it.timestamp < query_ts_ms
            ]
            # Deduplicate preserving order (recent last).
            seen: set[str] = set()
            out: list[str] = []
            for item in items:
                if item not in seen:
                    seen.add(item)
                    out.append(item)
            return out
        reviews = prefix_reviews_for_user(
            user_id,
            events,
            query_ts_ms,
            self._review_index,
        )
        return [r.item_id for r in reviews]

    def rank(
        self,
        user_id: str,
        candidates: list[str],
        *,
        query_ts_ms: int | None = None,
    ) -> list[str]:
        if query_ts_ms is not None:
            self.prepare_user_query(user_id, query_ts_ms)
        query_ts = query_ts_ms if query_ts_ms is not None else self._query_ts.get(user_id)
        if query_ts is None:
            raise RuntimeError(
                f"prepare_user_query not called for user_id={user_id!r}"
            )

        stage1_ranked = self._stage1.rank(
            user_id, candidates, query_ts_ms=query_ts
        )
        if not self._has_reviews(user_id, query_ts):
            self.n_stage1_only += 1
            return stage1_ranked

        pool_k = min(self._rerank_pool_k, len(stage1_ranked))
        pool = build_pool(stage1_ranked, pool_k)
        if not pool:
            return stage1_ranked

        scores = self._stage1.score(user_id, pool, query_ts_ms=query_ts)
        anchor_items = self._prefix_anchor_items(user_id, query_ts)
        boost_weights = lookup_co_items(anchor_items, set(pool), self._lookup)
        boosted_order = apply_cross_user_boosts(
            pool,
            scores,
            boost_weights,
            self._cross_user_boost,
        )
        llm_cap = min(self._llm_pool_cap, len(boosted_order))
        llm_subset = boosted_order[:llm_cap]
        numeric_order = list(llm_subset)

        row = self._tu_row(user_id, query_ts)
        t_u = row.T_u if row is not None else ""
        id_only = self._cross_user_mode == "id_only"
        if self._llm is not None and llm_subset:
            self.n_llm_calls += 1
            reranked_subset = llm_rerank_pool(
                self._llm,
                t_u=t_u,
                reviewed_items=anchor_items,
                lookup=self._lookup,
                pool=llm_subset,
                scores=scores,
                numeric_fallback=numeric_order,
                item_meta=self._item_meta or None,
                id_only=id_only,
            )
        else:
            reranked_subset = numeric_order

        seen_subset = set(reranked_subset)
        reranked_pool = reranked_subset + [
            item for item in boosted_order if item not in seen_subset
        ]

        if self._guardrail_mode == "reorder_head":
            # Permute only Stage-1's protected head; membership at k >=
            # reorder_head_n is preserved so hr@k / recall@k never regress.
            return reorder_within_head(
                stage1_ranked, reranked_pool, self._reorder_head_n
            )

        merged = merge_ranking(reranked_pool, stage1_ranked, pool_k)
        if self._guardrail_mode == "off":
            return merged
        if not check_guardrail(
            stage1_ranked,
            merged,
            top_n=self._guardrail_top_n,
            max_drop_rank=self._guardrail_max_drop_rank,
        ):
            self.n_fallback += 1
            return stage1_ranked
        return merged

    def score(
        self,
        user_id: str,
        candidates: list[str],
        *,
        query_ts_ms: int | None = None,
    ) -> dict[str, float]:
        return self._stage1.score(
            user_id, candidates, query_ts_ms=query_ts_ms
        )
