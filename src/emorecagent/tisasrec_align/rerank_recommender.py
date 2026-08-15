"""Stage 2 rerank recommender: Stage-1-anchored pool + LLM + guardrail.

Option B (paper §III.F / Fig. 4): ``guardrail_mode=context_dependent`` with
latent alignment fusion, evidence pack, listwise LLM cap C, merge (Eq. 18),
and context-dependent guardrail (Eqs. 19–21).

Ablations / legacy: ``reorder_head``, ``top_k_promote``, rank blend
(``llm_blend_beta > 0``), and confidence gate remain available via config.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ..baselines.base import Recommender
from ..config import Config
from ..data.types import Interaction
from ..llm.client import LLMClient
from .alignment_mlp import AlignmentMLP
from .cross_user_lookup import CrossUserLookup, load_lookup, lookup_co_items
from .item_metadata import ItemMeta, load_stage2_item_metadata
from .item_potential import (
    build_next_item_lookup,
    load_listwise_npz,
    rerank_pool_by_phi,
    score_pool_ltr,
    score_pool_potential,
)
from .review_context import (
    item_review_snippets_from_index,
    load_review_text_index,
    prefix_reviews_for_user,
)
from .stage1_factory import build_stage1_recommender
from .stage1_protocol import Stage1Scorer
from .stage2_llm_rerank import llm_rerank_pool
from .stage2_reason_promote import (
    blend_window_ranks,
    build_weighted_window,
    tu_channel_scores,
)
from .stage2_paper_guard import (
    compute_alignment_confidence,
    context_dependent_window,
    fuse_user_vector,
    fused_pool_scores,
    should_invoke_llm,
    stage1_margin_confidence,
)
from .stage2_rerank import (
    apply_cross_user_boosts,
    blend_rank_orders,
    build_pool,
    check_guardrail,
    merge_ranking,
    reorder_within_head,
)
from .text_encoder import HashEncoder, SentenceTransformerEncoder, TextEncoderBackend
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
        fusion_alpha: float = 0.7,
        guardrail_n0: int = 5,
        guardrail_m0: int = 10,
        guardrail_gamma_n: float = 3.0,
        guardrail_gamma_m: float = 5.0,
        guardrail_n_min: int = 3,
        guardrail_n_max: int = 8,
        guardrail_m_min: int = 8,
        guardrail_m_max: int = 15,
        guardrail_omega: float = 0.7,
        llm_gate_enabled: bool = False,
        llm_min_c_u: float = 0.45,
        llm_max_stage1_margin: float = 0.85,
        llm_rerank_mode: str = "listwise",
        llm_promote_k: int = 10,
        llm_protect_n: int = 5,
        llm_card_max_name: int = 80,
        llm_card_max_cats: int = 5,
        llm_card_review_snippets: bool = False,
        llm_card_max_review_chars: int = 100,
        llm_card_review_candidates: int = 1,
        llm_narrow_cap: int = 12,
        llm_reason_then_pick: bool = False,
        llm_reason_depth: str = "deep",
        llm_hybrid_gate_enabled: bool = True,
        llm_hybrid_overlap_delta: int = 1,
        llm_hybrid_overlap_delta_out_of_band: int = 2,
        llm_hybrid_rank_lo: int = 11,
        llm_hybrid_rank_hi: int = 40,
        llm_hybrid_first_enabled: bool = False,
        llm_hybrid_min_overlap: int = 0,
        llm_scorecard_focus_cap: int = 0,
        llm_overlap_inject: int = 0,
        llm_w_phi: float = 1.0,
        llm_w_tu: float = 0.0,
        llm_w_co: float = 0.0,
        llm_w_llm: float = 0.0,
        llm_pick_mode: str = "scorecard",
        llm_constraint_override: bool = False,
        llm_lexical_first_enabled: bool = False,
        llm_lexical_first_rank_lo: int = 11,
        llm_lexical_first_rank_hi: int = 20,
        llm_lexical_first_overlap_delta: int = 1,
        llm_blend_beta: float = 0.0,
        alignment_mlp: AlignmentMLP | None = None,
        text_encoder: TextEncoderBackend | None = None,
        device: torch.device | None = None,
        llm: LLMClient | None = None,
        skip_llm: bool = False,
        item_meta: dict[str, ItemMeta] | None = None,
        review_snippets: dict[str, list[str]] | None = None,
        cross_user_mode: str = "review_text",
        stage2_score: str = "llm",
        ltr_w: np.ndarray | None = None,
        ltr_mu: np.ndarray | None = None,
        ltr_sd: np.ndarray | None = None,
        item_pop: dict[str, float] | None = None,
        markov_lookup: dict[str, dict[str, float]] | None = None,
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
        self._fusion_alpha = fusion_alpha
        self._guardrail_n0 = guardrail_n0
        self._guardrail_m0 = guardrail_m0
        self._guardrail_gamma_n = guardrail_gamma_n
        self._guardrail_gamma_m = guardrail_gamma_m
        self._guardrail_n_min = guardrail_n_min
        self._guardrail_n_max = guardrail_n_max
        self._guardrail_m_min = guardrail_m_min
        self._guardrail_m_max = guardrail_m_max
        self._guardrail_omega = guardrail_omega
        self._llm_gate_enabled = llm_gate_enabled
        self._llm_min_c_u = llm_min_c_u
        self._llm_max_stage1_margin = llm_max_stage1_margin
        self._llm_rerank_mode = llm_rerank_mode
        self._llm_promote_k = llm_promote_k
        self._llm_protect_n = llm_protect_n
        self._llm_card_max_name = llm_card_max_name
        self._llm_card_max_cats = llm_card_max_cats
        self._llm_card_review_snippets = llm_card_review_snippets
        self._llm_card_max_review_chars = llm_card_max_review_chars
        self._llm_card_review_candidates = llm_card_review_candidates
        self._llm_narrow_cap = llm_narrow_cap
        self._llm_reason_then_pick = llm_reason_then_pick
        self._llm_reason_depth = (
            llm_reason_depth if llm_reason_depth in ("shallow", "deep") else "deep"
        )
        self._llm_hybrid_gate_enabled = bool(llm_hybrid_gate_enabled)
        self._llm_hybrid_overlap_delta = int(llm_hybrid_overlap_delta)
        self._llm_hybrid_overlap_delta_out_of_band = int(
            llm_hybrid_overlap_delta_out_of_band
        )
        self._llm_hybrid_rank_lo = int(llm_hybrid_rank_lo)
        self._llm_hybrid_rank_hi = int(llm_hybrid_rank_hi)
        self._llm_hybrid_first_enabled = bool(llm_hybrid_first_enabled)
        self._llm_hybrid_min_overlap = max(0, int(llm_hybrid_min_overlap))
        self._llm_scorecard_focus_cap = max(0, int(llm_scorecard_focus_cap))
        self._llm_overlap_inject = max(0, int(llm_overlap_inject))
        self._llm_w_phi = max(0.0, float(llm_w_phi))
        self._llm_w_tu = max(0.0, float(llm_w_tu))
        self._llm_w_co = max(0.0, float(llm_w_co))
        self._llm_w_llm = max(0.0, float(llm_w_llm))
        self._llm_pick_mode = (
            llm_pick_mode
            if llm_pick_mode
            in ("scorecard", "lexical_argmax", "argmax_llm_override")
            else "scorecard"
        )
        self._llm_constraint_override = bool(llm_constraint_override)
        self._llm_lexical_first_enabled = bool(llm_lexical_first_enabled)
        self._llm_lexical_first_rank_lo = int(llm_lexical_first_rank_lo)
        self._llm_lexical_first_rank_hi = int(llm_lexical_first_rank_hi)
        self._llm_lexical_first_overlap_delta = int(llm_lexical_first_overlap_delta)
        self._llm_blend_beta = llm_blend_beta
        self._alignment_mlp = alignment_mlp
        self._text_encoder = text_encoder
        self._device = device or torch.device("cpu")
        self._llm = None if skip_llm else llm
        self._item_meta = item_meta or {}
        self._review_snippets = review_snippets or {}
        self._cross_user_mode = cross_user_mode
        self._stage2_score = (
            stage2_score if stage2_score in ("llm", "ltr", "ltr_llm") else "llm"
        )
        self._ltr_w = ltr_w
        self._ltr_mu = ltr_mu
        self._ltr_sd = ltr_sd
        self._item_pop: dict[str, float] = dict(item_pop) if item_pop else {}
        self._markov_lookup: dict[str, dict[str, float]] = markov_lookup or {}
        self._p_u_cache: dict[str, np.ndarray] = {}
        if item_pop is None:
            self._rebuild_ltr_stats(train)
        self.n_fallback = 0
        self.n_llm_calls = 0
        self.n_llm_skipped_gate = 0
        self.n_stage1_only = 0
        self.n_ltr_rerank = 0
        self.n_guardrail_pass = 0
        self.n_stage2_swaps = 0
        self.n_stage2_empty_picks = 0
        self.n_stage2_hybrid_blocked = 0
        self.n_stage2_lexical_first = 0
        self.n_stage2_hybrid_first_filtered = 0
        self.n_stage2_lexical_argmax = 0
        self.n_stage2_llm_override = 0
        self._c_u_sum = 0.0
        self._c_u_count = 0
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
        # Always load title/category cards for LLM rerank (Amazon JSONL or RecBole .item).
        item_meta: dict[str, ItemMeta] = {}
        meta_root = (
            config.data.meta_path
            or config.data.inter_path
            or config.data.review_path
        )
        if meta_root:
            keep_ids = set(stage1.catalog_items())
            try:
                item_meta = load_stage2_item_metadata(meta_root, keep_ids=keep_ids)
            except (FileNotFoundError, ValueError) as exc:
                logger.warning("item metadata unavailable for LLM cards: %s", exc)
            if item_meta:
                logger.info(
                    "LLM candidate cards: item_meta=%s / catalog=%s (%.0f%% coverage)",
                    f"{len(item_meta):,}",
                    f"{len(keep_ids):,}",
                    100.0 * len(item_meta) / max(len(keep_ids), 1),
                )
            else:
                logger.warning(
                    "LLM candidate cards have no titles (item_meta empty); "
                    "rerank will see ASIN/ids only"
                )
        if cfg.cross_user_mode != "id_only":
            review_index = load_review_text_index(config.data.review_path)
        review_snippets: dict[str, list[str]] = {}
        ltr_mode = cfg.stage2_score in ("ltr", "ltr_llm")
        ltr_only = cfg.stage2_score == "ltr"
        if (cfg.llm_card_review_snippets or ltr_mode) and review_index:
            keep_for_snips = set(stage1.catalog_items())
            allowed_reviews = {
                (it.user_id, it.item, int(it.timestamp)) for it in train
            }
            review_snippets = item_review_snippets_from_index(
                review_index,
                keep_ids=keep_for_snips,
                allowed_reviews=allowed_reviews,
                max_chars=cfg.llm_card_max_review_chars,
                max_per_item=max(1, int(cfg.llm_card_review_candidates)),
            )
            logger.info(
                "LLM candidate cards: review_snippets=%s / catalog=%s "
                "(candidates/item=%s)",
                f"{len(review_snippets):,}",
                f"{len(keep_for_snips):,}",
                cfg.llm_card_review_candidates,
            )
        skip_llm = ltr_only or os.environ.get("NO_LLM", "").strip() in (
            "1",
            "true",
            "yes",
        )
        llm = None if skip_llm else LLMClient.from_config(config)

        alignment_mlp: AlignmentMLP | None = None
        text_encoder: TextEncoderBackend | None = None
        # Manifesto projection stays on CPU: TGI (7B) owns the GPU during Stage-2.
        device = torch.device("cpu")
        # Load MLP for paper fusion (context_dependent), confidence gate (C),
        # or LTR φ text/aligned-cosine channel.
        need_align = (
            cfg.guardrail_mode == "context_dependent"
            or cfg.llm_gate_enabled
            or ltr_mode
        )
        if need_align:
            text_encoder = (
                HashEncoder(dim=cfg.text_encoder_dim)
                if cfg.use_hash_encoder
                else SentenceTransformerEncoder()
            )
            if hasattr(text_encoder, "warm_up"):
                text_encoder.warm_up()
            align_path = Path(cfg.alignment_checkpoint_path)
            if align_path.is_file():
                payload = torch.load(align_path, map_location="cpu", weights_only=False)
                meta = payload.get("meta") or {}
                hidden = int(meta.get("hidden_dim", cfg.hidden_units))
                act = str(meta.get("activation", cfg.alignment_activation))
                alignment_mlp = AlignmentMLP(
                    cfg.text_encoder_dim, hidden, activation=act  # type: ignore[arg-type]
                )
                alignment_mlp.load_state_dict(payload["model"])
                alignment_mlp.to(device)
                alignment_mlp.eval()
                logger.info(
                    "Stage-2 alignment MLP+ST on %s (hidden=%s, mode=%s); "
                    "GPU left for TGI",
                    device,
                    hidden,
                    cfg.guardrail_mode,
                )
            else:
                logger.warning(
                    "Stage-2: missing alignment MLP at %s; "
                    "using α=1 (s_u only) / margin-only c_u when T_u unavailable",
                    align_path,
                )

        ltr_w = ltr_mu = ltr_sd = None
        if ltr_mode:
            ltr_path = Path(cfg.item_potential_ltr_path)
            if not ltr_path.is_file():
                raise FileNotFoundError(
                    f"stage2_score={cfg.stage2_score} requires weights at {ltr_path}"
                )
            ltr_w, ltr_mu, ltr_sd = load_listwise_npz(ltr_path)
            logger.info(
                "Stage-2 LTR φ: %s dim=%s%s",
                ltr_path,
                int(ltr_w.shape[0]),
                " (no LLM)" if ltr_only else (
                    " + LLM listwise" if cfg.llm_rerank_mode == "listwise"
                    else " + LLM scorecard"
                ),
            )

        item_pop: dict[str, float] = {}
        for it in train:
            item_pop[str(it.item)] = item_pop.get(str(it.item), 0.0) + 1.0
        markov_lookup = build_next_item_lookup(train) if ltr_mode else {}

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
            fusion_alpha=cfg.fusion_alpha,
            guardrail_n0=cfg.guardrail_n0,
            guardrail_m0=cfg.guardrail_m0,
            guardrail_gamma_n=cfg.guardrail_gamma_n,
            guardrail_gamma_m=cfg.guardrail_gamma_m,
            guardrail_n_min=cfg.guardrail_n_min,
            guardrail_n_max=cfg.guardrail_n_max,
            guardrail_m_min=cfg.guardrail_m_min,
            guardrail_m_max=cfg.guardrail_m_max,
            guardrail_omega=cfg.guardrail_omega,
            llm_gate_enabled=cfg.llm_gate_enabled,
            llm_min_c_u=cfg.llm_min_c_u,
            llm_max_stage1_margin=cfg.llm_max_stage1_margin,
            llm_rerank_mode=cfg.llm_rerank_mode,
            llm_promote_k=cfg.llm_promote_k,
            llm_protect_n=cfg.llm_protect_n,
            llm_card_max_name=cfg.llm_card_max_name,
            llm_card_max_cats=cfg.llm_card_max_cats,
            llm_card_review_snippets=cfg.llm_card_review_snippets,
            llm_card_max_review_chars=cfg.llm_card_max_review_chars,
            llm_card_review_candidates=cfg.llm_card_review_candidates,
            llm_narrow_cap=cfg.llm_narrow_cap,
            llm_reason_then_pick=cfg.llm_reason_then_pick,
            llm_reason_depth=cfg.llm_reason_depth,
            llm_hybrid_gate_enabled=cfg.llm_hybrid_gate_enabled,
            llm_hybrid_overlap_delta=cfg.llm_hybrid_overlap_delta,
            llm_hybrid_overlap_delta_out_of_band=(
                cfg.llm_hybrid_overlap_delta_out_of_band
            ),
            llm_hybrid_rank_lo=cfg.llm_hybrid_rank_lo,
            llm_hybrid_rank_hi=cfg.llm_hybrid_rank_hi,
            llm_hybrid_first_enabled=cfg.llm_hybrid_first_enabled,
            llm_hybrid_min_overlap=cfg.llm_hybrid_min_overlap,
            llm_scorecard_focus_cap=cfg.llm_scorecard_focus_cap,
            llm_overlap_inject=cfg.llm_overlap_inject,
            llm_w_phi=cfg.llm_w_phi,
            llm_w_tu=cfg.llm_w_tu,
            llm_w_co=cfg.llm_w_co,
            llm_w_llm=cfg.llm_w_llm,
            llm_pick_mode=cfg.llm_pick_mode,
            llm_constraint_override=cfg.llm_constraint_override,
            llm_lexical_first_enabled=cfg.llm_lexical_first_enabled,
            llm_lexical_first_rank_lo=cfg.llm_lexical_first_rank_lo,
            llm_lexical_first_rank_hi=cfg.llm_lexical_first_rank_hi,
            llm_lexical_first_overlap_delta=cfg.llm_lexical_first_overlap_delta,
            llm_blend_beta=cfg.llm_blend_beta,
            alignment_mlp=alignment_mlp,
            text_encoder=text_encoder,
            device=device,
            llm=llm,
            skip_llm=skip_llm,
            item_meta=item_meta,
            review_snippets=review_snippets,
            cross_user_mode=cfg.cross_user_mode,
            stage2_score=cfg.stage2_score,
            ltr_w=ltr_w,
            ltr_mu=ltr_mu,
            ltr_sd=ltr_sd,
            item_pop=item_pop,
            markov_lookup=markov_lookup,
        )

    def fit(self, interactions: list[Interaction]) -> RerankAlignRecommender:
        self._stage1.fit(interactions)
        self._train_by_user.clear()
        for it in interactions:
            self._train_by_user[it.user_id].append(it)
        if self._stage2_score in ("ltr", "ltr_llm"):
            self._rebuild_ltr_stats(interactions)
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

    def _project_manifesto(self, t_u: str) -> torch.Tensor | None:
        if self._alignment_mlp is None or self._text_encoder is None:
            return None
        if not t_u.strip():
            return None
        with torch.no_grad():
            emb = self._text_encoder.encode([t_u], device=self._device)
            return self._alignment_mlp(emb).squeeze(0)

    def _pool_scores_and_confidence(
        self,
        user_id: str,
        stage1_ranked: list[str],
        pool: list[str],
        query_ts: int,
        t_u: str,
    ) -> tuple[dict[str, float], float, float]:
        """Fused (or Stage-1) pool scores, ``c_u``, and Stage-1 margin confidence."""
        stage1_scores = self._stage1.score(user_id, pool, query_ts_ms=query_ts)
        s_u = None
        p_u_np = None
        # Dot-product fusion only when alignment MLP is loaded (Option B paths).
        # Without MLP, keep Stage-1 logits so Option A reorder_head is unchanged.
        if (
            self._alignment_mlp is not None
            and hasattr(self._stage1, "user_state")
            and hasattr(self._stage1, "item_embeddings")
        ):
            s_u = self._stage1.user_state(user_id, query_ts_ms=query_ts)
            p_u_t = self._project_manifesto(t_u)
            if p_u_t is not None:
                p_u_np = p_u_t.detach().cpu().numpy()
            x_u = fuse_user_vector(s_u, p_u_np, alpha=self._fusion_alpha)
            item_embs = self._stage1.item_embeddings(pool)
            scores = fused_pool_scores(x_u, item_embs)
            for item in pool:
                scores.setdefault(item, float(stage1_scores.get(item, 0.0)))
        else:
            scores = dict(stage1_scores)
            if hasattr(self._stage1, "user_state"):
                s_u = self._stage1.user_state(user_id, query_ts_ms=query_ts)

        # Margin uses full Stage-1 catalog scores when available.
        full_scores = self._stage1.score(
            user_id, stage1_ranked[: max(self._guardrail_n0, 1)], query_ts_ms=query_ts
        )
        margin_conf = stage1_margin_confidence(
            stage1_ranked, full_scores, n0=self._guardrail_n0
        )
        c_u = compute_alignment_confidence(
            s_u=s_u,
            p_u=p_u_np,
            stage1_ranked=stage1_ranked,
            stage1_scores=full_scores,
            n0=self._guardrail_n0,
            omega=self._guardrail_omega,
        )
        self._c_u_sum += float(c_u)
        self._c_u_count += 1
        return scores, float(c_u), float(margin_conf)

    def _gate_allows_llm(self, c_u: float, margin_conf: float) -> bool:
        # Quality-first override: still compute c_u upstream, but never skip LLM.
        if self._llm_constraint_override:
            return True
        invoke, reason = should_invoke_llm(
            c_u=c_u,
            stage1_margin_conf=margin_conf,
            enabled=self._llm_gate_enabled,
            min_c_u=self._llm_min_c_u,
            max_stage1_margin=self._llm_max_stage1_margin,
        )
        if not invoke:
            self.n_llm_skipped_gate += 1
            if self.n_llm_skipped_gate % 500 == 1:
                logger.info(
                    "stage2 LLM gate skip reason=%s c_u=%.3f margin=%.3f "
                    "(skipped=%s)",
                    reason,
                    c_u,
                    margin_conf,
                    self.n_llm_skipped_gate,
                )
        return invoke

    def _blend_beta(self, c_u: float) -> float | None:
        """Return blend weight only when ``llm_blend_beta > 0`` (ablation).

        Paper §III.F uses Eq. 18 merge + Eq. 21 only — no Stage-1↔LLM rank
        blend. ``c_u`` is unused when blending is off.
        """
        del c_u
        configured = float(self._llm_blend_beta)
        if configured > 0.0:
            return min(configured, 1.0)
        return None

    def _call_llm_rerank(
        self,
        *,
        t_u: str,
        anchor_items: list[str],
        llm_subset: list[str],
        scores: dict[str, float],
        numeric_order: list[str],
        stage1_ranks: dict[str, int],
        c_u: float | None,
        id_only: bool,
        protect_n: int | None = None,
        phi_scores: dict[str, float] | None = None,
        co_scores: dict[str, float] | None = None,
    ) -> list[str]:
        snips = (
            self._review_snippets
            if self._llm_card_review_snippets and self._review_snippets
            else None
        )
        stats = {
            "n_stage2_swaps": 0,
            "n_stage2_empty_picks": 0,
            "n_stage2_hybrid_blocked": 0,
            "n_stage2_lexical_first": 0,
            "n_stage2_hybrid_first_filtered": 0,
            "n_stage2_lexical_argmax": 0,
            "n_stage2_llm_override": 0,
        }
        out = llm_rerank_pool(
            self._llm,
            t_u=t_u,
            reviewed_items=anchor_items,
            lookup=self._lookup,
            pool=llm_subset,
            scores=scores,
            numeric_fallback=numeric_order,
            item_meta=self._item_meta or None,
            id_only=id_only,
            alignment_confidence=c_u,
            stage1_ranks=stage1_ranks,
            review_snippets=snips,
            rerank_mode=self._llm_rerank_mode,  # type: ignore[arg-type]
            promote_k=self._llm_promote_k,
            protect_n=(
                int(protect_n)
                if protect_n is not None
                else self._llm_protect_n
            ),
            card_max_name=self._llm_card_max_name,
            card_max_cats=self._llm_card_max_cats,
            card_max_review_chars=self._llm_card_max_review_chars,
            reason_then_pick=self._llm_reason_then_pick,
            narrow_cap=self._llm_narrow_cap,
            reason_depth=self._llm_reason_depth,
            hybrid_gate_enabled=self._llm_hybrid_gate_enabled,
            hybrid_overlap_delta=self._llm_hybrid_overlap_delta,
            hybrid_overlap_delta_out_of_band=(
                self._llm_hybrid_overlap_delta_out_of_band
            ),
            hybrid_rank_lo=self._llm_hybrid_rank_lo,
            hybrid_rank_hi=self._llm_hybrid_rank_hi,
            hybrid_first_enabled=self._llm_hybrid_first_enabled,
            hybrid_min_overlap=self._llm_hybrid_min_overlap,
            scorecard_focus_cap=self._llm_scorecard_focus_cap,
            lexical_first_enabled=self._llm_lexical_first_enabled,
            lexical_first_rank_lo=self._llm_lexical_first_rank_lo,
            lexical_first_rank_hi=self._llm_lexical_first_rank_hi,
            lexical_first_overlap_delta=self._llm_lexical_first_overlap_delta,
            pick_mode=self._llm_pick_mode,
            swap_stats=stats,
            phi_scores=phi_scores,
            co_scores=co_scores,
            listwise_w_phi=self._llm_w_phi,
            listwise_w_tu=self._llm_w_tu,
            listwise_w_co=self._llm_w_co,
            listwise_w_llm=self._llm_w_llm,
        )
        self.n_stage2_swaps += int(stats.get("n_stage2_swaps", 0))
        self.n_stage2_empty_picks += int(stats.get("n_stage2_empty_picks", 0))
        self.n_stage2_hybrid_blocked += int(
            stats.get("n_stage2_hybrid_blocked", 0)
        )
        self.n_stage2_lexical_first += int(
            stats.get("n_stage2_lexical_first", 0)
        )
        self.n_stage2_hybrid_first_filtered += int(
            stats.get("n_stage2_hybrid_first_filtered", 0)
        )
        self.n_stage2_lexical_argmax += int(
            stats.get("n_stage2_lexical_argmax", 0)
        )
        self.n_stage2_llm_override += int(
            stats.get("n_stage2_llm_override", 0)
        )
        return out

    def _rank_context_dependent(
        self,
        user_id: str,
        stage1_ranked: list[str],
        query_ts: int,
    ) -> list[str]:
        """Paper Algorithm 1 / Fig. 4 (§III.F): fusion → listwise C → merge → Eq. 21."""
        pool_k = min(self._rerank_pool_k, len(stage1_ranked))
        pool = build_pool(stage1_ranked, pool_k)
        if not pool:
            return stage1_ranked

        row = self._tu_row(user_id, query_ts)
        t_u = row.T_u if row is not None else ""
        scores, c_u, margin_conf = self._pool_scores_and_confidence(
            user_id, stage1_ranked, pool, query_ts, t_u
        )
        n_u, m_u = context_dependent_window(
            c_u,
            n0=self._guardrail_n0,
            m0=self._guardrail_m0,
            gamma_n=self._guardrail_gamma_n,
            gamma_m=self._guardrail_gamma_m,
            n_min=self._guardrail_n_min,
            n_max=self._guardrail_n_max,
            m_min=self._guardrail_m_min,
            m_max=self._guardrail_m_max,
        )

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
        # promote_swap / promote_preserve: merge base = Stage-1 pool order.
        # Other modes: reorder within the LLM shortlist, then append boost tail.
        structural = self._llm_rerank_mode in ("promote_preserve", "promote_swap")
        numeric_order = list(pool) if structural else list(llm_subset)
        stage1_ranks = {item: i + 1 for i, item in enumerate(stage1_ranked) if item in pool}
        phi_scores: dict[str, float] | None = None
        tu_scores: dict[str, float] = {}
        if self._stage2_score == "ltr_llm":
            phi_scores = self._phi_for_pool(user_id, pool, query_ts, t_u)
            if phi_scores:
                phi_order = rerank_pool_by_phi(pool, phi_scores)
                self.n_ltr_rerank += 1
                numeric_order = list(phi_order)
                if self._llm_rerank_mode == "listwise":
                    tu_scores = tu_channel_scores(
                        pool,
                        t_u=t_u,
                        item_meta=self._item_meta or None,
                        review_snippets=(
                            self._review_snippets
                            if self._llm_card_review_snippets
                            else None
                        ),
                    )
                    llm_subset = build_weighted_window(
                        pool,
                        phi_scores=phi_scores,
                        tu_scores=tu_scores,
                        co_scores=boost_weights,
                        cap=llm_cap,
                        w_phi=self._llm_w_phi,
                        w_tu=self._llm_w_tu,
                        w_co=self._llm_w_co,
                    )
                else:
                    # φ order is the abstain floor; scorecard may still swap.
                    llm_subset = list(phi_order[:llm_cap])
        # promote_swap: keep fixed protect_n (hr@10 window). preserve: max with N_u.
        if self._llm_rerank_mode == "promote_swap":
            protect_n = int(self._llm_protect_n)
        else:
            protect_n = max(int(self._llm_protect_n), int(n_u))

        id_only = self._cross_user_mode == "id_only"
        use_llm = bool(llm_subset) and (
            self._llm_pick_mode == "lexical_argmax"
            or (
                self._llm is not None
                and self._gate_allows_llm(c_u, margin_conf)
            )
        )
        if use_llm:
            if self._llm is not None:
                self.n_llm_calls += 1
            reranked_subset = self._call_llm_rerank(
                t_u=t_u,
                anchor_items=anchor_items,
                llm_subset=llm_subset,
                scores=scores,
                numeric_order=numeric_order,
                stage1_ranks=stage1_ranks,
                c_u=c_u,
                id_only=id_only,
                protect_n=protect_n,
                phi_scores=phi_scores,
                co_scores=boost_weights,
            )
            # φ-primary mix: LLM rank / T_u / co are secondary. Skip on parse
            # fail (fallback returns the full φ order, not the window).
            if (
                self._stage2_score == "ltr_llm"
                and self._llm_rerank_mode == "listwise"
                and phi_scores
                and set(reranked_subset) == set(llm_subset)
            ):
                reranked_subset = blend_window_ranks(
                    llm_subset,
                    llm_order=reranked_subset,
                    phi_scores=phi_scores,
                    tu_scores=tu_scores,
                    co_scores=boost_weights,
                    w_phi=self._llm_w_phi,
                    w_llm=self._llm_w_llm,
                    w_tu=self._llm_w_tu,
                    w_co=self._llm_w_co,
                )
            # Paper path: llm_blend_beta=0 → skip. Ablation only when != 0.
            blend_beta = self._blend_beta(c_u)
            if blend_beta is not None and not structural:
                reranked_subset = blend_rank_orders(
                    numeric_order, reranked_subset, beta=blend_beta
                )
        else:
            reranked_subset = numeric_order

        if structural:
            reranked_pool = list(reranked_subset)
        else:
            seen_subset = set(reranked_subset)
            tail_src = (
                numeric_order
                if self._stage2_score == "ltr_llm"
                else boosted_order
            )
            reranked_pool = reranked_subset + [
                item for item in tail_src if item not in seen_subset
            ]
            if len(reranked_pool) < len(boosted_order):
                seen_pool = set(reranked_pool)
                reranked_pool.extend(
                    item for item in boosted_order if item not in seen_pool
                )
        merged = merge_ranking(reranked_pool, stage1_ranked, pool_k)
        # Constraint override: accept promote_swap if frozen prefix intact, or
        # ltr_llm listwise (otherwise Eq. 21 would revert to π¹ and drop φ).
        if self._llm_constraint_override and (
            (
                self._llm_rerank_mode == "promote_swap"
                and merged[:protect_n] == stage1_ranked[:protect_n]
            )
            or (
                self._stage2_score == "ltr_llm"
                and self._llm_rerank_mode == "listwise"
            )
        ):
            self.n_guardrail_pass += 1
            self._maybe_log_guardrail_stats()
            return merged
        if not check_guardrail(
            stage1_ranked,
            merged,
            top_n=n_u,
            max_drop_rank=m_u,
        ):
            self.n_fallback += 1
            self._maybe_log_guardrail_stats()
            if self._stage2_score == "ltr_llm" and numeric_order:
                return merge_ranking(numeric_order, stage1_ranked, pool_k)
            return stage1_ranked
        self.n_guardrail_pass += 1
        self._maybe_log_guardrail_stats()
        return merged

    def _rank_reorder_head(
        self,
        user_id: str,
        stage1_ranked: list[str],
        query_ts: int,
    ) -> list[str]:
        """Permute only Stage-1 head (B); top-K promote + blend + gate (C)."""
        head_n = min(self._reorder_head_n, len(stage1_ranked))
        if head_n <= 0:
            return stage1_ranked
        pool = build_pool(stage1_ranked, head_n)

        row = self._tu_row(user_id, query_ts)
        t_u = row.T_u if row is not None else ""
        scores, c_u, margin_conf = self._pool_scores_and_confidence(
            user_id, stage1_ranked, pool, query_ts, t_u
        )

        if not self._gate_allows_llm(c_u, margin_conf):
            self._maybe_log_guardrail_stats()
            return stage1_ranked

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
        stage1_ranks = {
            item: i + 1 for i, item in enumerate(stage1_ranked) if item in pool
        }

        id_only = self._cross_user_mode == "id_only"
        if self._llm is not None and llm_subset:
            self.n_llm_calls += 1
            reranked_subset = self._call_llm_rerank(
                t_u=t_u,
                anchor_items=anchor_items,
                llm_subset=llm_subset,
                scores=scores,
                numeric_order=numeric_order,
                stage1_ranks=stage1_ranks,
                c_u=c_u,
                id_only=id_only,
            )
            blend_beta = self._blend_beta(c_u)
            if blend_beta is not None:
                reranked_subset = blend_rank_orders(
                    list(pool), reranked_subset, beta=blend_beta
                )
        else:
            reranked_subset = numeric_order

        seen_subset = set(reranked_subset)
        reranked_pool = reranked_subset + [
            item for item in boosted_order if item not in seen_subset
        ]
        self.n_guardrail_pass += 1
        self._maybe_log_guardrail_stats()
        return reorder_within_head(stage1_ranked, reranked_pool, head_n)

    def _maybe_log_guardrail_stats(self) -> None:
        total = self.n_fallback + self.n_guardrail_pass + self.n_llm_skipped_gate
        if total == 0 or total % 500 != 0:
            return
        mean_c = self._c_u_sum / max(self._c_u_count, 1)
        logger.info(
            "stage2 guardrail: pass=%s fallback=%s gate_skip=%s "
            "(%.1f%% reject) llm_calls=%s stage1_only=%s mean_c_u=%.3f "
            "item_meta=%s mode=%s",
            self.n_guardrail_pass,
            self.n_fallback,
            self.n_llm_skipped_gate,
            100.0 * self.n_fallback / max(self.n_fallback + self.n_guardrail_pass, 1),
            self.n_llm_calls,
            self.n_stage1_only,
            mean_c,
            f"{len(self._item_meta):,}",
            self._guardrail_mode,
        )

    def _rebuild_ltr_stats(self, interactions: list[Interaction]) -> None:
        pop: dict[str, float] = {}
        for it in interactions:
            pop[str(it.item)] = pop.get(str(it.item), 0.0) + 1.0
        self._item_pop = pop
        self._markov_lookup = build_next_item_lookup(interactions)

    def _history_items(self, user_id: str, query_ts: int) -> list[str]:
        events = sorted(
            (
                it
                for it in self._train_by_user.get(user_id, [])
                if int(it.timestamp) < query_ts
            ),
            key=lambda it: (int(it.timestamp), str(it.item)),
        )
        return [str(it.item) for it in events]

    def _t_u_for_query(self, user_id: str, query_ts: int) -> str:
        row = self._tu_row(user_id, query_ts)
        if row is not None:
            return str(row.T_u or "")
        return ""

    def _p_u_numpy(self, t_u: str) -> np.ndarray | None:
        key = t_u.strip()
        if not key:
            return None
        cached = self._p_u_cache.get(key)
        if cached is not None:
            return cached
        tensor = self._project_manifesto(t_u)
        if tensor is None:
            return None
        vec = tensor.detach().cpu().numpy()
        self._p_u_cache[key] = vec
        return vec

    def _phi_for_pool(
        self,
        user_id: str,
        pool: list[str],
        query_ts: int,
        t_u: str,
    ) -> dict[str, float] | None:
        """Listwise φ on ``pool``; None when LTR weights are missing."""
        if self._ltr_w is None or self._ltr_mu is None or self._ltr_sd is None:
            return None
        if not pool:
            return {}
        history_items = self._history_items(user_id, query_ts)
        p_u = self._p_u_numpy(t_u)
        seq_scores = self._stage1.score(user_id, pool, query_ts_ms=query_ts)
        need_ids = list(dict.fromkeys(list(pool) + history_items))
        item_embs: dict[str, np.ndarray] = {}
        if hasattr(self._stage1, "item_embeddings"):
            item_embs = self._stage1.item_embeddings(need_ids)
        scored = score_pool_potential(
            pool,
            t_u=t_u,
            p_u=p_u,
            item_embs=item_embs,
            seq_scores=seq_scores,
            history_items=history_items,
            anchor_items=history_items,
            lookup=self._lookup,
            item_meta=self._item_meta or None,
            review_snippets=self._review_snippets or None,
            item_pop=self._item_pop,
            markov_lookup=self._markov_lookup,
        )
        return score_pool_ltr(
            pool, scored, w=self._ltr_w, mu=self._ltr_mu, sd=self._ltr_sd
        )

    def _rank_ltr_pool(
        self,
        user_id: str,
        stage1_ranked: list[str],
        query_ts: int,
    ) -> list[str]:
        """Rerank π¹[:K] by listwise φ; keep Stage-1 tail (no LLM)."""
        pool_k = min(self._rerank_pool_k, len(stage1_ranked))
        pool = list(stage1_ranked[:pool_k])
        if not pool:
            return stage1_ranked
        t_u = self._t_u_for_query(user_id, query_ts)
        phi = self._phi_for_pool(user_id, pool, query_ts, t_u)
        if not phi:
            self.n_stage1_only += 1
            return stage1_ranked
        reranked = rerank_pool_by_phi(pool, phi)
        self.n_ltr_rerank += 1
        return merge_ranking(reranked, stage1_ranked, pool_k)

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
        if self._stage2_score == "ltr":
            return self._rank_ltr_pool(user_id, stage1_ranked, query_ts)
        if not self._has_reviews(user_id, query_ts):
            if self._stage2_score == "ltr_llm":
                return self._rank_ltr_pool(user_id, stage1_ranked, query_ts)
            self.n_stage1_only += 1
            return stage1_ranked

        if self._guardrail_mode == "context_dependent":
            return self._rank_context_dependent(user_id, stage1_ranked, query_ts)

        if self._guardrail_mode == "reorder_head":
            return self._rank_reorder_head(user_id, stage1_ranked, query_ts)

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
