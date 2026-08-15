"""Tests for RerankAlignRecommender."""

from __future__ import annotations

from collections import defaultdict
from unittest.mock import MagicMock

import torch

from emorecagent.data.types import Interaction
from emorecagent.sequential.id_maps import IdMaps
from emorecagent.tisasrec_align.alignment_mlp import AlignmentMLP
from emorecagent.tisasrec_align.checkpoint import AlignBundle
from emorecagent.tisasrec_align.full_rank_recommender import AlignFullRankRecommender
from emorecagent.tisasrec_align.model import TiSASRecArgs, TiSASRecModel
from emorecagent.tisasrec_align.rerank_recommender import RerankAlignRecommender
from emorecagent.tisasrec_align.tu_cache import TuCacheRow


def _minimal_bundle(hidden: int = 8) -> AlignBundle:
    args = TiSASRecArgs(
        hidden_units=hidden,
        maxlen=5,
        num_blocks=1,
        num_heads=1,
        dropout_rate=0.0,
        time_span=16,
        l2_emb=0.0,
    )
    model = TiSASRecModel(4, args)
    mlp = AlignmentMLP(768, hidden)
    e_i = torch.randn(4, hidden)
    return AlignBundle(
        tisasrec=model,
        alignment_mlp=mlp,
        item_ids=["i1", "i2", "i3"],
        e_i_matrix=e_i,
        args=args,
        tau=0.1,
    )


def _stage1_rec() -> AlignFullRankRecommender:
    bundle = _minimal_bundle()
    id_maps = IdMaps(
        user_to_idx={"u1": 1},
        item_to_idx={"i1": 1, "i2": 2, "i3": 3},
    )
    rec = AlignFullRankRecommender(
        bundle,
        id_maps,
        {},
        fusion_alpha=0.7,
        device=torch.device("cpu"),
        use_hash_encoder=True,
        stage1_only=True,
    )
    rec._user_events = defaultdict(list, {"u1": [(500, "i1")]})
    return rec


def _rerank_rec(
    *,
    tu_cache: dict[str, TuCacheRow] | None = None,
    lookup: dict | None = None,
    llm=None,
    guardrail_max_drop_rank: int = 10,
) -> RerankAlignRecommender:
    train = [Interaction(user_id="u1", item="i1", rating=5.0, timestamp=500)]
    return RerankAlignRecommender(
        _stage1_rec(),
        tu_cache or {},
        lookup or {},
        review_index={("u1", "i1", 500): "nice"},
        train=train,
        rerank_pool_k=3,
        llm_pool_cap=2,
        cross_user_boost=0.05,
        guardrail_top_n=5,
        guardrail_max_drop_rank=guardrail_max_drop_rank,
        llm=llm,
        skip_llm=llm is None,
    )


def test_no_reviews_returns_stage1_order() -> None:
    rec = _rerank_rec()
    rec.prepare_user_query("u1", 1000)
    candidates = ["i1", "i2", "i3"]
    stage1 = rec._stage1.rank("u1", candidates, query_ts_ms=1000)
    ranked = rec.rank("u1", candidates, query_ts_ms=1000)
    assert ranked == stage1
    assert rec.n_stage1_only == 1


def test_with_reviews_reranks_without_llm() -> None:
    tu_cache = {
        "u1|1000": TuCacheRow(
            user_id="u1",
            query_ts_ms=1000,
            T_u="likes i1",
            has_reviews=True,
        )
    }
    rec = _rerank_rec(tu_cache=tu_cache)
    rec.prepare_user_query("u1", 1000)
    ranked = rec.rank("u1", ["i1", "i2", "i3"], query_ts_ms=1000)
    assert len(ranked) == 3
    assert set(ranked) == {"i1", "i2", "i3"}


def test_guardrail_fallback_to_stage1(monkeypatch) -> None:
    tu_cache = {
        "u1|1000": TuCacheRow(
            user_id="u1",
            query_ts_ms=1000,
            T_u="likes i1",
            has_reviews=True,
        )
    }
    llm = MagicMock()
    llm.invoke_ranking_json.return_value = ["i3", "i2"]
    rec = _rerank_rec(
        tu_cache=tu_cache,
        llm=llm,
        guardrail_max_drop_rank=1,
    )
    rec.prepare_user_query("u1", 1000)
    candidates = ["i1", "i2", "i3"]
    stage1 = rec._stage1.rank("u1", candidates, query_ts_ms=1000)
    ranked = rec.rank("u1", candidates, query_ts_ms=1000)
    assert ranked == stage1
    assert rec.n_fallback == 1


def test_reorder_head_preserves_membership_beyond_head() -> None:
    tu_cache = {
        "u1|1000": TuCacheRow(
            user_id="u1",
            query_ts_ms=1000,
            T_u="likes i1",
            has_reviews=True,
        )
    }
    llm = MagicMock()
    llm.invoke_ranking_json.return_value = ["i3", "i2", "i1"]
    train = [Interaction(user_id="u1", item="i1", rating=5.0, timestamp=500)]
    rec = RerankAlignRecommender(
        _stage1_rec(),
        tu_cache,
        {},
        review_index={("u1", "i1", 500): "nice"},
        train=train,
        rerank_pool_k=3,
        llm_pool_cap=3,
        guardrail_mode="reorder_head",
        reorder_head_n=2,
        llm=llm,
        skip_llm=False,
    )
    candidates = ["i1", "i2", "i3"]
    stage1 = rec._stage1.rank("u1", candidates, query_ts_ms=1000)
    ranked = rec.rank("u1", candidates, query_ts_ms=1000)
    assert set(ranked[:2]) == set(stage1[:2])
    assert ranked[2:] == stage1[2:]
    assert rec.n_llm_calls == 1


def test_llm_gate_skips_on_high_margin() -> None:
    tu_cache = {
        "u1|1000": TuCacheRow(
            user_id="u1",
            query_ts_ms=1000,
            T_u="likes i1",
            has_reviews=True,
        )
    }
    llm = MagicMock()
    llm.invoke_ranking_json.return_value = ["i3", "i2", "i1"]
    train = [Interaction(user_id="u1", item="i1", rating=5.0, timestamp=500)]
    rec = RerankAlignRecommender(
        _stage1_rec(),
        tu_cache,
        {},
        review_index={("u1", "i1", 500): "nice"},
        train=train,
        rerank_pool_k=3,
        llm_pool_cap=3,
        guardrail_mode="reorder_head",
        reorder_head_n=2,
        llm_gate_enabled=True,
        llm_min_c_u=0.0,
        llm_max_stage1_margin=0.0,  # any positive margin → skip
        llm=llm,
        skip_llm=False,
    )
    candidates = ["i1", "i2", "i3"]
    stage1 = rec._stage1.rank("u1", candidates, query_ts_ms=1000)
    ranked = rec.rank("u1", candidates, query_ts_ms=1000)
    assert ranked == stage1
    assert rec.n_llm_calls == 0
    assert rec.n_llm_skipped_gate == 1
    llm.invoke_ranking_json.assert_not_called()


def test_ltr_pool_rerank_merges_stage1_tail(monkeypatch) -> None:
    import numpy as np

    rec = _rerank_rec()
    rec._stage2_score = "ltr"
    rec._ltr_w = np.zeros(21)
    rec._ltr_mu = np.zeros(21)
    rec._ltr_sd = np.ones(21)
    rec._rerank_pool_k = 3
    rec.prepare_user_query("u1", 1000)

    rec._stage1.rank = MagicMock(return_value=["a", "b", "c", "d"])
    rec._stage1.score = MagicMock(return_value={"a": 3.0, "b": 2.0, "c": 1.0, "d": 0.0})
    monkeypatch.setattr(
        "emorecagent.tisasrec_align.rerank_recommender.score_pool_potential",
        lambda *args, **kwargs: MagicMock(),
    )
    monkeypatch.setattr(
        "emorecagent.tisasrec_align.rerank_recommender.score_pool_ltr",
        lambda pool, scored, **kwargs: {"a": 0.0, "b": 1.0, "c": 2.0},
    )

    ranked = rec.rank("u1", ["a", "b", "c", "d"], query_ts_ms=1000)
    assert ranked == ["c", "b", "a", "d"]
    assert rec.n_ltr_rerank == 1
    assert rec.n_stage1_only == 0


def test_ltr_without_weights_returns_stage1() -> None:
    rec = _rerank_rec()
    rec._stage2_score = "ltr"
    rec._ltr_w = rec._ltr_mu = rec._ltr_sd = None
    rec.prepare_user_query("u1", 1000)
    candidates = ["i1", "i2", "i3"]
    stage1 = rec._stage1.rank("u1", candidates, query_ts_ms=1000)
    ranked = rec.rank("u1", candidates, query_ts_ms=1000)
    assert ranked == stage1
    assert rec.n_stage1_only == 1


def test_fit_history_includes_valid_prefix_items() -> None:
    rec = _rerank_rec()
    rec.fit(
        [
            Interaction(user_id="u1", item="i1", rating=5.0, timestamp=500),
            Interaction(user_id="u1", item="i2", rating=5.0, timestamp=700),
        ]
    )
    rec._stage2_score = "ltr"
    rec._rebuild_ltr_stats(list(rec._train_by_user["u1"]))
    assert rec._history_items("u1", 1000) == ["i1", "i2"]
    assert rec._item_pop["i2"] == 1.0
