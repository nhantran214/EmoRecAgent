"""Unit tests for complementary item-potential φ(u,i)."""

from __future__ import annotations

import numpy as np

from emorecagent.tisasrec_align.item_metadata import ItemMeta
from emorecagent.tisasrec_align.item_potential import (
    greedy_potential_swaps,
    ltr_feature_matrix,
    oof_listwise_scores,
    rank_penalty,
    score_pool_potential,
    swap_delta,
    zscore_map,
)


def test_zscore_map_standardizes_and_handles_constant() -> None:
    z = zscore_map({"a": 1.0, "b": 3.0})
    assert abs(z["a"] + z["b"]) < 1e-9
    assert z["b"] > 0
    assert zscore_map({"a": 2.0, "b": 2.0}) == {"a": 0.0, "b": 0.0}


def test_rank_penalty_higher_for_top_slots() -> None:
    assert rank_penalty(1, 20) > rank_penalty(10, 20) > rank_penalty(20, 20)
    assert rank_penalty(1, 20) == 1.0
    assert rank_penalty(20, 20) == 1.0 / 20.0


def test_swap_delta_requires_gap_beyond_head_penalty() -> None:
    # Same φ, rank-1 incumbent → negative Δ (do not evict).
    assert swap_delta(0.0, 0.0, 1, tau=0.25, head_n=20) < 0
    # Large φ gap vs rank-20 incumbent → positive.
    assert swap_delta(1.0, -1.0, 20, tau=0.25, head_n=20) > 0


def test_score_pool_text_and_hist_rank_matching_item_higher() -> None:
    pool = ["serum", "lipstick", "phone"]
    meta = {
        "serum": ItemMeta(item_id="serum", name="Vitamin C Serum", categories="Skincare"),
        "lipstick": ItemMeta(item_id="lipstick", name="Red Lipstick", categories="Makeup"),
        "phone": ItemMeta(item_id="phone", name="Phone Case", categories="Electronics"),
    }
    # Orthogonal item embs; history = serum-like vector.
    embs = {
        "serum": np.array([1.0, 0.0, 0.0]),
        "lipstick": np.array([0.0, 1.0, 0.0]),
        "phone": np.array([0.0, 0.0, 1.0]),
    }
    p_u = np.array([1.0, 0.0, 0.0])
    seq = {"serum": 0.1, "lipstick": 0.9, "phone": 0.5}  # Stage-1 prefers lipstick
    lookup = {"serum": {"serum": 1}}  # unused co (self not in lookup_co_items)
    out = score_pool_potential(
        pool,
        t_u="wants vitamin C serum for dry skin",
        p_u=p_u,
        item_embs=embs,
        seq_scores=seq,
        history_items=["serum"],
        anchor_items=["serum"],
        lookup=lookup,
        item_meta=meta,
        review_snippets={"serum": ["this serum fixed my dry skin"]},
        weights={"text": 0.45, "co": 0.0, "hist": 0.45, "seq": 0.10},
    )
    assert out.phi["serum"] > out.phi["lipstick"]
    assert out.phi["serum"] > out.phi["phone"]


def test_cold_start_drops_empty_channels_uses_seq() -> None:
    pool = ["a", "b"]
    embs = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    out = score_pool_potential(
        pool,
        t_u="",
        p_u=None,
        item_embs=embs,
        seq_scores={"a": 2.0, "b": 0.0},
        history_items=[],
        anchor_items=[],
        lookup={},
        item_meta=None,
        review_snippets=None,
    )
    assert "seq" in out.weights_used
    assert "text" not in out.weights_used
    assert "hist" not in out.weights_used
    assert out.phi["a"] > out.phi["b"]


def test_greedy_swaps_gold_from_tail_into_low_phi_head() -> None:
    # π¹: a,b in head; gold g at rank 4. φ(g) >> φ(b).
    order = ["a", "b", "c", "g"]
    phi = {"a": 1.0, "b": -1.0, "c": 0.0, "g": 2.0}
    new, n = greedy_potential_swaps(
        order,
        phi,
        head_n=2,
        focus_k=4,
        tau=0.1,
        gamma=0.0,
        max_swaps=3,
    )
    assert n >= 1
    assert new[0] == "a" or new[1] == "g" or new[0] == "g"
    assert "g" in new[:2]


def test_backbone_residual_alpha_zero_copies_seq_order() -> None:
    pool = ["a", "b", "c"]
    embs = {
        "a": np.array([1.0, 0.0]),
        "b": np.array([0.0, 1.0]),
        "c": np.array([0.5, 0.5]),
    }
    scored = score_pool_potential(
        pool,
        t_u="serum",
        p_u=None,
        item_embs=embs,
        seq_scores={"a": 3.0, "b": 2.0, "c": 1.0},
        history_items=[],
        anchor_items=[],
        lookup={},
        item_meta={
            "a": ItemMeta(item_id="a", name="A", categories="Skincare"),
            "b": ItemMeta(item_id="b", name="B", categories="Makeup"),
            "c": ItemMeta(item_id="c", name="C", categories="Hair"),
        },
    )
    from emorecagent.tisasrec_align.item_potential import mix_backbone_residual

    phi0 = mix_backbone_residual(pool, scored, alpha=0.0)
    assert phi0["a"] > phi0["b"] > phi0["c"]


def test_rerank_pool_by_phi_keeps_ties_in_stage1_order() -> None:
    from emorecagent.tisasrec_align.item_potential import rerank_pool_by_phi

    pool = ["a", "b", "c"]
    assert rerank_pool_by_phi(pool, {"a": 1.0, "b": 2.0, "c": 2.0}) == ["b", "c", "a"]


def test_markov_prefers_observed_successor() -> None:
    from types import SimpleNamespace

    from emorecagent.tisasrec_align.item_potential import (
        build_next_item_lookup,
        markov_scores,
    )

    train = [
        SimpleNamespace(user_id="u1", timestamp=1, item="serum"),
        SimpleNamespace(user_id="u1", timestamp=2, item="moisturizer"),
        SimpleNamespace(user_id="u2", timestamp=1, item="serum"),
        SimpleNamespace(user_id="u2", timestamp=2, item="moisturizer"),
        SimpleNamespace(user_id="u3", timestamp=1, item="serum"),
        SimpleNamespace(user_id="u3", timestamp=2, item="lipstick"),
    ]
    lookup = build_next_item_lookup(train)
    scores = markov_scores(
        ["moisturizer", "lipstick", "phone"],
        ["serum"],
        lookup,
        last_k=1,
    )
    assert scores["moisturizer"] > scores["lipstick"]
    assert scores["phone"] == 0.0


def test_augment_v1_cache_adds_three_cols() -> None:
    from emorecagent.tisasrec_align.item_potential import augment_v1_cache_features

    X = np.zeros((6, 14), dtype=np.float64)
    X[:, 2] = [3, 2, 1, 3, 2, 1]
    ranks = np.array([1, 2, 3, 1, 2, 3])
    groups = np.array([0, 0, 0, 1, 1, 1])
    out = augment_v1_cache_features(X, ranks, groups)
    assert out.shape == (6, 17)
    from emorecagent.tisasrec_align.item_potential import rerank_pool_by_phi

    pool = ["a", "b", "c"]
    assert rerank_pool_by_phi(pool, {"a": 1.0, "b": 2.0, "c": 2.0}) == ["b", "c", "a"]


def test_listwise_oof_uses_rank_backbone() -> None:
    """Gold always π¹ rank-1 → OOF listwise should rank it first."""
    rng = np.random.RandomState(0)
    rows = []
    y = []
    groups = []
    gid = 0
    for u in range(12):
        pool = [f"u{u}_i{i}" for i in range(8)]
        scored = score_pool_potential(
            pool,
            t_u="",
            p_u=None,
            item_embs={p: rng.normal(size=4) for p in pool},
            seq_scores={p: float(8 - i) for i, p in enumerate(pool)},
            history_items=[],
            anchor_items=[],
            lookup={},
        )
        X = ltr_feature_matrix(pool, scored)
        rows.append(X)
        labels = np.zeros(len(pool), dtype=np.float64)
        labels[0] = 1.0
        y.append(labels)
        groups.append(np.full(len(pool), gid))
        gid += 1
    X = np.vstack(rows)
    yv = np.concatenate(y)
    gv = np.concatenate(groups)
    oof = oof_listwise_scores(X, yv, gv, n_splits=4, l2=0.05, seed=0)
    # Per user, argmax OOF should be the gold row (index 0 in each block of 8).
    hits = 0
    for u in range(12):
        sl = oof[u * 8 : (u + 1) * 8]
        if int(np.argmax(sl)) == 0:
            hits += 1
    assert hits >= 10


def test_listwise_can_use_text_when_rank_misleading() -> None:
    """Gold at rank 4 with high z_text; rank-1 has low z_text."""
    rng = np.random.RandomState(1)
    rows = []
    y = []
    groups = []
    for u in range(16):
        pool = [f"u{u}_i{i}" for i in range(6)]
        seq = {p: float(6 - i) for i, p in enumerate(pool)}
        embs = {p: rng.normal(size=3) for p in pool}
        meta = {
            pool[0]: ItemMeta(item_id=pool[0], name="Random Lipstick", categories="Makeup"),
            pool[3]: ItemMeta(
                item_id=pool[3], name="Vitamin C Serum Dry Skin", categories="Skincare"
            ),
        }
        for p in pool:
            meta.setdefault(p, ItemMeta(item_id=p, name="Other", categories="Misc"))
        scored = score_pool_potential(
            pool,
            t_u="wants vitamin C serum for dry skin",
            p_u=None,
            item_embs=embs,
            seq_scores=seq,
            history_items=[],
            anchor_items=[],
            lookup={},
            item_meta=meta,
            review_snippets={pool[3]: ["serum fixed dry skin"]},
        )
        X = ltr_feature_matrix(pool, scored)
        rows.append(X)
        labels = np.zeros(len(pool), dtype=np.float64)
        labels[3] = 1.0
        y.append(labels)
        groups.append(np.full(len(pool), u))
    oof = oof_listwise_scores(
        np.vstack(rows),
        np.concatenate(y),
        np.concatenate(groups),
        n_splits=4,
        l2=0.01,
        seed=1,
    )
    hits = sum(
        int(np.argmax(oof[u * 6 : (u + 1) * 6]) == 3) for u in range(16)
    )
    # Residual text should beat always picking π¹ rank-1 on this synthetic set.
    assert hits >= 8
