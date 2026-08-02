"""Tests for dedup and k-core filtering."""

from __future__ import annotations

from emorecagent.data.kcore import k_core_filter, k_core_summary
from emorecagent.data.loader import dedup_earliest
from emorecagent.data.types import Interaction


def _mk(user: str, item: str, ts: int, rating: float = 5.0, helpful: int = 0):
    return Interaction(user, item, rating, ts, helpful)


def test_dedup_keeps_earliest():
    data = [
        _mk("u1", "i1", ts=200, rating=4.0),
        _mk("u1", "i1", ts=100, rating=2.0),  # earlier -> wins
        _mk("u1", "i2", ts=150),
    ]
    out = {(i.user_id, i.item): i for i in dedup_earliest(data)}
    assert out[("u1", "i1")].timestamp == 100
    assert out[("u1", "i1")].rating == 2.0
    assert len(out) == 2


def test_kcore_k1_is_noop():
    data = [_mk("u1", "i1", 1), _mk("u2", "i2", 2)]
    assert len(k_core_filter(data, 1)) == 2


def test_kcore_removes_low_degree_until_stable():
    # Build a 2-core: users u1,u2,u3 each interact with items i1,i2 (degree 2),
    # plus one sparse edge (u4,i3) that must be pruned.
    data = []
    for u in ("u1", "u2", "u3"):
        for it in ("i1", "i2"):
            data.append(_mk(u, it, ts=1))
    data.append(_mk("u4", "i3", ts=1))  # u4 deg 1, i3 deg 1 -> removed

    kept = k_core_filter(data, 2)
    users = {i.user_id for i in kept}
    items = {i.item for i in kept}
    assert "u4" not in users and "i3" not in items
    assert users == {"u1", "u2", "u3"} and items == {"i1", "i2"}
    # Every surviving node has degree >= 2.
    assert len(kept) == 6


def test_kcore_5_requires_five_reviews_per_user_and_item():
    """5-core: every surviving user/item has degree >= 5."""
    data = []
    # Dense 5x5 block: 5 users each review 5 items -> all survive 5-core.
    users = [f"u{i}" for i in range(5)]
    items = [f"i{j}" for j in range(5)]
    for u in users:
        for it in items:
            data.append(_mk(u, it, ts=1))
    # Sparse outliers must be pruned.
    data.append(_mk("u_sparse", "i0", ts=2))
    data.append(_mk("u0", "i_sparse", ts=2))

    kept = k_core_filter(data, 5)
    summary = k_core_summary(kept, 5)
    assert summary.min_user_degree >= 5
    assert summary.min_item_degree >= 5
    assert summary.n_users == 5
    assert summary.n_items == 5
    assert "u_sparse" not in {i.user_id for i in kept}
    assert "i_sparse" not in {i.item for i in kept}


def test_kcore_cascade_can_empty():
    # A chain too sparse for 5-core collapses entirely.
    data = [_mk("u1", "i1", 1), _mk("u1", "i2", 2), _mk("u2", "i1", 3)]
    assert k_core_filter(data, 5) == []
