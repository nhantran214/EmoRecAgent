"""Tests for U2 dedup + k-core filtering (R1)."""

from __future__ import annotations

from emorecagent.data.kcore import k_core_filter
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


def test_kcore_cascade_can_empty():
    # A chain too sparse for 5-core collapses entirely.
    data = [_mk("u1", "i1", 1), _mk("u1", "i2", 2), _mk("u2", "i1", 3)]
    assert k_core_filter(data, 5) == []
