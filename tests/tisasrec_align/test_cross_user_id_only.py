"""ID-only cross-user co-visit lookup (no review text gate)."""

from __future__ import annotations

from emorecagent.data.types import Interaction
from emorecagent.tisasrec_align.cross_user_lookup import (
    build_lookup_from_config,
    build_lookup_id_only,
    lookup_co_items,
)


def test_id_only_lookup_uses_all_visited_anchors() -> None:
    train = [
        Interaction(user_id="u1", item="A", rating=5.0, timestamp=100),
        Interaction(user_id="u1", item="X", rating=5.0, timestamp=200),
        Interaction(user_id="u1", item="Y", rating=5.0, timestamp=300),
    ]
    # No review text — review_text mode would yield empty; id_only should not.
    lookup = build_lookup_id_only(train)
    assert lookup["A"]["X"] == 1
    assert lookup["A"]["Y"] == 1
    assert lookup["X"]["A"] == 1


def test_id_only_via_config_mode() -> None:
    train = [
        Interaction(user_id="u1", item="A", rating=5.0, timestamp=100),
        Interaction(user_id="u1", item="B", rating=5.0, timestamp=200),
    ]
    lookup = build_lookup_from_config(train, "/nonexistent", mode="id_only")
    assert lookup["A"]["B"] == 1


def test_id_only_boosts_co_visited_pool_items() -> None:
    lookup = {"A": {"X": 4, "Y": 2}}
    scores = lookup_co_items(["A"], {"X", "Z"}, lookup)
    assert scores["X"] == 1.0
    assert "Z" not in scores
    assert lookup_co_items(["Z"], {"X"}, lookup) == {}
