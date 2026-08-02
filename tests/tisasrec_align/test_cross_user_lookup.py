"""Tests for cross-user lookup build and query."""

from __future__ import annotations

import json

from emorecagent.data.types import Interaction
from emorecagent.tisasrec_align.cross_user_lookup import (
    build_lookup_from_train,
    load_lookup,
    lookup_co_items,
    save_lookup,
)


def test_build_lookup_reviewed_anchor_co_purchases() -> None:
    train = [
        Interaction(user_id="u1", item="A", rating=5.0, timestamp=100),
        Interaction(user_id="u1", item="X", rating=5.0, timestamp=200),
        Interaction(user_id="u1", item="Y", rating=5.0, timestamp=300),
    ]
    review_index = {("u1", "A", 100): "great product"}
    lookup = build_lookup_from_train(train, review_index)
    assert "A" in lookup
    assert lookup["A"]["X"] == 1
    assert lookup["A"]["Y"] == 1


def test_lookup_co_items_empty_without_evidence() -> None:
    lookup = {"A": {"X": 3}}
    assert lookup_co_items(["Z"], {"X", "Y"}, lookup) == {}


def test_lookup_co_items_normalizes_scores() -> None:
    lookup = {"A": {"X": 4, "Y": 2}}
    scores = lookup_co_items(["A"], {"X", "Y", "Z"}, lookup)
    assert scores["X"] == 1.0
    assert scores["Y"] == 0.5
    assert "Z" not in scores


def test_save_and_load_lookup_roundtrip(tmp_path) -> None:
    lookup = {"A": {"X": 2}}
    path = tmp_path / "lookup.json"
    save_lookup(path, lookup)
    loaded = load_lookup(path)
    assert loaded == lookup


def test_load_lookup_missing_returns_empty(tmp_path) -> None:
    assert load_lookup(tmp_path / "missing.json") == {}


def test_load_lookup_skips_bad_entries(tmp_path) -> None:
    path = tmp_path / "lookup.json"
    path.write_text(json.dumps({"A": {"X": 1}, "bad": "nope"}), encoding="utf-8")
    loaded = load_lookup(path)
    assert loaded == {"A": {"X": 1}}
