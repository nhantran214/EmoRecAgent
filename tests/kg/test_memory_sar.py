"""Tests for InMemoryKG aspect-strong recall index."""

from __future__ import annotations

from emorecagent.kg.memory import InMemoryKG


def test_items_strong_on_returns_high_aspect_items() -> None:
    kg = InMemoryKG()
    kg.upsert_item_sentiment("i_strong", "comfort", 0.9, 5, ts=1)
    kg.upsert_item_sentiment("i_weak", "comfort", 0.1, 5, ts=1)
    out = kg.items_strong_on(["comfort"], limit=5, exclude=set(), tau=0.65, min_support=3)
    assert out == ["i_strong"]
