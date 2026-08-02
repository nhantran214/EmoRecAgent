"""Regression tests for indexed InMemoryKG lookups."""

from __future__ import annotations

import time

from emorecagent.kg.memory import InMemoryKG


def test_item_aspect_lookup_uses_index():
    kg = InMemoryKG()
    for i in range(500):
        for j in range(20):
            kg.upsert_item_sentiment(f"i{i}", f"aspect_{j}", 0.5, 1, ts=1)

    t0 = time.monotonic()
    for i in range(500):
        aspects = kg.get_item_aspects(f"i{i}")
        assert aspects
    elapsed = time.monotonic() - t0
    assert elapsed < 0.5
