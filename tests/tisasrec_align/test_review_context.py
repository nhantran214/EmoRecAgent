"""Tests for review text lookup (Amazon Reviews 2023 uses ``text`` field)."""

from __future__ import annotations

import json

from emorecagent.data.types import Interaction
from emorecagent.tisasrec_align.review_context import (
    load_review_text_index,
    prefix_reviews_for_user,
)


def test_load_review_text_index_reads_amazon2023_text_field(tmp_path):
    raw = tmp_path / "reviews.jsonl"
    raw.write_text(
        json.dumps(
            {
                "user_id": "U1",
                "parent_asin": "I1",
                "timestamp": 1000,
                "text": "Great moisturizer for dry skin.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    index = load_review_text_index(raw)
    assert index[("U1", "I1", 1000)] == "Great moisturizer for dry skin."


def test_prefix_reviews_for_user_uses_index(tmp_path):
    raw = tmp_path / "reviews.jsonl"
    rows = [
        {
            "user_id": "U1",
            "parent_asin": "I1",
            "timestamp": 1000,
            "text": "First purchase review.",
        },
        {
            "user_id": "U1",
            "parent_asin": "I2",
            "timestamp": 2000,
            "text": "Second item review.",
        },
    ]
    raw.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    index = load_review_text_index(raw)
    interactions = [
        Interaction("U1", "I1", 5.0, 1000),
        Interaction("U1", "I2", 4.0, 2000),
    ]
    revs = prefix_reviews_for_user("U1", interactions, 2000, index)
    assert len(revs) == 1
    assert revs[0].item_id == "I1"
    assert "First purchase" in revs[0].review_text
