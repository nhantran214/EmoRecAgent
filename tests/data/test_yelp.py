"""Yelp → Amazon-shaped JSONL conversion tests."""

from __future__ import annotations

import json
from pathlib import Path

from emorecagent.data.loader import load_interactions
from emorecagent.data.yelp import (
    convert_businesses,
    convert_reviews,
    yelp_date_to_timestamp_ms,
    yelp_review_to_amazon,
)


def test_yelp_date_to_timestamp_ms() -> None:
    assert yelp_date_to_timestamp_ms("2016-03-09") == 1457481600000
    assert yelp_date_to_timestamp_ms("2016-03-09 12:30:00") == 1457526600000
    assert yelp_date_to_timestamp_ms("") is None


def test_yelp_review_to_amazon_fields() -> None:
    row = yelp_review_to_amazon(
        {
            "review_id": "r1",
            "user_id": "u1",
            "business_id": "b1",
            "stars": 4,
            "date": "2016-03-09",
            "text": "Great food",
            "useful": 2,
        }
    )
    assert row is not None
    assert row["user_id"] == "u1"
    assert row["parent_asin"] == "b1"
    assert row["asin"] == "b1"
    assert row["rating"] == 4.0
    assert row["timestamp"] == 1457481600000
    assert row["helpful_vote"] == 2
    assert row["verified_purchase"] is False
    assert row["text"] == "Great food"
    assert row["review_id"] == "r1"


def test_convert_roundtrip_loadable(tmp_path: Path) -> None:
    review_src = tmp_path / "review.json"
    biz_src = tmp_path / "business.json"
    review_src.write_text(
        json.dumps(
            {
                "review_id": "r1",
                "user_id": "u1",
                "business_id": "b1",
                "stars": 5,
                "date": "2018-01-01",
                "text": "Love it",
                "useful": 1,
            }
        )
        + "\n"
        + json.dumps(
            {
                "review_id": "r2",
                "user_id": "u2",
                "business_id": "b2",
                "stars": 2,
                "date": "2018-02-01",
                "text": "Meh",
                "useful": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    biz_src.write_text(
        json.dumps(
            {
                "business_id": "b1",
                "name": "Cafe",
                "city": "Philadelphia",
                "categories": "Restaurants, Coffee & Tea",
                "stars": 4.5,
                "review_count": 10,
            }
        )
        + "\n"
        + json.dumps(
            {
                "business_id": "b2",
                "name": "Shop",
                "city": "Las Vegas",
                "categories": "Shopping",
                "stars": 3.0,
                "review_count": 3,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    meta_out = tmp_path / "meta.jsonl"
    review_out = tmp_path / "reviews.jsonl"
    n_meta, keep = convert_businesses(
        biz_src, meta_out, cities=["Philadelphia"], categories_substr=["Restaurants"]
    )
    assert n_meta == 1
    assert keep == {"b1"}
    n_rev = convert_reviews(review_src, review_out, keep_business_ids=keep)
    assert n_rev == 1

    interactions = load_interactions(review_out)
    assert len(interactions) == 1
    assert interactions[0].user_id == "u1"
    assert interactions[0].item == "b1"
    assert interactions[0].rating == 5.0


def test_stream_reviews_reads_native_yelp_dir(tmp_path: Path) -> None:
    """build_dataset can point review_path at the unpacked yelp_dataset directory."""
    from emorecagent.data.loader import stream_reviews
    from emorecagent.data.yelp import resolve_review_source

    yelp_dir = tmp_path / "yelp_dataset"
    yelp_dir.mkdir()
    (yelp_dir / "yelp_academic_dataset_review.json").write_text(
        json.dumps(
            {
                "review_id": "r1",
                "user_id": "u1",
                "business_id": "b1",
                "stars": 4,
                "date": "2016-03-09",
                "text": "Great food",
                "useful": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    resolved = resolve_review_source(yelp_dir)
    assert resolved.name == "yelp_academic_dataset_review.json"
    rows = list(stream_reviews(yelp_dir))
    assert len(rows) == 1
    assert rows[0]["parent_asin"] == "b1"
    assert rows[0]["rating"] == 4.0
    assert rows[0]["timestamp"] == 1457481600000
    interactions = load_interactions(yelp_dir)
    assert len(interactions) == 1
    assert interactions[0].item == "b1"
