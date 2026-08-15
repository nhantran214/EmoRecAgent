"""Metadata-derived T_u and item cards for Yelp_AC / ID-only Stage-2."""

from __future__ import annotations

from emorecagent.data.types import Interaction
from emorecagent.tisasrec_align.item_metadata import (
    ItemMeta,
    format_item_card,
    load_amazon_meta_jsonl,
    load_item_metadata,
    load_stage2_item_metadata,
)
from emorecagent.tisasrec_align.preference_text import (
    generate_preference_text_from_metadata,
)
from emorecagent.tisasrec_align.stage2_llm_rerank import build_candidate_cards


def test_load_item_metadata_roundtrip(tmp_path) -> None:
    path = tmp_path / "yelp.item"
    path.write_text(
        "item_id:token\titem_name:token_seq\tcategories:token_seq\tcity:token\tstate:token\n"
        "A\tAlpha Cafe\tFood, Coffee\tAustin\tTX\n"
        "B\tBeta Gym\tActive Life\tAustin\tTX\n",
        encoding="utf-8",
    )
    meta = load_item_metadata(path)
    assert meta["A"].name == "Alpha Cafe"
    assert "Coffee" in meta["A"].categories
    assert meta["B"].city == "Austin"


def test_metadata_tu_nonempty_with_categories() -> None:
    meta = {
        "A": ItemMeta(item_id="A", name="Alpha", categories="Food, Coffee"),
        "B": ItemMeta(item_id="B", name="Beta", categories="Bars"),
        "C": ItemMeta(item_id="C", name="Gamma", categories="Food"),
    }
    interactions = [
        Interaction(user_id="u1", item="A", rating=5.0, timestamp=100),
        Interaction(user_id="u1", item="B", rating=5.0, timestamp=200),
        Interaction(user_id="u1", item="C", rating=5.0, timestamp=300),
    ]
    result = generate_preference_text_from_metadata(
        user_id="u1",
        query_ts_ms=400,
        user_interactions=interactions,
        item_meta=meta,
    )
    assert result.has_reviews is True
    assert result.T_u
    assert "Food" in result.T_u or "Alpha" in result.T_u


def test_metadata_tu_empty_prefix() -> None:
    result = generate_preference_text_from_metadata(
        user_id="u1",
        query_ts_ms=50,
        user_interactions=[
            Interaction(user_id="u1", item="A", rating=5.0, timestamp=100),
        ],
        item_meta={},
    )
    assert result.has_reviews is False
    assert result.T_u == ""


def test_load_amazon_meta_jsonl_keep_ids(tmp_path) -> None:
    path = tmp_path / "meta_Beauty.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"parent_asin":"B001","title":"Serum A","categories":["Beauty","Skin"]}',
                '{"parent_asin":"B002","title":"Serum B","main_category":"Beauty"}',
                '{"asin":"B003","title":"Oil C","categories":[["Beauty","Hair"]]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    meta = load_amazon_meta_jsonl(path, keep_ids={"B001", "B003"})
    assert set(meta) == {"B001", "B003"}
    assert meta["B001"].name == "Serum A"
    assert "Skin" in meta["B001"].categories
    assert "Hair" in meta["B003"].categories
    # Auto loader picks JSONL by suffix.
    auto = load_stage2_item_metadata(path, keep_ids={"B002"})
    assert auto["B002"].name == "Serum B"


def test_candidate_cards_include_metadata() -> None:
    meta = {
        "A": ItemMeta(item_id="A", name="Alpha Cafe", categories="Food, Coffee, Nightlife"),
    }
    cards = build_candidate_cards(
        ["A", "B"],
        {"A": 0.9, "B": 0.1},
        meta,
        stage1_ranks={"A": 1, "B": 2},
    )
    assert "name=Alpha Cafe" in cards
    assert "cats=Food, Coffee, Nightlife" in cards or "cats=Food, Coffee" in cards
    assert "B | S=0.1000" in cards
    # Titles must not drop Stage-1 rank (regression that hurt Beauty Stage-2).
    assert "π¹_rank=1" in cards
    assert "π¹_rank=2" in cards
    assert format_item_card("A", 0.5, meta["A"]).startswith("A | S=0.5000")
