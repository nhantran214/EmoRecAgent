"""Tests for KG loaders (train-scoped hydration)."""

from __future__ import annotations

import json

from emorecagent.absa.cache import AbsaCache
from emorecagent.data.types import Interaction
from emorecagent.kg.loaders import load_train_kg
from emorecagent.kg.memory import InMemoryKG
from emorecagent.llm.schemas import AbsaTriple, TripleSet


def test_load_train_kg_ignores_valid_test_paths(tmp_path) -> None:
    split = tmp_path / "split"
    split.mkdir()
    train = [
        Interaction("u1", "i1", 5.0, 1000, verified_purchase=True),
        Interaction("u2", "i2", 4.0, 2000),
    ]
    valid = [Interaction("u3", "i3", 3.0, 3000)]
    test = [Interaction("u4", "i4", 2.0, 4000)]
    for name, rows in (("train", train), ("valid", valid), ("test", test)):
        path = split / f"{name}.jsonl"
        with path.open("w") as fh:
            for it in rows:
                fh.write(
                    json.dumps(
                        {
                            "user_id": it.user_id,
                            "item": it.item,
                            "rating": it.rating,
                            "timestamp": it.timestamp,
                            "helpful_vote": it.helpful_vote,
                            "verified_purchase": it.verified_purchase,
                        }
                    )
                    + "\n"
                )

    kg = InMemoryKG()
    stats = load_train_kg(
        kg,
        train_path=split / "train.jsonl",
        raw_review_path=split / "missing_raw.jsonl",
        cache_path=split / "missing_cache.sqlite",
        helpful_cap=10,
    )
    assert stats["interactions"] == 2
    assert len(kg.interactions) == 2
    assert kg.interactions[0].verified_purchase is True


def test_load_train_kg_loads_absa_triples(tmp_path) -> None:
    split = tmp_path / "split"
    split.mkdir()
    train_path = split / "train.jsonl"
    train_path.write_text(
        '{"user_id":"u1","item":"i1","rating":5,"timestamp":1000,'
        '"helpful_vote":2,"verified_purchase":true}\n'
    )
    raw_path = split / "raw.jsonl"
    raw_path.write_text(
        json.dumps(
            {
                "user_id": "u1",
                "parent_asin": "i1",
                "timestamp": 1000,
                "text": "great scent",
            }
        )
        + "\n"
    )
    cache_path = split / "cache.sqlite"
    cache = AbsaCache(cache_path)
    rid = "u1|i1|1000"
    cache.put(
        rid,
        TripleSet(
            triples=[
                AbsaTriple(
                    aspect="scent", opinion="great", sentiment="positive", confidence=0.9
                )
            ]
        ),
    )
    cache.close()

    kg = InMemoryKG()
    stats = load_train_kg(
        kg,
        train_path=train_path,
        raw_review_path=raw_path,
        cache_path=cache_path,
        helpful_cap=10,
    )
    assert stats["reviews_with_triples"] == 1
    assert stats["sentiment_edges"] >= 1
    assert kg.get_item_aspects("i1").get("scent") is not None
