"""Tests for aspect vocabulary from ABSA cache."""

from __future__ import annotations

import json
import sqlite3

import pytest

from emorecagent.hgt.aspect_vocab import (
    AspectVocab,
    build_aspect_vocab,
    count_aspects_from_cache,
)
from emorecagent.hgt.schema import OTHER_ASPECT


def _write_cache(path, rows: list[tuple[str, list[dict]]]) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE absa_cache (
            review_id TEXT PRIMARY KEY,
            triples_json TEXT NOT NULL
        )
        """
    )
    for rid, triples in rows:
        payload = json.dumps({"triples": triples})
        conn.execute(
            "INSERT INTO absa_cache (review_id, triples_json) VALUES (?, ?)",
            (rid, payload),
        )
    conn.commit()
    conn.close()


def test_build_aspect_vocab_top_k_and_other(tmp_path):
    cache = tmp_path / "cache.sqlite"
    triples = [
        ("r1", [{"aspect": "scent", "sentiment": "positive"}] * 6),
        ("r2", [{"aspect": "comfort", "sentiment": "negative"}] * 4),
        ("r3", [{"aspect": "rare_aspect", "sentiment": "neutral"}]),
    ]
    _write_cache(cache, triples)
    vocab = build_aspect_vocab(cache, top_k=2, min_support=1)
    assert "scent" in vocab.aspects
    assert OTHER_ASPECT in vocab.aspects
    assert vocab.id_for("rare_aspect") == vocab.other_id


def test_empty_cache_raises(tmp_path):
    cache = tmp_path / "empty.sqlite"
    _write_cache(cache, [])
    with pytest.raises(ValueError, match="No aspects"):
        build_aspect_vocab(cache, top_k=5, min_support=1)


def test_vocab_id_for_known_aspect():
    vocab = AspectVocab(aspects=("scent", OTHER_ASPECT), other_id=1)
    assert vocab.id_for("scent") == 0
    assert vocab.id_for("unknown") == 1


def test_count_respects_min_support(tmp_path):
    cache = tmp_path / "cache.sqlite"
    _write_cache(
        cache,
        [
            ("r1", [{"aspect": "scent", "sentiment": "positive"}]),
            ("r2", [{"aspect": "scent", "sentiment": "positive"}]),
            ("r3", [{"aspect": "comfort", "sentiment": "positive"}]),
        ],
    )
    counts = count_aspects_from_cache(cache, min_support=2)
    assert counts["scent"] == 2
    assert "comfort" not in counts
