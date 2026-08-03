"""Tests for HGT graph builder."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from emorecagent.data.types import Interaction
from emorecagent.hgt.graph_builder import build_hgt_graph


def _write_cache(path: Path, rows: list[tuple[str, list[dict]]]) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE absa_cache (review_id TEXT PRIMARY KEY, triples_json TEXT NOT NULL)"
    )
    for rid, triples in rows:
        conn.execute(
            "INSERT INTO absa_cache VALUES (?, ?)",
            (rid, json.dumps({"triples": triples})),
        )
    conn.commit()
    conn.close()


def test_toy_graph_counts(tmp_path):
    cache = tmp_path / "cache.sqlite"
    rid = "u1|i1|1000"
    _write_cache(
        cache,
        [(rid, [{"aspect": "scent", "sentiment": "positive", "confidence": 1.0}])],
    )
    raw = tmp_path / "reviews.jsonl"
    raw.write_text(
        json.dumps(
            {
                "user_id": "u1",
                "parent_asin": "i1",
                "timestamp": 1000,
                "text": "great scent",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    meta = tmp_path / "meta.jsonl"
    meta.write_text(
        json.dumps({"parent_asin": "i1", "title": "Soap", "description": "mild"})
        + "\n",
        encoding="utf-8",
    )
    train = [
        Interaction("u1", "i1", 5.0, 1000),
        Interaction("u1", "i2", 4.0, 2000),
        Interaction("u2", "i2", 5.0, 1500),
    ]
    bundle, vocab, stats = build_hgt_graph(
        train=train,
        cache_path=cache,
        meta_path=meta,
        raw_review_path=raw,
        aspect_top_k=10,
        min_aspect_support=1,
        text_encoder="hash",
        feature_dim=16,
        seed=0,
    )
    assert stats.n_users == 2
    assert stats.n_items == 2
    assert stats.n_train_pairs == 3
    assert bundle.edge_index.shape[1] >= 4
    assert int(bundle.edge_time.max()) < 240
    assert int(bundle.edge_time.min()) >= 0
    assert vocab.size >= 1
