"""Tests for train-scoped ABSA target export."""

from __future__ import annotations

import json
from pathlib import Path

from emorecagent.absa.targets import build_train_scope, export_absa_targets
from emorecagent.data.types import Interaction
from emorecagent.eval.runner import load_split_jsonl


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


def test_build_train_scope_cutoff(tmp_path: Path) -> None:
    train = [
        Interaction("u1", "i1", 4.0, 100),
        Interaction("u2", "i2", 3.0, 200),
    ]
    scope = build_train_scope(train)
    assert scope.train_users == frozenset({"u1", "u2"})
    assert scope.train_items == frozenset({"i1", "i2"})
    assert scope.cutoff_ts == 200


def test_export_absa_targets_filters_by_train_scope(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    _write_jsonl(
        train_path,
        [
            {"user_id": "u1", "item": "i1", "rating": 5.0, "timestamp": 100},
            {"user_id": "u2", "item": "i2", "rating": 4.0, "timestamp": 200},
        ],
    )

    raw_path = tmp_path / "raw.jsonl"
    _write_jsonl(
        raw_path,
        [
            {
                "review_id": "r_in",
                "user_id": "u1",
                "parent_asin": "i1",
                "timestamp": 100,
                "text": "great product",
            },
            {
                "review_id": "r_late",
                "user_id": "u1",
                "parent_asin": "i1",
                "timestamp": 999,
                "text": "too late",
            },
            {
                "review_id": "r_out",
                "user_id": "u9",
                "parent_asin": "i9",
                "timestamp": 50,
                "text": "wrong user",
            },
            {
                "review_id": "r_dup",
                "user_id": "u2",
                "parent_asin": "i2",
                "timestamp": 150,
                "text": "first",
            },
            {
                "review_id": "r_dup",
                "user_id": "u2",
                "parent_asin": "i2",
                "timestamp": 180,
                "text": "duplicate id",
            },
        ],
    )

    out_path = tmp_path / "targets.jsonl"
    stats = export_absa_targets(
        train_path=train_path,
        raw_review_path=raw_path,
        out_path=out_path,
    )

    assert stats.n_train_interactions == 2
    assert stats.n_targets_written == 2
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    ids = {r["review_id"] for r in rows}
    assert ids == {"r_in", "r_dup"}
    assert load_split_jsonl(train_path)[0].user_id == "u1"
