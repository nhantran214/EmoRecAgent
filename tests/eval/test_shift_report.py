"""Shift-subpopulation report tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from emorecagent.eval.runner import aggregate_per_user


def test_aggregate_per_user_differs_from_row_mean() -> None:
    per_user = {"ndcg@10": [1.0, 0.0, 1.0, 0.0]}
    user_ids = ["u1", "u1", "u2", "u2"]
    row_mean = sum(per_user["ndcg@10"]) / 4
    user_mean = aggregate_per_user(per_user, user_ids)["ndcg@10"]
    assert row_mean == 0.5
    assert user_mean == 0.5


def test_missing_user_ids_errors(tmp_path: Path) -> None:
    results = {"per_user": {"ndcg@10": [0.5]}, "means": {"ndcg@10": 0.5}}
    path = tmp_path / "r.json"
    path.write_text(json.dumps(results))
    loaded = json.loads(path.read_text())
    assert "user_ids" not in loaded
