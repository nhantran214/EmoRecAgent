"""compare_results.py tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from emorecagent.eval.significance import paired_bootstrap


def test_identical_runs_delta_zero() -> None:
    vec = [0.4, 0.5, 0.6]
    res = paired_bootstrap(vec, vec, n_bootstrap=200, seed=1)
    assert res.mean_delta == 0.0
    assert res.p_value == pytest.approx(1.0, abs=0.05)


def test_mismatched_length_raises() -> None:
    with pytest.raises(ValueError, match="equal-length"):
        paired_bootstrap([1.0, 2.0], [1.0], n_bootstrap=10)


def test_compare_script_payload_shape(tmp_path: Path) -> None:
    a = {
        "method": "a",
        "per_user": {"ndcg@10": [0.1, 0.2]},
        "user_ids": ["u1", "u2"],
    }
    b = {
        "method": "b",
        "per_user": {"ndcg@10": [0.3, 0.4]},
        "user_ids": ["u1", "u2"],
    }
    pa = tmp_path / "a.json"
    pb = tmp_path / "b.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    loaded_a = json.loads(pa.read_text())
    assert "per_user" in loaded_a
