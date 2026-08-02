"""Tests for RecBole .inter loading and timestamp filtering."""

from __future__ import annotations

from pathlib import Path

import pytest

from emorecagent.data.recbole_inter import (
    ACTSR_YELP_MAX_TIMESTAMP_S,
    ACTSR_YELP_MIN_TIMESTAMP_S,
    load_recbole_inter,
    resolve_inter_path,
)
from emorecagent.data.split import leave_last_out


def _write_inter(path: Path, rows: list[str]) -> Path:
    header = (
        "user_id:token\titem_id:token\trating:float\ttimestamp:float\t"
        "useful:float\tfunny:float\tcool:float\treview_id:token\n"
    )
    path.write_text(header + "".join(rows), encoding="utf-8")
    return path


def test_load_recbole_inter_filters_date_window(tmp_path: Path) -> None:
    # Inside closed 2019 window vs outside.
    in_ts = ACTSR_YELP_MIN_TIMESTAMP_S + 1000
    out_ts = ACTSR_YELP_MIN_TIMESTAMP_S - 1000
    inter = _write_inter(
        tmp_path / "yelp.inter",
        [
            f"u1\ti1\t5.0\t{out_ts}\t0\t0\t0\tr1\n",
            f"u1\ti2\t4.0\t{in_ts}\t0\t0\t0\tr2\n",
            f"u1\ti3\t3.0\t{ACTSR_YELP_MAX_TIMESTAMP_S + 1}\t0\t0\t0\tr3\n",
        ],
    )
    rows = load_recbole_inter(
        inter,
        min_timestamp_s=ACTSR_YELP_MIN_TIMESTAMP_S,
        max_timestamp_s=ACTSR_YELP_MAX_TIMESTAMP_S,
    )
    assert len(rows) == 1
    assert rows[0].item == "i2"
    assert rows[0].timestamp == in_ts * 1000


def test_resolve_inter_path_directory(tmp_path: Path) -> None:
    _write_inter(tmp_path / "yelp.inter", ["u1\ti1\t5.0\t1546265800\t0\t0\t0\tr1\n"])
    assert resolve_inter_path(tmp_path).name == "yelp.inter"


def test_missing_inter_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_inter_path(tmp_path / "missing.inter")


def test_bad_header_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.inter"
    p.write_text("foo:token\tbar:token\n1\t2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_recbole_inter(p)


def test_leave_last_out_after_inter_load(tmp_path: Path) -> None:
    base = ACTSR_YELP_MIN_TIMESTAMP_S
    inter = _write_inter(
        tmp_path / "yelp.inter",
        [
            f"u1\ta\t5.0\t{base + 1}\t0\t0\t0\tr1\n",
            f"u1\tb\t5.0\t{base + 2}\t0\t0\t0\tr2\n",
            f"u1\tc\t5.0\t{base + 3}\t0\t0\t0\tr3\n",
            f"u2\tx\t5.0\t{base + 1}\t0\t0\t0\tr4\n",  # <3 → train-only
        ],
    )
    rows = load_recbole_inter(
        inter,
        min_timestamp_s=ACTSR_YELP_MIN_TIMESTAMP_S,
        max_timestamp_s=ACTSR_YELP_MAX_TIMESTAMP_S,
    )
    split = leave_last_out(rows, min_history=0)
    assert [it.item for it in split.test] == ["c"]
    assert [it.item for it in split.valid] == ["b"]
    train_items = {it.item for it in split.train}
    assert "a" in train_items and "x" in train_items
