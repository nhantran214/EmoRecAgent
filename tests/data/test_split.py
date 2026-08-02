"""Tests for chronological split and sampling."""

from __future__ import annotations

import json

from emorecagent.data.split import (
    chronological_split,
    leave_last_out,
    sample_subset,
    write_split,
    _write_jsonl,
)
from emorecagent.data.types import Interaction
from emorecagent.eval.runner import load_split_jsonl


def _mk(user: str, item: str, ts: int, verified: bool = False):
    return Interaction(user, item, 5.0, ts, 0, verified_purchase=verified)


def test_chronological_split_80_10_10_per_user():
    # 10 interactions → 8 train, 1 valid, 1 test
    data = [_mk("u1", f"i{j}", j * 10) for j in range(10)]
    split = chronological_split(data, min_history=0)
    assert len(split.train) == 8
    assert len(split.valid) == 1
    assert len(split.test) == 1
    assert [i.item for i in split.train] == [f"i{j}" for j in range(8)]
    assert split.valid[0].item == "i8"
    assert split.test[0].item == "i9"
    assert split.manifest["split_method"] == "chronological_ratio"


def test_chronological_split_preserves_time_order_no_leakage():
    data = [_mk("u1", f"i{j}", j) for j in range(10)]
    split = chronological_split(data, min_history=0)
    max_train_ts = max(i.timestamp for i in split.train)
    min_valid_ts = min(i.timestamp for i in split.valid)
    max_valid_ts = max(i.timestamp for i in split.valid)
    min_test_ts = min(i.timestamp for i in split.test)
    assert max_train_ts < min_valid_ts
    assert max_valid_ts < min_test_ts


def test_chronological_split_no_overlap_between_partitions():
    data = [_mk("u1", f"i{j}", j) for j in range(10)]
    split = chronological_split(data, min_history=0)
    train_keys = {(i.user_id, i.item) for i in split.train}
    valid_keys = {(i.user_id, i.item) for i in split.valid}
    test_keys = {(i.user_id, i.item) for i in split.test}
    assert train_keys.isdisjoint(valid_keys)
    assert train_keys.isdisjoint(test_keys)
    assert valid_keys.isdisjoint(test_keys)


def test_chronological_split_short_users_are_train_only():
    data = [_mk("u1", f"i{j}", j) for j in range(5)]  # 80%→4 train, 0 valid
    split = chronological_split(data, min_history=0)
    assert split.valid == [] and split.test == []
    assert len(split.train) == 5


def test_chronological_split_min_history_excludes_users():
    short = [_mk("us", f"i{j}", j) for j in range(10)]  # train=8 ok if min_history<=8
    split = chronological_split(short, min_history=9)
    assert split.test == []
    assert len(split.train) == 10

    long = [_mk("ul", f"j{j}", j) for j in range(20)]  # train=16 >= 9
    split = chronological_split(long, min_history=9)
    assert {i.user_id for i in split.test} == {"ul"}


def test_chronological_split_ratios_must_sum_to_one():
    import pytest

    data = [_mk("u1", "i1", 1)]
    with pytest.raises(ValueError, match="sum to 1.0"):
        chronological_split(data, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.2)


def test_leave_last_out_assigns_one_test_one_valid():
    data = [
        _mk("u1", "i1", 10),
        _mk("u1", "i2", 20),
        _mk("u1", "i3", 30),
        _mk("u1", "i4", 40),
    ]
    split = leave_last_out(data, min_history=0)
    assert [i.item for i in split.test] == ["i4"]   # latest
    assert [i.item for i in split.valid] == ["i3"]  # second latest
    assert {i.item for i in split.train} == {"i1", "i2"}


def test_no_test_leakage_into_train():
    data = [_mk("u1", "i1", 1), _mk("u1", "i2", 2), _mk("u1", "i3", 3)]
    split = leave_last_out(data, min_history=0)
    train_keys = {(i.user_id, i.item) for i in split.train}
    test_keys = {(i.user_id, i.item) for i in split.test}
    valid_keys = {(i.user_id, i.item) for i in split.valid}
    assert train_keys.isdisjoint(test_keys)
    assert train_keys.isdisjoint(valid_keys)


def test_users_with_few_interactions_are_train_only():
    data = [_mk("u1", "i1", 1), _mk("u1", "i2", 2)]  # only 2 -> no eval
    split = leave_last_out(data, min_history=0)
    assert split.test == [] and split.valid == []
    assert len(split.train) == 2
    assert split.manifest["n_test_users"] == 0


def test_min_history_excludes_short_history_users():
    # u_short: 3 interactions -> train would be 1, below min_history=3 -> train-only
    short = [_mk("us", f"i{j}", j) for j in range(3)]
    # u_long: 6 interactions -> train 4 >= 3 -> eligible
    long = [_mk("ul", f"j{j}", j) for j in range(6)]
    split = leave_last_out(short + long, min_history=3)
    test_users = {i.user_id for i in split.test}
    assert test_users == {"ul"}
    assert split.manifest["n_test_users"] == 1


def test_sample_subset_dual_cap_survives_sparse_long_tail():
    # Hub items are dense; private items are degree-1 long tail.
    data = []
    hubs = [f"hub{j}" for j in range(20)]
    for u in range(100):
        for j in range(5):
            data.append(_mk(f"u{u}", hubs[j], u * 20 + j))
        for j in range(5):
            data.append(_mk(f"u{u}", f"priv{u}_{j}", u * 20 + j + 5))
    out = sample_subset(data, k_core=5, max_users=50, max_items=30, seed=42)
    assert out, "degree-ranked caps should preserve a non-empty 5-core"


def test_sample_subset_is_deterministic_and_restores_kcore():
    # 4 users x 3 items fully connected -> a 3-core.
    data = [
        _mk(u, it, ts=1)
        for u in ("u1", "u2", "u3", "u4")
        for it in ("i1", "i2", "i3")
    ]
    a = sample_subset(data, k_core=3, max_users=3, max_items=None, seed=42)
    b = sample_subset(data, k_core=3, max_users=3, max_items=None, seed=42)
    assert [(i.user_id, i.item) for i in a] == [(i.user_id, i.item) for i in b]
    # Sampling to 3 users keeps the 3-core property (each survivor degree >= 3).
    assert len({i.user_id for i in a}) <= 3
    assert a, "expected a non-empty 3-core after sampling"


def test_verified_purchase_round_trip(tmp_path) -> None:
    rows = [_mk("u1", "i1", 10, verified=True), _mk("u1", "i2", 20, verified=False)]
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, rows)
    loaded = load_split_jsonl(path)
    assert loaded[0].verified_purchase is True
    assert loaded[1].verified_purchase is False


def test_legacy_jsonl_missing_verified_defaults_false(tmp_path) -> None:
    path = tmp_path / "legacy.jsonl"
    path.write_text(
        '{"user_id":"u1","item":"i1","rating":5.0,"timestamp":1,"helpful_vote":0}\n'
    )
    loaded = load_split_jsonl(path)
    assert loaded[0].verified_purchase is False


def test_write_split_emits_files_and_manifest(tmp_path):
    data = [_mk("u1", f"i{j}", j) for j in range(4)]
    split = leave_last_out(data, min_history=0)
    manifest_path = write_split(tmp_path, split, seed=42, k_core=5)
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "valid.jsonl").exists()
    assert (tmp_path / "test.jsonl").exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["seed"] == 42
    assert manifest["k_core"] == 5
    assert manifest["n_test"] == 1
    assert "pct_verified_test" in manifest
    line = (tmp_path / "test.jsonl").read_text()
    assert '"verified_purchase"' in line
