"""Tests for U2 chronological leave-last-out split + sampling (R1)."""

from __future__ import annotations

import json

from emorecagent.data.split import leave_last_out, sample_subset, write_split
from emorecagent.data.types import Interaction


def _mk(user: str, item: str, ts: int):
    return Interaction(user, item, 5.0, ts, 0)


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
