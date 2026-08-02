"""Tests for chronological valid eval cases."""

from __future__ import annotations

from emorecagent.data.types import Interaction
from emorecagent.hettisasrec.sequence_data import (
    build_test_eval_cases,
    build_valid_eval_cases,
)
from emorecagent.sequential.id_maps import IdMaps


def test_valid_case_history_excludes_future_valid_items():
    id_maps = IdMaps(
        user_to_idx={"u1": 1},
        item_to_idx={"i1": 1, "i2": 2, "i3": 3, "i4": 4},
    )
    train = [
        Interaction("u1", "i1", 5.0, 1000),
        Interaction("u1", "i2", 5.0, 2000),
    ]
    valid = [
        Interaction("u1", "i3", 5.0, 3000),
        Interaction("u1", "i4", 5.0, 4000),
    ]
    cases = build_valid_eval_cases(train, valid, id_maps)
    assert len(cases) == 2

    first = cases[0]
    assert first.gold_local == 3
    assert [i for i, _ in first.history] == [1, 2]

    second = cases[1]
    assert second.gold_local == 4
    assert [i for i, _ in second.history] == [1, 2, 3]


def test_test_case_history_includes_valid_but_not_future_test_items():
    id_maps = IdMaps(
        user_to_idx={"u1": 1},
        item_to_idx={"i1": 1, "i2": 2, "i3": 3, "i4": 4, "i5": 5},
    )
    train = [Interaction("u1", "i1", 5.0, 1000)]
    valid = [Interaction("u1", "i2", 5.0, 2000)]
    test = [
        Interaction("u1", "i3", 5.0, 3000),
        Interaction("u1", "i4", 5.0, 4000),
    ]
    cases = build_test_eval_cases(train, valid, test, id_maps)
    assert len(cases) == 2

    first = cases[0]
    assert first.gold_local == 3
    assert [i for i, _ in first.history] == [1, 2]

    second = cases[1]
    assert second.gold_local == 4
    assert [i for i, _ in second.history] == [1, 2, 3]
