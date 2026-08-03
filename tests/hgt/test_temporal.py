"""Tests for pyHGT relative temporal encoding."""

from __future__ import annotations

import numpy as np

from emorecagent.hgt.temporal import (
    RTE_MAX_LEN,
    assert_rte_edge_time,
    build_node_day_times,
    edge_time_from_nodes,
    encode_relational_time,
    timestamp_to_day_index,
)


def test_encode_relational_time_centered_at_zero_delta() -> None:
    assert encode_relational_time(10, 10) == 120


def test_encode_relational_time_clamps() -> None:
    assert encode_relational_time(0, 500) == RTE_MAX_LEN - 1
    assert encode_relational_time(500, 0) == 0


def test_timestamp_to_day_index() -> None:
    min_ts = 1_000_000
    day_ms = 86_400_000
    assert timestamp_to_day_index(min_ts + 2 * day_ms, min_ts) == 2


def test_edge_time_from_nodes_in_range() -> None:
    node_days = build_node_day_times(
        [(0, 0, 1), (1, 1, 5)],
        n_users=2,
        n_items=2,
        n_aspects=1,
        cutoff_day=10,
    )
    edge_index = np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int64)
    edge_time = edge_time_from_nodes(edge_index, node_days)
    assert_rte_edge_time(edge_time)
    assert edge_time.max() < RTE_MAX_LEN
