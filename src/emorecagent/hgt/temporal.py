"""Relative temporal encoding for pyHGT-style RTE (max_len=240)."""

from __future__ import annotations

import numpy as np

RTE_MAX_LEN = 240
RTE_CENTER = RTE_MAX_LEN // 2  # 120 — zero delta maps here (pyHGT convention)
MS_PER_DAY = 86_400_000


def timestamp_to_day_index(ts: int, min_ts: int) -> int:
    """Map a review timestamp (ms) to whole days since dataset minimum."""
    return int((int(ts) - int(min_ts)) // MS_PER_DAY)


def encode_relational_time(
    source_day: int,
    target_day: int,
    *,
    max_len: int = RTE_MAX_LEN,
) -> int:
    """Encode target−source day delta into RTE embedding index."""
    center = max_len // 2
    delta = int(target_day) - int(source_day)
    return max(0, min(max_len - 1, delta + center))


def build_node_day_times(
    train_timestamps: list[tuple[int, int, int]],
    *,
    n_users: int,
    n_items: int,
    n_aspects: int,
    cutoff_day: int,
) -> np.ndarray:
    """Per-node day index: users/items from interactions, aspects at cutoff."""
    user_days = np.zeros(n_users, dtype=np.int64)
    item_days = np.zeros(n_items, dtype=np.int64)
    for u_local, i_local, day in train_timestamps:
        user_days[u_local] = max(user_days[u_local], day)
        item_days[i_local] = max(item_days[i_local], day)
    aspect_days = np.full(n_aspects, cutoff_day, dtype=np.int64)
    return np.concatenate([user_days, item_days, aspect_days])


def edge_time_from_nodes(
    edge_index: np.ndarray,
    node_day_times: np.ndarray,
    *,
    max_len: int = RTE_MAX_LEN,
) -> np.ndarray:
    """Vectorized RTE indices for all edges."""
    src = edge_index[0]
    dst = edge_index[1]
    center = max_len // 2
    delta = node_day_times[dst] - node_day_times[src]
    return np.clip(delta + center, 0, max_len - 1).astype(np.int64)


def assert_rte_edge_time(edge_time: np.ndarray, *, max_len: int = RTE_MAX_LEN) -> None:
    if edge_time.size == 0:
        return
    lo = int(edge_time.min())
    hi = int(edge_time.max())
    if lo < 0 or hi >= max_len:
        raise ValueError(
            f"edge_time must be in [0, {max_len - 1}] for RelTemporalEncoding; "
            f"got min={lo} max={hi}. Rebuild the graph: make build-hgt-graph"
        )
