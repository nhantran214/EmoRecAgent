"""Sequence helpers for TiSASRec inference."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_time_scale(
    times: list[int] | tuple[int, ...],
    *,
    time_unit_seconds: int | None = None,
) -> int:
    """Divisor turning raw unix-ms timestamps into TiSASRec relative units.

    ``time_unit_seconds=None`` keeps Li et al.'s per-user rule (divide by that
    user's smallest non-zero gap). That rule assumes a coarse, roughly uniform
    cadence. On Yelp it collapses: most users have a sub-hour gap somewhere, so
    the divisor becomes minutes, a year of history overflows ``time_span``, and
    every bucket saturates. It also makes the unit user-specific, so the shared
    interval embedding sees "35 minutes" and "5 months" as the same bucket.

    Passing an explicit unit (e.g. 86400 for days) gives one global,
    interpretable scale instead.
    """
    if time_unit_seconds is not None:
        return max(1, int(time_unit_seconds) * 1000)
    diffs = {
        times[i + 1] - times[i]
        for i in range(len(times) - 1)
        if times[i + 1] - times[i] != 0
    }
    return min(diffs) if diffs else 1


def compute_repos(time_seq: np.ndarray, time_span: int) -> np.ndarray:
    size = time_seq.shape[0]
    time_matrix = np.zeros([size, size], dtype=np.int32)
    for i in range(size):
        for j in range(size):
            span = abs(int(time_seq[i]) - int(time_seq[j]))
            time_matrix[i][j] = time_span if span > time_span else span
    return time_matrix


class TiSASRecArgs:
    """Minimal args namespace expected by baseline TiSASRec."""

    def __init__(self, model_cfg: dict[str, Any], device: str) -> None:
        self.device = device
        self.maxlen = model_cfg.get("maxlen", 50)
        self.hidden_units = model_cfg.get("hidden_units", 64)
        self.num_blocks = model_cfg.get("num_blocks", 2)
        self.num_heads = model_cfg.get("num_heads", 1)
        self.dropout_rate = model_cfg.get("dropout_rate", 0.2)
        self.l2_emb = model_cfg.get("l2_emb", 1e-4)
        self.time_span = model_cfg.get("time_span", 256)
