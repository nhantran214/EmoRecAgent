"""Relative-time scaling for the TiSASRec interval matrix."""

from __future__ import annotations

import numpy as np

from emorecagent.sequential.seq_utils import compute_repos, compute_time_scale

DAY_MS = 86_400_000


def test_default_keeps_per_user_min_gap() -> None:
    """``None`` must preserve Li et al. behaviour for the Amazon track."""
    times = [0, 2000, 5000]
    assert compute_time_scale(times) == 2000


def test_single_event_scale_is_safe() -> None:
    assert compute_time_scale([123]) == 1
    assert compute_time_scale([]) == 1


def test_fixed_unit_ignores_min_gap() -> None:
    times = [0, 60_000, 400 * DAY_MS]
    assert compute_time_scale(times, time_unit_seconds=86400) == DAY_MS


def test_fixed_unit_prevents_time_span_saturation() -> None:
    """The Yelp failure mode: one short gap collapses the whole matrix.

    A user with a 1-minute gap plus a year of history gets a 60s divisor under
    the min-gap rule, so every interval overflows ``time_span`` and the shared
    interval embedding degenerates to a constant.
    """
    times = [0, 60_000] + [d * DAY_MS for d in (30, 120, 250)]
    span = 256

    def clipped_fraction(scale: int) -> float:
        seq = np.array([round(t / scale) + 1 for t in times])
        matrix = compute_repos(seq, span)
        off_diag = ~np.eye(len(times), dtype=bool)
        return float((matrix[off_diag] == span).mean())

    min_gap = clipped_fraction(compute_time_scale(times))
    day = clipped_fraction(compute_time_scale(times, time_unit_seconds=86400))

    assert min_gap > 0.8
    assert day == 0.0
