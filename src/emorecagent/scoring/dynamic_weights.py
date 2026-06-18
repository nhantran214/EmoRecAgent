"""Dynamic, time-decayed aspect-preference weights w_u(a, t) — the core of the
Dynamic Preference Shifting contribution.

For each new affective signal on aspect a at time t_k, the user's interest in a
accumulates a time-decayed *salience* signal:

    I_u(a, t) = sum_{t_k < t}  intensity(s_k) * exp(-lambda * (t - t_k))
    w_u(a, t) = I_u(a, t) / sum_{a'} I_u(a', t)

`w_u` encodes salience (how much the user cares about a), NOT preference
direction — direction is supplied later by the item sentiment E_i(a). Hence
`intensity` peaks for strong sentiment of *either* polarity, so a fresh strong
complaint elevates the aspect. Time decay lives here and nowhere else.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

MS_PER_DAY = 86_400_000.0


@dataclass(frozen=True, slots=True)
class AspectSignal:
    """An affective signal on one aspect, from one of the user's past reviews."""

    aspect: str
    polarity: float  # in [-1, 1]; sign = direction, magnitude = strength
    timestamp_ms: int


def intensity(polarity: float) -> float:
    """Salience of a signal: strong sentiment of either polarity counts most."""
    return abs(polarity)


def interest_scores(
    signals: list[AspectSignal],
    t_query_ms: int,
    lambda_per_day: float,
) -> dict[str, float]:
    """Time-decayed, unnormalized interest I_u(a, t) per aspect.

    Only signals strictly before the query time contribute (no leakage).
    """
    out: dict[str, float] = defaultdict(float)
    for s in signals:
        if s.timestamp_ms >= t_query_ms:
            continue
        dt_days = (t_query_ms - s.timestamp_ms) / MS_PER_DAY
        out[s.aspect] += intensity(s.polarity) * math.exp(-lambda_per_day * dt_days)
    return dict(out)


def compute_weights(
    signals: list[AspectSignal],
    t_query_ms: int,
    lambda_per_day: float,
) -> dict[str, float]:
    """Normalized weights w_u(a, t); empty dict when there is no past signal."""
    interest = interest_scores(signals, t_query_ms, lambda_per_day)
    total = sum(interest.values())
    if total <= 0.0:
        return {}
    return {a: v / total for a, v in interest.items()}


def top_k_aspects(weights: dict[str, float], k: int) -> list[tuple[str, float]]:
    """The k highest-weight aspects, descending (ties broken by aspect name)."""
    return sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


def aspect_gammas(
    signals: list[AspectSignal],
    t_query_ms: int,
    lambda_per_day: float,
) -> dict[str, float]:
    """Signed, time-decayed salience per aspect for HGT user embedding injection."""
    out: dict[str, float] = defaultdict(float)
    for s in signals:
        if s.timestamp_ms >= t_query_ms:
            continue
        dt_days = (t_query_ms - s.timestamp_ms) / MS_PER_DAY
        out[s.aspect] += (
            s.polarity * intensity(s.polarity) * math.exp(-lambda_per_day * dt_days)
        )
    return dict(out)
