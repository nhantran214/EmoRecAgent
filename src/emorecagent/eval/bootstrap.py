"""Percentile bootstrap confidence intervals for metric aggregates."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    low: float
    high: float
    n_bootstrap: int


def bootstrap_ci(
    values: list[float],
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> BootstrapCI:
    """Percentile bootstrap CI for the mean of ``values``."""
    if not values:
        return BootstrapCI(low=0.0, high=0.0, n_bootstrap=n_bootstrap)
    n = len(values)
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int((1 - ci) / 2 * n_bootstrap)
    hi_idx = min(n_bootstrap - 1, int((1 + ci) / 2 * n_bootstrap))
    return BootstrapCI(low=means[lo_idx], high=means[hi_idx], n_bootstrap=n_bootstrap)


def bootstrap_user_mean_ci(
    per_user: dict[str, list[float]],
    user_ids: list[str],
    metric_key: str,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> BootstrapCI:
    """Bootstrap CI on user-mean for ``metric_key``."""
    by_user: dict[str, list[float]] = defaultdict(list)
    for idx, uid in enumerate(user_ids):
        by_user[uid].append(per_user[metric_key][idx])
    user_means = [sum(v) / len(v) for v in by_user.values()]
    return bootstrap_ci(user_means, n_bootstrap=n_bootstrap, seed=seed)
