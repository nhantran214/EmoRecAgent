"""Paired significance testing for ranking metric deltas.

Methods deltas (full vs ablation/baseline) are reported with a paired test over
per-user metric values plus a bootstrap confidence interval, so the claimed
improvements are defensible rather than anecdotal. Per-user pairing is required:
the same users are scored by both methods, so the comparison must be paired.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from scipy import stats


@dataclass(frozen=True, slots=True)
class PairedResult:
    mean_delta: float          # mean(a) - mean(b)
    p_value: float             # two-sided
    ci_low: float
    ci_high: float
    n: int
    method: str


def _validate(a: list[float], b: list[float]) -> None:
    if len(a) != len(b):
        raise ValueError("paired comparison requires equal-length samples")
    if len(a) < 2:
        raise ValueError("need at least 2 paired observations")


def paired_bootstrap(
    a: list[float],
    b: list[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> PairedResult:
    """Bootstrap the paired mean difference a - b.

    The two-sided p-value is the bootstrap-tail probability that the resampled
    mean delta crosses zero (relative to the observed mean delta), doubled and
    clamped to [0, 1].
    """
    _validate(a, b)
    deltas = [x - y for x, y in zip(a, b)]
    n = len(deltas)
    observed = sum(deltas) / n
    rng = random.Random(seed)

    resampled: list[float] = []
    crossings = 0
    for _ in range(n_bootstrap):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        m = sum(sample) / n
        resampled.append(m)
        # count resamples on the opposite side of zero from the observed effect
        if (observed >= 0 and m <= 0) or (observed < 0 and m >= 0):
            crossings += 1

    resampled.sort()
    lo_idx = int((1 - ci) / 2 * n_bootstrap)
    hi_idx = min(n_bootstrap - 1, int((1 + ci) / 2 * n_bootstrap))
    p = min(1.0, 2.0 * crossings / n_bootstrap)
    return PairedResult(
        mean_delta=observed,
        p_value=p,
        ci_low=resampled[lo_idx],
        ci_high=resampled[hi_idx],
        n=n,
        method="paired_bootstrap",
    )


def paired_ttest(a: list[float], b: list[float]) -> PairedResult:
    """Paired two-sided t-test on a - b (parametric companion to the bootstrap)."""
    _validate(a, b)
    deltas = [x - y for x, y in zip(a, b)]
    n = len(deltas)
    mean_delta = sum(deltas) / n
    result = stats.ttest_rel(a, b)
    sd = stats.tstd(deltas) if n > 1 else 0.0
    se = sd / (n ** 0.5) if n > 0 else 0.0
    half = 1.96 * se
    return PairedResult(
        mean_delta=mean_delta,
        p_value=float(result.pvalue),
        ci_low=mean_delta - half,
        ci_high=mean_delta + half,
        n=n,
        method="paired_ttest",
    )
