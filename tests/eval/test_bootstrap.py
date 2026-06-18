"""Bootstrap CI tests."""

from __future__ import annotations

from emorecagent.eval.bootstrap import bootstrap_ci


def test_constant_vector_zero_width() -> None:
    ci = bootstrap_ci([0.5, 0.5, 0.5], n_bootstrap=100, seed=1)
    assert ci.low == ci.high == 0.5


def test_empty_returns_zeros() -> None:
    ci = bootstrap_ci([], n_bootstrap=10)
    assert ci.low == 0.0 and ci.high == 0.0
