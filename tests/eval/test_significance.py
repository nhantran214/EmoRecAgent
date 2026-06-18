"""Significance tests: paired bootstrap and t-test on synthetic deltas."""

from __future__ import annotations

import pytest

from emorecagent.eval.significance import paired_bootstrap, paired_ttest


def test_bootstrap_detects_clear_positive_delta() -> None:
    a = [1.0] * 30           # method A always wins
    b = [0.0] * 30
    res = paired_bootstrap(a, b, n_bootstrap=500, seed=1)
    assert res.mean_delta == pytest.approx(1.0)
    assert res.p_value < 0.05
    assert res.ci_low > 0.0


def test_bootstrap_no_effect_is_not_significant() -> None:
    # identical methods -> zero delta, should never be significant
    a = [0.3, 0.7, 0.5, 0.6, 0.4, 0.5, 0.55, 0.45]
    res = paired_bootstrap(a, list(a), n_bootstrap=500, seed=2)
    assert res.mean_delta == pytest.approx(0.0)
    assert res.p_value == pytest.approx(1.0)


def test_ttest_agrees_on_direction() -> None:
    a = [0.8, 0.9, 0.7, 0.85, 0.95, 0.75]
    b = [0.5, 0.4, 0.45, 0.5, 0.55, 0.42]
    res = paired_ttest(a, b)
    assert res.mean_delta > 0
    assert res.p_value < 0.05


def test_unequal_lengths_rejected() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, 2.0], [1.0], n_bootstrap=10)


def test_too_few_observations_rejected() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], [0.0], n_bootstrap=10)
