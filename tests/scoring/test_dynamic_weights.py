"""Tests for dynamic time-decayed aspect weights."""

from __future__ import annotations

import math

from emorecagent.agents.profiling_agent import DynamicUserProfilingAgent
from emorecagent.scoring.dynamic_weights import (
    MS_PER_DAY,
    AspectSignal,
    aspect_gammas,
    compute_weights,
    top_k_aspects,
)

DAY = int(MS_PER_DAY)


def test_recent_strong_complaint_outweighs_older_positive():
    t_now = 100 * DAY
    signals = [
        # old, strong positive on "scent" (90 days ago)
        AspectSignal("scent", polarity=1.0, timestamp_ms=t_now - 90 * DAY),
        # recent, strong complaint on "comfort" (1 day ago)
        AspectSignal("comfort", polarity=-1.0, timestamp_ms=t_now - 1 * DAY),
    ]
    w = compute_weights(signals, t_now, lambda_per_day=0.05)
    assert w["comfort"] > w["scent"]
    assert top_k_aspects(w, 1)[0][0] == "comfort"


def test_weights_sum_to_one():
    t_now = 10 * DAY
    signals = [
        AspectSignal("a", 0.8, t_now - DAY),
        AspectSignal("b", -0.6, t_now - 2 * DAY),
        AspectSignal("c", 0.5, t_now - 3 * DAY),
    ]
    w = compute_weights(signals, t_now, lambda_per_day=0.02)
    assert math.isclose(sum(w.values()), 1.0, rel_tol=1e-9)


def test_larger_lambda_decays_old_signal_faster():
    t_now = 50 * DAY
    signals = [
        AspectSignal("recent", 1.0, t_now - 1 * DAY),
        AspectSignal("old", 1.0, t_now - 40 * DAY),
    ]
    w_slow = compute_weights(signals, t_now, lambda_per_day=0.001)
    w_fast = compute_weights(signals, t_now, lambda_per_day=0.2)
    # Faster decay shifts more weight onto the recent aspect.
    assert w_fast["recent"] > w_slow["recent"]


def test_single_aspect_gets_full_weight():
    t_now = 5 * DAY
    w = compute_weights(
        [AspectSignal("only", 0.4, t_now - DAY)], t_now, lambda_per_day=0.1
    )
    assert w == {"only": 1.0}


def test_empty_history_returns_empty_profile():
    assert compute_weights([], 1000, lambda_per_day=0.1) == {}


def test_future_signals_excluded():
    t_now = 10 * DAY
    # a signal at/after query time must not contribute (no leakage)
    w = compute_weights(
        [AspectSignal("future", 1.0, t_now + DAY)], t_now, lambda_per_day=0.1
    )
    assert w == {}


class _FakeSource:
    def __init__(self, signals):
        self._signals = signals
        self.persisted = None

    def get_user_aspect_signals(self, user_id):
        return self._signals

    def upsert_user_preferences(self, user_id, weights, updated_ts):
        self.persisted = (user_id, weights, updated_ts)


def test_profiling_agent_computes_and_persists():
    t_now = 20 * DAY
    src = _FakeSource(
        [
            AspectSignal("comfort", -1.0, t_now - DAY),
            AspectSignal("scent", 0.5, t_now - 10 * DAY),
        ]
    )
    agent = DynamicUserProfilingAgent(src, lambda_per_day=0.05)
    weights = agent.profile("u1", t_now, top_k=5, persist=True)
    assert math.isclose(sum(weights.values()), 1.0, rel_tol=1e-9)
    assert src.persisted is not None and src.persisted[0] == "u1"
    assert agent.top_aspects("u1", t_now, 1)[0][0] == "comfort"


def test_aspect_gammas_is_signed():
    t_now = 10 * DAY
    signals = [
        AspectSignal("scent", polarity=0.8, timestamp_ms=t_now - DAY),
        AspectSignal("comfort", polarity=-0.9, timestamp_ms=t_now - DAY),
    ]
    gammas = aspect_gammas(signals, t_now, lambda_per_day=0.01)
    assert gammas["scent"] > 0
    assert gammas["comfort"] < 0
