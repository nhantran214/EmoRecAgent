"""Claim-specific evaluation — the headline evidence for the temporal mechanism
(U12 / R11).

Aggregate ranking on a single held-out item cannot, by itself, demonstrate a
*temporal* effect. Two targeted analyses do:

1. **Shift-subpopulation selection** (`select_shift_users`): pre-register the
   users whose most recent pre-test review introduces a strong complaint on an
   aspect that was previously non-salient for them. The dynamic-vs-static lift is
   then reported on *this* subpopulation, where the mechanism is supposed to act.

2. **Counterfactual probe** (`counterfactual_probe`): inject a synthetic recent
   complaint on an aspect and verify the ranking actually shifts toward items
   that are strong on that aspect — a *measured* result, not only a unit test.
   (A complaint raises the aspect's salience w_u(a); items with high Ê_i(a) — i.e.
   that handle the aspect well — should rise.)
"""

from __future__ import annotations

from dataclasses import dataclass

from ..scoring.dynamic_weights import AspectSignal, compute_weights
from ..scoring.score import rank_items
from ..baselines.aspect_aware import static_weights


def select_shift_users(
    user_signals: dict[str, list[AspectSignal]],
    neg_threshold: float = 0.5,
    prior_salience_threshold: float = 0.1,
) -> dict[str, str]:
    """Users whose latest signal is a strong complaint on a previously
    non-salient aspect. Returns {user_id: shifted_aspect}.

    Pre-registration discipline: this selection uses only information available
    before the test item (the user's own prior review stream), so it does not
    peek at the held-out label.
    """
    selected: dict[str, str] = {}
    for user, signals in user_signals.items():
        if len(signals) < 2:
            continue
        ordered = sorted(signals, key=lambda s: s.timestamp_ms)
        latest, prior = ordered[-1], ordered[:-1]
        if latest.polarity > -neg_threshold:
            continue  # not a strong complaint
        prior_w = static_weights(prior)
        if prior_w.get(latest.aspect, 0.0) <= prior_salience_threshold:
            selected[user] = latest.aspect
    return selected


@dataclass(frozen=True, slots=True)
class ProbeResult:
    target_item: str
    rank_before: int          # 1-indexed
    rank_after: int           # 1-indexed
    moved_up: bool


def counterfactual_probe(
    base_signals: list[AspectSignal],
    t_query_ms: int,
    lambda_per_day: float,
    alpha: float,
    s_base: dict[str, float],
    item_aspects: dict[str, dict[str, float]],
    inject_aspect: str,
    candidates: list[str],
    inject_polarity: float = -0.9,
) -> ProbeResult:
    """Rank candidates before and after injecting a recent complaint on
    `inject_aspect`; track the candidate that best handles that aspect.
    """
    target = max(
        candidates, key=lambda i: (item_aspects.get(i, {}).get(inject_aspect, 0.0), i)
    )

    def _rank_of(signals: list[AspectSignal]) -> int:
        weights = compute_weights(signals, t_query_ms, lambda_per_day)
        ranking = rank_items(alpha, s_base, weights, item_aspects)
        order = [item for item, _ in ranking]
        return order.index(target) + 1

    rank_before = _rank_of(base_signals)
    injected = base_signals + [
        AspectSignal(inject_aspect, inject_polarity, t_query_ms - 1)
    ]
    rank_after = _rank_of(injected)
    return ProbeResult(
        target_item=target,
        rank_before=rank_before,
        rank_after=rank_after,
        moved_up=rank_after < rank_before,
    )
