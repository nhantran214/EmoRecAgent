"""U12 claim-specific eval tests: shift-subpopulation selection + counterfactual."""

from __future__ import annotations

from emorecagent.eval.shift_eval import counterfactual_probe, select_shift_users
from emorecagent.scoring.dynamic_weights import AspectSignal

DAY = 86_400_000


def test_select_picks_only_users_with_new_salient_complaint() -> None:
    user_signals = {
        # u1: latest is a strong NEW complaint on "comfort" (prior only "scent")
        "u1": [
            AspectSignal("scent", 0.6, 1 * DAY),
            AspectSignal("scent", 0.5, 2 * DAY),
            AspectSignal("comfort", -0.9, 3 * DAY),
        ],
        # u2: latest is positive -> not a complaint
        "u2": [
            AspectSignal("scent", -0.4, 1 * DAY),
            AspectSignal("comfort", 0.8, 2 * DAY),
        ],
        # u3: complaint on comfort, but comfort was ALREADY salient -> not a shift
        "u3": [
            AspectSignal("comfort", -0.8, 1 * DAY),
            AspectSignal("comfort", -0.7, 2 * DAY),
            AspectSignal("comfort", -0.9, 3 * DAY),
        ],
    }
    selected = select_shift_users(user_signals)
    assert selected == {"u1": "comfort"}


def test_single_signal_user_is_not_selected() -> None:
    assert select_shift_users({"u": [AspectSignal("a", -0.9, DAY)]}) == {}


def test_counterfactual_complaint_moves_aspect_item_up() -> None:
    base_signals = [AspectSignal("scent", 0.5, 90 * DAY)]
    item_aspects = {
        "comfy": {"comfort": 0.95, "scent": 0.20},
        "scented": {"comfort": 0.10, "scent": 0.95},
        "plain": {},
    }
    s_base = {"comfy": 0.5, "scented": 0.5, "plain": 0.0}
    candidates = ["comfy", "scented", "plain"]

    res = counterfactual_probe(
        base_signals=base_signals,
        t_query_ms=100 * DAY,
        lambda_per_day=0.01,
        alpha=0.5,
        s_base=s_base,
        item_aspects=item_aspects,
        inject_aspect="comfort",
        candidates=candidates,
    )
    assert res.target_item == "comfy"   # best handles comfort
    assert res.rank_before == 2          # behind "scented" before the complaint
    assert res.rank_after == 1           # rises after the comfort complaint
    assert res.moved_up
