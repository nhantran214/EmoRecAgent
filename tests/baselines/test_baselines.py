"""U11 baseline tests: shared interface, k/exclusion, and per-baseline behavior."""

from __future__ import annotations

from emorecagent.baselines import (
    AspectAwareRecommender,
    ItemKNNRecommender,
    PopularityRecommender,
    SequentialRecommender,
    SVDRecommender,
)
from emorecagent.baselines.aspect_aware import static_weights
from emorecagent.data.types import Interaction
from emorecagent.scoring.dynamic_weights import AspectSignal

DAY = 86_400_000


def _interaction(u: str, i: str, t: int) -> Interaction:
    return Interaction(user_id=u, item=i, rating=5.0, timestamp=t)


def _toy() -> list[Interaction]:
    # u1 likes a-items, u2 likes b-items; popularity: i_a1 most frequent.
    return [
        _interaction("u1", "i_a1", 1 * DAY),
        _interaction("u1", "i_a2", 2 * DAY),
        _interaction("u2", "i_a1", 1 * DAY),
        _interaction("u2", "i_b1", 3 * DAY),
        _interaction("u3", "i_a1", 1 * DAY),
        _interaction("u3", "i_b1", 2 * DAY),
    ]


def test_recommend_returns_exactly_k_and_excludes_nothing_implicitly() -> None:
    rec = PopularityRecommender().fit(_toy())
    cands = ["i_a1", "i_a2", "i_b1"]
    top = rec.recommend("u1", cands, k=2)
    assert len(top) == 2
    assert top[0] == "i_a1"  # most popular


def test_harness_excludes_train_items_via_candidate_set() -> None:
    # The harness is responsible for exclusion: it passes only non-train items.
    rec = SVDRecommender(factors=2, seed=0).fit(_toy())
    train_items = {"i_a1", "i_a2"}
    candidates = [i for i in ["i_a1", "i_a2", "i_b1"] if i not in train_items]
    top = rec.recommend("u1", candidates, k=5)
    assert set(top).isdisjoint(train_items)
    assert top == ["i_b1"]


def test_popularity_is_user_independent() -> None:
    rec = PopularityRecommender().fit(_toy())
    cands = ["i_a1", "i_a2", "i_b1"]
    assert rec.score("u1", cands) == rec.score("u2", cands)


def test_svd_and_itemknn_beat_random_expectation_on_fixture() -> None:
    # u1 co-occurs with u3 on i_a1; u3 also has i_b1. A CF method should rank
    # i_b1 above an item u1 has no collaborative path to.
    data = _toy() + [_interaction("u1", "i_a1b", 4 * DAY)]
    for rec in (SVDRecommender(factors=3, seed=0), ItemKNNRecommender(seed=0)):
        rec.fit(data)
        scores = rec.score("u1", ["i_b1", "i_unrelated"])
        assert scores["i_b1"] >= scores["i_unrelated"]


def test_static_weights_ignore_time() -> None:
    # Same aspect signals at very different times → identical static weights.
    near = [
        AspectSignal("comfort", -0.9, 100 * DAY),
        AspectSignal("scent", 0.5, 101 * DAY),
    ]
    far = [
        AspectSignal("comfort", -0.9, 1 * DAY),
        AspectSignal("scent", 0.5, 2 * DAY),
    ]
    assert static_weights(near) == static_weights(far)


def test_aspect_aware_consumes_same_e_hat_inputs() -> None:
    user_signals = {
        "u1": [AspectSignal("comfort", -0.8, 1 * DAY), AspectSignal("scent", 0.4, 2 * DAY)]
    }
    item_aspects = {
        "i_comfy": {"comfort": 0.9, "scent": 0.5},
        "i_smelly": {"comfort": 0.2, "scent": 0.9},
    }
    rec = AspectAwareRecommender(user_signals, item_aspects).fit(_toy())
    scores = rec.score("u1", ["i_comfy", "i_smelly"])
    # comfort carries ~0.67 static weight; the comfort-strong item should win.
    assert scores["i_comfy"] > scores["i_smelly"]


def test_aspect_aware_unknown_user_scores_zero() -> None:
    rec = AspectAwareRecommender({}, {"i": {"a": 1.0}}).fit(_toy())
    assert rec.score("ghost", ["i"]) == {"i": 0.0}


def test_sequential_respects_interaction_order() -> None:
    # Build a clear transition: i_a1 -> i_a2 occurs, i_a1 -> i_b1 never.
    data = [
        _interaction("u1", "i_a1", 1 * DAY),
        _interaction("u1", "i_a2", 2 * DAY),
        _interaction("u2", "i_a1", 1 * DAY),
        _interaction("u2", "i_a2", 2 * DAY),
        _interaction("u3", "i_a1", 5 * DAY),  # u3's last item is i_a1
    ]
    rec = SequentialRecommender().fit(data)
    scores = rec.score("u3", ["i_a2", "i_b1"])
    assert scores["i_a2"] > scores["i_b1"]


def test_sequential_cold_context_falls_back_to_popularity() -> None:
    rec = SequentialRecommender().fit(_toy())
    # A user whose last item has no recorded outgoing transition.
    scores = rec.score("ghost", ["i_a1", "i_b1"])
    assert scores["i_a1"] >= scores["i_b1"]  # i_a1 is most popular
