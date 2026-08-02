"""Order-aware sequential baseline.

Defends the *temporal* claim against a model that already exploits interaction
order. This is a first-order Markov item-transition recommender: it learns, from
each user's chronologically ordered train history, how often item j follows item
i, and scores candidates by the transition distribution from the user's most
recent train item (with popularity back-off for unseen contexts).

A neural session model (SASRec / TiSASRec) is the heavier drop-in replacement
behind the same `Recommender` interface; it is deferred here to avoid pulling a
deep-learning dependency into the numeric core. The transition model is enough to
show that order-awareness alone does not capture the *aspect-shift* signal our
dynamic mechanism targets.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from ..data.types import Interaction
from .base import Recommender


class SequentialRecommender(Recommender):
    name = "sequential"

    def __init__(self) -> None:
        self._transitions: dict[str, Counter[str]] = defaultdict(Counter)
        self._last_item: dict[str, str] = {}
        self._popularity: Counter[str] = Counter()

    def fit(self, interactions: list[Interaction]) -> "SequentialRecommender":
        by_user: dict[str, list[Interaction]] = defaultdict(list)
        for it in interactions:
            by_user[it.user_id].append(it)
            self._popularity[it.item] += 1
        for user, items in by_user.items():
            items.sort(key=lambda x: (x.timestamp, x.item))
            for prev, nxt in zip(items, items[1:]):
                self._transitions[prev.item][nxt.item] += 1
            self._last_item[user] = items[-1].item
        return self

    def score(self, user_id: str, candidates: list[str]) -> dict[str, float]:
        last = self._last_item.get(user_id)
        trans = self._transitions.get(last, Counter()) if last else Counter()
        if trans:
            return {i: float(trans.get(i, 0)) for i in candidates}
        # cold context: fall back to global popularity so ranking is still defined
        return {i: float(self._popularity.get(i, 0)) for i in candidates}
