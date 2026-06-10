"""Iterative k-core filtering.

Repeatedly removes users with fewer than k distinct items and items with fewer
than k distinct users until the interaction set is stable (the k-core).
"""

from __future__ import annotations

from collections import Counter

from .types import Interaction


def k_core_filter(interactions: list[Interaction], k: int) -> list[Interaction]:
    """Return the k-core of the interaction set.

    k <= 1 is a no-op. Operates on distinct (user, item) edges; callers should
    de-duplicate first if they want degree to mean distinct partners.
    """
    if k <= 1:
        return list(interactions)

    current = list(interactions)
    while True:
        user_deg = Counter(it.user_id for it in current)
        item_deg = Counter(it.item for it in current)

        kept = [
            it
            for it in current
            if user_deg[it.user_id] >= k and item_deg[it.item] >= k
        ]
        if len(kept) == len(current):
            return kept
        current = kept
        if not current:
            return current
