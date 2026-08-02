"""Iterative k-core filtering.

Repeatedly removes users with fewer than k reviews (distinct items) and items
with fewer than k reviews (distinct users) until the graph is stable.

With the default ``k_core: 5`` config, every surviving user has reviewed at
least 5 products and every surviving product has at least 5 distinct reviewers.
That density is required for reliable User Profiling and agentic reasoning.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .types import Interaction


@dataclass(frozen=True, slots=True)
class KCoreSummary:
    """Post-filter statistics used in dataset manifests."""

    k: int
    n_interactions: int
    n_users: int
    n_items: int
    min_user_degree: int
    min_item_degree: int

    def as_manifest(self) -> dict:
        return {
            "k_core": self.k,
            "n_interactions_after_kcore": self.n_interactions,
            "n_users_after_kcore": self.n_users,
            "n_items_after_kcore": self.n_items,
            "min_user_degree": self.min_user_degree,
            "min_item_degree": self.min_item_degree,
        }


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


def k_core_summary(interactions: list[Interaction], k: int) -> KCoreSummary:
    """Summarize degrees after filtering; empty input yields zeros."""
    if not interactions:
        return KCoreSummary(
            k=k, n_interactions=0, n_users=0, n_items=0,
            min_user_degree=0, min_item_degree=0,
        )
    user_deg = Counter(it.user_id for it in interactions)
    item_deg = Counter(it.item for it in interactions)
    return KCoreSummary(
        k=k,
        n_interactions=len(interactions),
        n_users=len(user_deg),
        n_items=len(item_deg),
        min_user_degree=min(user_deg.values()),
        min_item_degree=min(item_deg.values()),
    )
