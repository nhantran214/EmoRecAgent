"""Shared lightweight types for the data pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Interaction:
    """A single user-item interaction (rating_only style, no review text).

    Review text is intentionally excluded to keep interaction sets small in
    memory; ABSA (U4) reads raw review text separately by id.
    """

    user_id: str
    item: str  # parent_asin
    rating: float
    timestamp: int  # unix ms
    helpful_vote: int = 0
    verified_purchase: bool = False
