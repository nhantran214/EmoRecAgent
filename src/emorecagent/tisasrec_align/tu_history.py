"""Resolve interaction history used when building $T_u$ (Protocol B).

Stage-1 and Stage-2 must share the same history scope at test time. Never include
the eval split (test/valid target rows) in the prefix — that leaks future
relevant items into the manifesto (especially under ``user_batch`` where
``t_q = max(test_ts)``).
"""

from __future__ import annotations

from ..data.types import Interaction


def resolve_tu_history_interactions(
    *,
    split: str,
    train: list[Interaction],
    valid: list[Interaction] | None = None,
    test_history: str = "train",
) -> list[Interaction]:
    """Return interactions allowed in the $T_u$ / ABSA prefix.

    Parameters
    ----------
    split:
        Which split's query keys are being written (``train`` / ``valid`` / ``test``).
    test_history:
        Config ``tisasrec_align.test_history``: ``train`` or ``train_valid``.
        Matches Stage-1 fit history at test time.
    """
    if split == "train":
        return list(train)
    if test_history == "train_valid" and valid:
        return list(train) + list(valid)
    return list(train)


def interaction_keys(rows: list[Interaction]) -> set[tuple[str, str, int]]:
    """``(user_id, item_id, timestamp_ms)`` keys for ABSA / review scoping."""
    return {(it.user_id, it.item, int(it.timestamp)) for it in rows}
