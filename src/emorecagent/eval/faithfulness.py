"""Non-circular explanation faithfulness via score perturbation.

We do NOT score faithfulness by lexical overlap (ROUGE-L is dropped — overlap is
not faithfulness). Instead we use an ERASER-style perturbation test on the score
itself: an explanation cites the aspects it claims drove the recommendation, so
removing exactly those aspect contributions must actually hurt the item.

- **comprehensiveness**: zero out the *cited* aspects' contributions for the item
  and re-rank. A faithful explanation cites true drivers, so the item's rank
  drops (and its score drops). The magnitude is the comprehensiveness.
- **sufficiency**: keep *only* the cited aspects (plus the base term). A faithful
  explanation's cited aspects nearly reconstruct the score, so the drop is small.
- **unfaithful control**: citing non-driving aspects yields ~0 comprehensiveness;
  this proves the metric discriminates rather than rewarding any explanation.

All functions operate on `ScoreBreakdown`, which carries per-aspect
contribution magnitudes precisely so this test is possible.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..scoring.score import ScoreBreakdown


def _rank_of(totals: dict[str, float], target: str) -> int:
    """1-indexed rank of target (descending score, ties by item id)."""
    order = sorted(totals, key=lambda i: (-totals[i], i))
    return order.index(target) + 1


def _total_without(breakdown: ScoreBreakdown, cited: set[str]) -> float:
    removed = sum(
        c for a, c in breakdown.aspect_contributions.items() if a in cited
    )
    return breakdown.total - removed


def _total_only(breakdown: ScoreBreakdown, cited: set[str]) -> float:
    kept = sum(
        c for a, c in breakdown.aspect_contributions.items() if a in cited
    )
    return breakdown.base_contribution + kept


@dataclass(frozen=True, slots=True)
class FaithfulnessResult:
    target_item: str
    rank_before: int
    rank_after: int           # after removing cited aspects from the target
    rank_drop: int            # rank_after - rank_before (>0 = fell, faithful)
    comprehensiveness: float  # score removed by deleting cited aspects (>=0)
    sufficiency_gap: float    # |full - only-cited| (small = cited explain it)
    faithful: bool            # rank actually dropped


def evaluate_explanation(
    breakdowns: dict[str, ScoreBreakdown],
    target_item: str,
    cited_aspects: set[str],
) -> FaithfulnessResult:
    """Perturbation faithfulness for one item's explanation within its candidate
    set. Only the target item's score is perturbed; the rest hold fixed.
    """
    totals_full = {i: b.total for i, b in breakdowns.items()}
    target_bd = breakdowns[target_item]
    rank_before = _rank_of(totals_full, target_item)

    totals_removed = dict(totals_full)
    totals_removed[target_item] = _total_without(target_bd, cited_aspects)
    rank_after = _rank_of(totals_removed, target_item)

    comprehensiveness = target_bd.total - _total_without(target_bd, cited_aspects)
    sufficiency_gap = abs(target_bd.total - _total_only(target_bd, cited_aspects))

    return FaithfulnessResult(
        target_item=target_item,
        rank_before=rank_before,
        rank_after=rank_after,
        rank_drop=rank_after - rank_before,
        comprehensiveness=comprehensiveness,
        sufficiency_gap=sufficiency_gap,
        faithful=rank_after > rank_before,
    )
