"""Dynamic Preference Shifting score S(u, i).

    S(u, i) = alpha * S_base(u, i) + (1 - alpha) * sum_{a in A_u} w_u(a,t) * Ê_i(a)

Time decay lives entirely inside w_u (see scoring.dynamic_weights); there is no
second decay term here. Aspects with no item sentiment contribute nothing; when
there is no aspect overlap, S falls back to alpha * S_base (recorded in the
breakdown, never silent).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    """Decomposition of S(u, i) for explanations and faithfulness checks."""

    total: float
    base_contribution: float  # alpha * S_base
    aspect_contributions: dict[str, float] = field(default_factory=dict)

    @property
    def affective_total(self) -> float:
        return sum(self.aspect_contributions.values())


def score_item(
    alpha: float,
    s_base: float,
    weights: dict[str, float],
    item_aspects: dict[str, float],
) -> ScoreBreakdown:
    """Compute S(u, i) and its per-aspect breakdown.

    `weights` is w_u(a, t) (sums to 1 over A_u); `item_aspects` is Ê_i(a) in
    [0, 1]. Only aspects present in both contribute to the affective term.
    """
    base_contribution = alpha * s_base
    aspect_contributions: dict[str, float] = {}
    for aspect, w in weights.items():
        e_hat = item_aspects.get(aspect)
        if e_hat is None:
            continue
        aspect_contributions[aspect] = (1.0 - alpha) * w * e_hat
    total = base_contribution + sum(aspect_contributions.values())
    return ScoreBreakdown(
        total=total,
        base_contribution=base_contribution,
        aspect_contributions=aspect_contributions,
    )


def rank_items(
    alpha: float,
    s_base: dict[str, float],
    weights: dict[str, float],
    item_aspects: dict[str, dict[str, float]],
) -> list[tuple[str, ScoreBreakdown]]:
    """Score and rank candidate items (descending S, ties by item id)."""
    scored = [
        (
            item,
            score_item(alpha, s_base.get(item, 0.0), weights, item_aspects.get(item, {})),
        )
        for item in s_base
    ]
    scored.sort(key=lambda kv: (-kv[1].total, kv[0]))
    return scored
