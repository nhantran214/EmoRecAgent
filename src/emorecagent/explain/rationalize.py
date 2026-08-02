"""Evidence-grounded rationalized explanations."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..llm.client import LLMClient
from ..llm.prompts import EXPLANATION_V1, format_prompt
from ..llm.schemas import ExplanationClaims
from ..scoring.dynamic_weights import AspectSignal
from ..scoring.score import ScoreBreakdown


@dataclass(frozen=True, slots=True)
class AspectEvidence:
    aspect: str
    contribution: float
    e_hat: float
    n_support: int
    user_weight: float


@dataclass
class RationalizedExplanation:
    item_id: str
    summary: str
    aspects: list[AspectEvidence] = field(default_factory=list)
    claims: list[str] = field(default_factory=list)

    @property
    def cited_aspects(self) -> list[str]:
        return [a.aspect for a in self.aspects]


def _top_drivers(breakdown: ScoreBreakdown, k: int = 3) -> list[tuple[str, float]]:
    return sorted(
        breakdown.aspect_contributions.items(), key=lambda kv: -kv[1]
    )[:k]


def _user_aspect_context(
    signals: list[AspectSignal], aspect: str
) -> str | None:
    relevant = [s for s in signals if s.aspect == aspect]
    if not relevant:
        return None
    latest = max(relevant, key=lambda s: s.timestamp_ms)
    direction = "positive" if latest.polarity > 0 else "negative" if latest.polarity < 0 else "neutral"
    return f"you previously expressed {direction} sentiment about {aspect}"


def explain_recommendation(
    item_id: str,
    breakdown: ScoreBreakdown,
    weights: dict[str, float],
    item_e_hat: dict[str, float],
    aspect_support: dict[str, int],
    user_signals: list[AspectSignal],
    *,
    llm: LLMClient | None = None,
    polish_with_llm: bool = False,
    numeric_specs: dict[str, str | None] | None = None,
) -> RationalizedExplanation:
    """Build a grounded explanation from score drivers only.

    `numeric_specs` may include optional meta fields (e.g. weight, price); null
    values are omitted from claims — never invented.
    """
    drivers = _top_drivers(breakdown)
    aspects: list[AspectEvidence] = []
    aspect_claims: list[str] = []
    numeric_claims: list[str] = []

    for aspect, contrib in drivers:
        e_hat = item_e_hat.get(aspect, 0.0)
        n_sup = aspect_support.get(aspect, 0)
        w = weights.get(aspect, 0.0)
        aspects.append(
            AspectEvidence(
                aspect=aspect,
                contribution=contrib,
                e_hat=e_hat,
                n_support=n_sup,
                user_weight=w,
            )
        )
        polarity = "positively" if e_hat >= 0.5 else "negatively"
        ctx = _user_aspect_context(user_signals, aspect)
        base = (
            f"This item rates {polarity} on {aspect} "
            f"(Ê={e_hat:.2f}, {n_sup} supporting reviews, contribution {contrib:.3f})"
        )
        if ctx:
            base = f"{ctx}; {base}"
        aspect_claims.append(base)

    if numeric_specs:
        for key, val in numeric_specs.items():
            if val is not None:
                numeric_claims.append(f"{key}: {val}")

    claims = aspect_claims + numeric_claims

    contributions = {a.aspect: a.contribution for a in aspects}
    template_summary = (
        f"Recommended {item_id} because it aligns with your priorities on "
        f"{', '.join(a.aspect for a in aspects)} "
        f"(top driver contribution: {max(contributions.values()) if contributions else 0:.3f})."
    )

    summary = template_summary
    if polish_with_llm and llm is not None:
        prompt = format_prompt(
            EXPLANATION_V1,
            user_aspects=str({a.aspect: a.user_weight for a in aspects}),
            item_aspects=str({a.aspect: a.e_hat for a in aspects}),
            contributions=str(contributions),
        )
        try:
            polished = llm.invoke_structured(prompt, ExplanationClaims)
            summary = polished.summary
            if polished.claims:
                claims = list(polished.claims)
        except Exception:
            summary = template_summary

    # Faithfulness guard: aspect claims must cite breakdown drivers only.
    allowed = set(breakdown.aspect_contributions)
    aspects = [a for a in aspects if a.aspect in allowed]
    if allowed:
        aspect_claims = [c for c in aspect_claims if any(asp in c for asp in allowed)]
    claims = aspect_claims + numeric_claims

    return RationalizedExplanation(
        item_id=item_id,
        summary=summary,
        aspects=aspects,
        claims=claims,
    )
