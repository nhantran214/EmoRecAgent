"""Three-agent path: ABSA → Profiling → manifesto LLM → $T_u$.

Also supports metadata-only $T_u$ for ID-only tracks (e.g. Yelp_AC).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass

from ..agents.profiling_agent import DynamicUserProfilingAgent
from ..data.types import Interaction
from ..llm.client import LLMClient
from ..llm.prompts import PREFERENCE_MANIFESTO_V1, format_prompt
from .absa_signal_source import AbsaCacheSignalSource
from .item_metadata import ItemMeta
from .review_context import PrefixReview, prefix_reviews_for_user

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PreferenceTextResult:
    T_u: str
    has_reviews: bool


def _template_tu(
    weights: dict[str, float],
    reviews: list[PrefixReview],
) -> str:
    top = sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    aspects = ", ".join(f"{a} ({w:.2f})" for a, w in top) if top else "general products"
    items = ", ".join(r.item_id for r in reviews[-3:]) if reviews else "none"
    return (
        f"User currently prioritizes {aspects} based on recent reviews "
        f"for items {items}."
    )


def _absa_summary(
    source: AbsaCacheSignalSource,
    user_id: str,
    query_ts_ms: int,
) -> str:
    lines = []
    for s in source.get_user_aspect_signals(user_id):
        if s.timestamp_ms >= query_ts_ms:
            continue
        pol = "pos" if s.polarity > 0 else "neg" if s.polarity < 0 else "neu"
        lines.append(f"- {s.aspect}: {pol}")
    return "\n".join(lines[:20]) or "(no ABSA signals)"


def generate_preference_text(
    *,
    user_id: str,
    query_ts_ms: int,
    user_interactions: list[Interaction],
    signal_source: AbsaCacheSignalSource,
    review_index: dict[tuple[str, str, int], str],
    profiling: DynamicUserProfilingAgent,
    llm: LLMClient | None = None,
    top_k_aspects: int = 5,
) -> PreferenceTextResult:
    reviews = prefix_reviews_for_user(
        user_id, user_interactions, query_ts_ms, review_index
    )
    has_reviews = len(reviews) > 0
    weights = profiling.profile(user_id, query_ts_ms, top_k_aspects, persist=False)

    if not has_reviews:
        return PreferenceTextResult(T_u="", has_reviews=False)

    if llm is None:
        return PreferenceTextResult(
            T_u=_template_tu(weights, reviews), has_reviews=True
        )

    snippets = "\n".join(
        f"- [{r.item_id}] {r.review_text[:200]}" for r in reviews[-5:]
    )
    weights_summary = ", ".join(
        f"{a}: {w:.3f}" for a, w in sorted(weights.items(), key=lambda kv: -kv[1])[:8]
    )
    prompt = format_prompt(
        PREFERENCE_MANIFESTO_V1,
        absa_summary=_absa_summary(signal_source, user_id, query_ts_ms),
        weights_summary=weights_summary or "(uniform)",
        review_snippets=snippets or "(none)",
    )
    try:
        raw = llm.invoke_text(prompt)
        for line in reversed(raw.strip().splitlines()):
            line = line.strip()
            if not line.startswith("{"):
                continue
            data = json.loads(line)
            stmt = data.get("preference_statement")
            if stmt:
                return PreferenceTextResult(T_u=str(stmt), has_reviews=True)
        return PreferenceTextResult(
            T_u=_template_tu(weights, reviews), has_reviews=True
        )
    except Exception as exc:
        logger.warning("manifesto LLM failed for %s: %s", user_id, exc)
        return PreferenceTextResult(
            T_u=_template_tu(weights, reviews), has_reviews=True
        )


def _prefix_items(
    user_id: str,
    user_interactions: list[Interaction],
    query_ts_ms: int,
) -> list[Interaction]:
    events = [
        it
        for it in user_interactions
        if it.user_id == user_id and it.timestamp < query_ts_ms
    ]
    events.sort(key=lambda it: (it.timestamp, it.item))
    return events


def _template_tu_metadata(
    prefix: list[Interaction],
    item_meta: dict[str, ItemMeta],
) -> str:
    cat_counts: Counter[str] = Counter()
    names: list[str] = []
    for it in prefix:
        meta = item_meta.get(it.item)
        if meta is None:
            continue
        if meta.name:
            names.append(meta.name)
        for cat in (c.strip() for c in meta.categories.split(",") if c.strip()):
            cat_counts[cat] += 1
    top_cats = [c for c, _ in cat_counts.most_common(5)]
    recent_names = names[-3:] if names else [it.item for it in prefix[-3:]]
    cats_str = ", ".join(top_cats) if top_cats else "general venues"
    places = ", ".join(recent_names) if recent_names else "none"
    return (
        f"User currently prefers venues in categories {cats_str} "
        f"based on recent visits to {places}."
    )


def generate_preference_text_from_metadata(
    *,
    user_id: str,
    query_ts_ms: int,
    user_interactions: list[Interaction],
    item_meta: dict[str, ItemMeta],
) -> PreferenceTextResult:
    """Build $T_u$ from RecBole item name/categories (no review/ABSA)."""
    prefix = _prefix_items(user_id, user_interactions, query_ts_ms)
    if not prefix:
        return PreferenceTextResult(T_u="", has_reviews=False)
    # Stage-2 eligibility: any prefix history counts as a preference signal
    # (even if some items lack metadata rows).
    if not any(it.item in item_meta for it in prefix):
        ids = ", ".join(it.item for it in prefix[-3:])
        return PreferenceTextResult(
            T_u=f"User recently visited items {ids}.",
            has_reviews=True,
        )
    return PreferenceTextResult(
        T_u=_template_tu_metadata(prefix, item_meta),
        has_reviews=True,
    )
