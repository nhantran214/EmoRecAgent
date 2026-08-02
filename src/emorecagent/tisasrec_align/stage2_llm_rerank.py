"""LLM listwise rerank for Stage 2 pool."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..llm.client import LLMError
from ..llm.prompts import STAGE2_RERANK_V1, format_prompt
from ..llm.schemas import ranking_max_tokens
from .cross_user_lookup import CrossUserLookup
from .item_metadata import ItemMeta, format_item_card

if TYPE_CHECKING:
    from ..llm.client import LLMClient

logger = logging.getLogger(__name__)


def build_lookup_hints(
    anchor_items: list[str],
    pool: set[str],
    lookup: CrossUserLookup,
    *,
    max_hints: int = 20,
    id_only: bool = False,
) -> str:
    verb = "visited" if id_only else "reviewed"
    bought = "also visited" if id_only else "also bought"
    lines: list[str] = []
    for anchor in anchor_items:
        co_map = lookup.get(anchor)
        if not co_map:
            continue
        for co_item, count in sorted(co_map.items(), key=lambda kv: (-kv[1], kv[0])):
            if co_item not in pool:
                continue
            lines.append(
                f"users who {verb} {anchor} {bought} {co_item} (n={count})"
            )
            if len(lines) >= max_hints:
                return "\n".join(lines)
    return "\n".join(lines) if lines else "(none)"


def build_candidate_cards(
    pool: list[str],
    scores: dict[str, float],
    item_meta: dict[str, ItemMeta] | None = None,
) -> str:
    if not item_meta:
        cards = [
            f"{item} | S={float(scores.get(item, 0.0)):.4f}" for item in pool
        ]
        return "\n".join(cards)
    cards = [
        format_item_card(item, float(scores.get(item, 0.0)), item_meta.get(item))
        for item in pool
    ]
    return "\n".join(cards)


def llm_rerank_pool(
    llm: LLMClient | None,
    *,
    t_u: str,
    reviewed_items: list[str],
    lookup: CrossUserLookup,
    pool: list[str],
    scores: dict[str, float],
    numeric_fallback: list[str],
    ranking_num_predict: int | None = None,
    item_meta: dict[str, ItemMeta] | None = None,
    id_only: bool = False,
) -> list[str]:
    """Rerank pool via LLM; on failure or missing client return numeric_fallback."""
    if not pool:
        return []
    if llm is None:
        return list(numeric_fallback)
    reviewed_str = ", ".join(reviewed_items) if reviewed_items else "(none)"
    lookup_hints = build_lookup_hints(
        reviewed_items, set(pool), lookup, id_only=id_only
    )
    prompt = format_prompt(
        STAGE2_RERANK_V1,
        T_u=t_u.strip() or "(empty)",
        reviewed_items=reviewed_str,
        lookup_hints=lookup_hints,
        candidate_cards=build_candidate_cards(pool, scores, item_meta),
    )
    cap = ranking_num_predict or 512
    try:
        return llm.invoke_ranking_json(
            prompt,
            pool_ids=pool,
            max_tokens=ranking_max_tokens([len(pool)], cap=cap),
        )
    except LLMError as exc:
        logger.warning("stage2_rerank_fallback=true reason=%s", exc)
        return list(numeric_fallback)
