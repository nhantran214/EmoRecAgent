"""LLM listwise / top-K promote / reason-then-pick rerank for Stage 2 pool."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Literal

from ..llm.client import LLMError
from ..llm.prompts import (
    RANKING_PROMOTE_JSON_SUFFIX,
    STAGE2_RERANK_PAPER_V1,
    STAGE2_RERANK_PROMOTE_PRESERVE_V1,
    STAGE2_RERANK_PROMOTE_SWAP_V1,
    STAGE2_RERANK_PROMOTE_V1,
    STAGE2_RERANK_V1,
    STAGE2_RERANK_V7_PHI,
    format_prompt,
)
from ..llm.schemas import ranking_max_tokens
from .cross_user_lookup import CrossUserLookup
from .item_metadata import ItemMeta, format_anchor_card, format_item_card
from .stage2_reason_promote import (
    annotate_cards_with_overlap,
    build_scorecard_focus,
    extract_preference_facts,
    filter_eligible_hybrid_first,
    filter_picks_hybrid_gate,
    match_snippets_to_tu,
    narrow_llm_shortlist,
    preference_tokens,
    reason_pick_promotions,
    select_lexical_argmax_top_k,
    select_lexical_first_promotion,
    token_overlap_score,
)
from .stage2_rerank import promote_preserving_head, promote_swap, promote_then_fill

if TYPE_CHECKING:
    from ..llm.client import LLMClient

logger = logging.getLogger(__name__)

RerankMode = Literal[
    "listwise", "top_k_promote", "promote_preserve", "promote_swap"
]


def build_lookup_hints(
    anchor_items: list[str],
    pool: set[str],
    lookup: CrossUserLookup,
    *,
    max_hints: int = 20,
    id_only: bool = False,
    item_meta: dict[str, ItemMeta] | None = None,
    max_name: int = 40,
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
            co_label = co_item
            if item_meta and co_item in item_meta and item_meta[co_item].name:
                name = item_meta[co_item].name
                if len(name) > max_name:
                    name = name[: max_name - 1] + "…"
                co_label = f"{co_item} ({name})"
            lines.append(
                f"users who {verb} {anchor} {bought} {co_label} (n={count})"
            )
            if len(lines) >= max_hints:
                return "\n".join(lines)
    return "\n".join(lines) if lines else "(none)"


def build_reviewed_items_str(
    reviewed_items: list[str],
    item_meta: dict[str, ItemMeta] | None = None,
    *,
    max_items: int = 12,
    max_name: int = 80,
) -> str:
    if not reviewed_items:
        return "(none)"
    cards = [
        format_anchor_card(
            item,
            (item_meta or {}).get(item),
            max_name=max_name,
        )
        for item in reviewed_items[:max_items]
    ]
    return "\n".join(cards)


def summarize_review_snippet(
    text: str,
    *,
    max_chars: int,
    t_u: str = "",
) -> str:
    """Extractive short summary of a review for compact Stage-2 cards.

    Prefers the clause/sentence with the strongest T_u lexical overlap so the
    LLM still sees the evidence needed to reason, even under a tight char budget.
    Falls back to a head truncate when no clause fits.
    """
    raw = " ".join(str(text or "").split())
    if not raw:
        return ""
    width = max(12, int(max_chars))
    if len(raw) <= width:
        return raw

    # Split into light clauses; keep order for fallback stitching.
    parts = [
        p.strip(" \t-–—")
        for p in re.split(r"[.!?;/\n]+|(?<=\w),\s+", raw)
        if p and p.strip(" \t-–—")
    ]
    if not parts:
        return raw[: width - 1] + "…"

    q = preference_tokens(t_u) if t_u else set()
    scored: list[tuple[float, int, str]] = []
    for i, part in enumerate(parts):
        score = token_overlap_score(q, part) if q else 0.0
        scored.append((score, i, part))
    scored.sort(key=lambda t: (-t[0], t[1]))

    chosen: list[tuple[int, str]] = []
    used = 0
    for score, idx, part in scored:
        if q and score <= 0.0 and chosen:
            break
        piece = part if len(part) <= width else (part[: width - 1] + "…")
        # Prefer a single best clause when budget is tight.
        if not chosen:
            chosen.append((idx, piece))
            used = len(piece)
            if used >= width or (q and score > 0.0):
                # One strong T_u-matched clause is enough for reasoning.
                if q and score > 0.0:
                    break
            continue
        sep = 2  # "; "
        if used + sep + len(piece) > width:
            break
        chosen.append((idx, piece))
        used += sep + len(piece)
        break  # at most two clauses

    if not chosen:
        return raw[: width - 1] + "…"
    chosen.sort(key=lambda t: t[0])
    summary = "; ".join(p for _, p in chosen)
    if len(summary) > width:
        summary = summary[: width - 1] + "…"
    return summary


def compact_review_snippets(
    review_snippets: dict[str, list[str]] | None,
    *,
    max_chars: int,
    t_u: str = "",
) -> dict[str, list[str]] | None:
    """Summarize each item's primary review snippet to ``max_chars``."""
    if not review_snippets:
        return review_snippets
    width = max(12, int(max_chars))
    out: dict[str, list[str]] = {}
    for item_id, snips in review_snippets.items():
        if not snips:
            out[str(item_id)] = []
            continue
        summary = summarize_review_snippet(
            snips[0], max_chars=width, t_u=t_u
        )
        out[str(item_id)] = [summary] if summary else []
    return out


def card_budget_for_pool(
    n_cards: int,
    *,
    max_name: int,
    max_cats: int,
    max_review_chars: int,
    review_snippets: dict[str, list[str]] | None,
    t_u: str = "",
) -> tuple[int, int, int, dict[str, list[str]] | None]:
    """Shrink card text when the LLM sees a large π¹ pool (e.g. top-300).

    Keeps every candidate visible. For large pools, **summarizes** reviews to a
    short T_u-grounded gist (does not drop review evidence).
    """
    n = max(0, int(n_cards))
    if n <= 40:
        return max_name, max_cats, max_review_chars, review_snippets
    if n <= 100:
        rev = min(int(max_review_chars), 40)
        return (
            min(int(max_name), 60),
            min(int(max_cats), 3),
            rev,
            compact_review_snippets(
                review_snippets, max_chars=rev, t_u=t_u
            ),
        )
    # Full-pool path (≈100–300): tighter title/cats + short review summary.
    rev = min(int(max_review_chars), 36)
    return (
        min(int(max_name), 50),
        min(int(max_cats), 2),
        rev,
        compact_review_snippets(review_snippets, max_chars=rev, t_u=t_u),
    )


def build_candidate_cards(
    pool: list[str],
    scores: dict[str, float],
    item_meta: dict[str, ItemMeta] | None = None,
    *,
    stage1_ranks: dict[str, int] | None = None,
    review_snippets: dict[str, list[str]] | None = None,
    max_name: int = 80,
    max_cats: int = 5,
    max_review_chars: int = 100,
    t_u: str = "",
    phi_scores: dict[str, float] | None = None,
    co_scores: dict[str, float] | None = None,
) -> str:
    """Build LLM cards; always keep Stage-1 rank when provided (even with titles)."""
    cards: list[str] = []
    for item in pool:
        if item_meta:
            base = format_item_card(
                item,
                float(scores.get(item, 0.0)),
                item_meta.get(item),
                max_name=max_name,
                max_cats=max_cats,
            )
        else:
            base = f"{item} | S={float(scores.get(item, 0.0)):.4f}"
        if stage1_ranks is not None and item in stage1_ranks:
            base = f"{base} | π¹_rank={stage1_ranks[item]}"
        if phi_scores is not None and item in phi_scores:
            base = f"{base} | φ={float(phi_scores[item]):.3f}"
        co_val = float((co_scores or {}).get(item, 0.0))
        if co_val > 0.0:
            base = f"{base} | co={co_val:.2f}"
        if review_snippets and max_review_chars > 0:
            snips = review_snippets.get(item) or []
            if snips:
                rev = summarize_review_snippet(
                    snips[0],
                    max_chars=max_review_chars,
                    t_u=t_u,
                )
                # Escape pipes so card parsing stays readable.
                rev = rev.replace("|", "/")
                base = f"{base} | rev={rev}"
            else:
                base = f"{base} | rev=(no match)"
        cards.append(base)
    return "\n".join(cards)


def _reason_then_pick_swap(
    llm: LLMClient | None,
    *,
    t_u: str,
    pool: list[str],
    scores: dict[str, float],
    numeric_fallback: list[str],
    item_meta: dict[str, ItemMeta] | None,
    stage1_ranks: dict[str, int] | None,
    review_snippets: dict[str, list[str]] | None,
    promote_k: int,
    protect_n: int,
    narrow_cap: int,
    card_max_name: int,
    card_max_cats: int,
    card_max_review_chars: int,
    alignment_confidence: float | None,
    reason_depth: str = "deep",
    hybrid_gate_enabled: bool = True,
    hybrid_overlap_delta: int = 1,
    hybrid_overlap_delta_out_of_band: int = 2,
    hybrid_rank_lo: int = 11,
    hybrid_rank_hi: int = 40,
    hybrid_first_enabled: bool = False,
    hybrid_min_overlap: int = 0,
    scorecard_focus_cap: int = 0,
    lexical_first_enabled: bool = False,
    lexical_first_rank_lo: int = 11,
    lexical_first_rank_hi: int = 20,
    lexical_first_overlap_delta: int = 1,
    pick_mode: str = "scorecard",
    phi_scores: dict[str, float] | None = None,
    co_scores: dict[str, float] | None = None,
) -> tuple[list[str], int, int, int, int, int, int]:
    """3+4+1 path: optional lexical-first → narrow → hybrid-first → pick → swap.

    ``pick_mode``:
      - ``scorecard``: LLM reason-then-pick
      - ``lexical_argmax``: deterministic top-k overlap champs (no LLM)
      - ``argmax_llm_override``: try quality-first LLM (may ignore T_u/overlap
        gates); fall back to lexical_argmax with ``hybrid_min_overlap``

    Returns
    ``(order, n_picks, n_hybrid_blocked, n_lexical_first, n_hybrid_first_filtered,
    n_lexical_argmax, n_llm_override)``.
    """
    base_order = numeric_fallback if numeric_fallback else pool
    prot = max(0, min(int(protect_n), len(base_order)))
    k_out = min(max(int(promote_k), 1), max(0, len(base_order) - prot) or 1)
    head_end = min(prot + k_out, len(base_order))
    head_set = set(base_order[:head_end])
    # LLM pool may be a boosted shortlist; head membership follows Stage-1 order.
    visible_outside = [item for item in pool if item not in head_set]
    if not visible_outside:
        return list(base_order), 0, 0, 0, 0, 0, 0

    matched = match_snippets_to_tu(
        review_snippets,
        t_u,
        item_ids=list(dict.fromkeys(list(pool) + list(base_order[:head_end]))),
        max_chars=card_max_review_chars,
    )
    snips_for_gate = matched or review_snippets
    displacee = list(base_order[prot:head_end])
    ranks = stage1_ranks or {}

    # Lexical-first: deterministic swap from π¹ near-miss band (skip LLM).
    if lexical_first_enabled:
        lo = int(lexical_first_rank_lo)
        hi = int(lexical_first_rank_hi)
        band = [
            item
            for item in pool
            if item not in head_set and lo <= int(ranks.get(item, 10**9)) <= hi
        ]
        band.sort(key=lambda x: (int(ranks.get(x, 10**9)), x))
        lex_pick = select_lexical_first_promotion(
            band,
            displacee_ids=displacee,
            t_u=t_u,
            item_meta=item_meta,
            review_snippets=snips_for_gate,
            stage1_ranks=ranks,
            overlap_delta=int(lexical_first_overlap_delta),
        )
        if lex_pick:
            return (
                promote_swap(
                    base_order,
                    [lex_pick],
                    promote_k=k_out,
                    protect_n=prot,
                ),
                1,
                0,
                1,
                0,
                0,
                0,
            )

    # Narrow only among outside-head visibles (prepend dummy head so helper works).
    narrow_input = list(base_order[:head_end]) + visible_outside
    shortlist = narrow_llm_shortlist(
        narrow_input,
        t_u=t_u,
        stage1_ranks=stage1_ranks,
        item_meta=item_meta,
        review_snippets=snips_for_gate,
        protect_n=prot,
        promote_k=k_out,
        narrow_cap=narrow_cap,
    )
    eligible = [item for item in shortlist if item not in head_set]
    if not eligible:
        return list(base_order), 0, 0, 0, 0, 0, 0

    mode = (
        pick_mode
        if pick_mode in ("scorecard", "lexical_argmax", "argmax_llm_override")
        else "scorecard"
    )

    def _lexical_argmax_swap(
        cand_ids: list[str],
    ) -> tuple[list[str], int, int, int] | None:
        """Return (order, n_picks, n_hf_filtered, n_argmax) or None if no champ."""
        work = list(cand_ids)
        n_hf = 0
        if hybrid_first_enabled:
            work, n_hf = filter_eligible_hybrid_first(
                work,
                displacee_ids=displacee,
                t_u=t_u,
                preference_facts=None,
                item_meta=item_meta,
                review_snippets=snips_for_gate,
                stage1_ranks=stage1_ranks,
                overlap_delta=hybrid_overlap_delta,
                overlap_delta_out_of_band=hybrid_overlap_delta_out_of_band,
                rank_lo=hybrid_rank_lo,
                rank_hi=hybrid_rank_hi,
                min_overlap=hybrid_min_overlap,
            )
        if not work:
            return None
        champs = select_lexical_argmax_top_k(
            work,
            promote_k=k_out,
            t_u=t_u,
            preference_facts=None,
            item_meta=item_meta,
            review_snippets=snips_for_gate,
            stage1_ranks=ranks,
            min_overlap=hybrid_min_overlap,
        )
        if not champs:
            return None
        return (
            promote_swap(
                base_order,
                champs,
                promote_k=k_out,
                protect_n=prot,
            ),
            len(champs),
            n_hf,
            1,
        )

    # Lexical-argmax only (strict min_overlap champions).
    if mode == "lexical_argmax":
        out = _lexical_argmax_swap(eligible)
        if out is None:
            return list(base_order), 0, 0, 0, 0, 0, 0
        order, n_picks, n_hf, n_arg = out
        return order, n_picks, 0, 0, n_hf, n_arg, 0

    # Argmax + LLM override: LLM first (may contradict T_u/overlap); else argmax.
    if mode == "argmax_llm_override":
        if llm is None:
            out = _lexical_argmax_swap(eligible)
            if out is None:
                return list(base_order), 0, 0, 0, 0, 0, 0
            order, n_picks, n_hf, n_arg = out
            return order, n_picks, 0, 0, n_hf, n_arg, 0

        c_u_str = (
            f"{float(alignment_confidence):.4f}"
            if alignment_confidence is not None
            else "n/a"
        )
        depth = reason_depth if reason_depth in ("shallow", "deep") else "deep"
        facts = extract_preference_facts(llm, t_u, depth=depth)
        # focus_cap=0 → no cut (LLM sees full eligible / π¹ pool outside head).
        focus_cap = int(scorecard_focus_cap)
        if focus_cap > 0:
            llm_eligible = build_scorecard_focus(
                eligible,
                t_u=t_u,
                preference_facts=facts,
                item_meta=item_meta,
                review_snippets=snips_for_gate,
                stage1_ranks=ranks,
                focus_cap=focus_cap,
                near_miss_lo=hybrid_rank_lo,
                near_miss_hi=hybrid_rank_hi,
                phi_scores=phi_scores,
            )
            if not llm_eligible:
                llm_eligible = list(eligible)
        else:
            llm_eligible = list(eligible)

        name_b, cats_b, rev_b, snips_b = card_budget_for_pool(
            len(llm_eligible),
            max_name=card_max_name,
            max_cats=card_max_cats,
            max_review_chars=card_max_review_chars,
            review_snippets=matched or None,
            t_u=t_u,
        )
        displacee_cards = build_candidate_cards(
            displacee,
            scores,
            item_meta,
            stage1_ranks=stage1_ranks,
            review_snippets=matched or None,
            max_name=card_max_name,
            max_cats=card_max_cats,
            max_review_chars=card_max_review_chars,
            t_u=t_u,
            phi_scores=phi_scores,
            co_scores=co_scores,
        )
        candidate_cards = build_candidate_cards(
            llm_eligible,
            scores,
            item_meta,
            stage1_ranks=stage1_ranks,
            review_snippets=snips_b,
            max_name=name_b,
            max_cats=cats_b,
            max_review_chars=rev_b,
            t_u=t_u,
            phi_scores=phi_scores,
            co_scores=co_scores,
        )
        displacee_cards = annotate_cards_with_overlap(
            displacee_cards,
            displacee,
            t_u=t_u,
            preference_facts=facts,
            displacee_ids=displacee,
            item_meta=item_meta,
            review_snippets=snips_for_gate,
        )
        candidate_cards = annotate_cards_with_overlap(
            candidate_cards,
            llm_eligible,
            t_u=t_u,
            preference_facts=facts,
            displacee_ids=displacee,
            item_meta=item_meta,
            review_snippets=snips_for_gate,
        )
        use_phi = bool(phi_scores)
        picks = reason_pick_promotions(
            llm,
            t_u=t_u,
            preference_facts=facts,
            candidate_cards=candidate_cards,
            displacee_cards=displacee_cards,
            promote_k=k_out,
            eligible_ids=llm_eligible,
            c_u=c_u_str,
            depth=depth,
            overlap_grounded=False,
            quality_override=not use_phi,
            phi_grounded=use_phi,
        )
        # Constraint override: do NOT apply hybrid lexical post-gate to LLM picks.
        if picks:
            return (
                promote_swap(
                    base_order,
                    picks,
                    promote_k=k_out,
                    protect_n=prot,
                ),
                len(picks),
                0,
                0,
                0,
                0,
                1,
            )
        if use_phi:
            # Abstain keeps φ-ranked numeric_fallback (do not lexical-argmax).
            return list(base_order), 0, 0, 0, 0, 0, 0
        out = _lexical_argmax_swap(eligible)
        if out is None:
            return list(base_order), 0, 0, 0, 0, 0, 0
        order, n_picks, n_hf, n_arg = out
        return order, n_picks, 0, 0, n_hf, n_arg, 0

    c_u_str = (
        f"{float(alignment_confidence):.4f}"
        if alignment_confidence is not None
        else "n/a"
    )
    depth = reason_depth if reason_depth in ("shallow", "deep") else "deep"
    facts = extract_preference_facts(llm, t_u, depth=depth)

    n_hf_filtered = 0
    if hybrid_first_enabled and not phi_scores:
        eligible, n_hf_filtered = filter_eligible_hybrid_first(
            eligible,
            displacee_ids=displacee,
            t_u=t_u,
            preference_facts=facts,
            item_meta=item_meta,
            review_snippets=snips_for_gate,
            stage1_ranks=stage1_ranks,
            overlap_delta=hybrid_overlap_delta,
            overlap_delta_out_of_band=hybrid_overlap_delta_out_of_band,
            rank_lo=hybrid_rank_lo,
            rank_hi=hybrid_rank_hi,
            min_overlap=hybrid_min_overlap,
        )
        if not eligible:
            # No lexical-viable cand → keep Stage-1 (abstain); skip pick call.
            return list(base_order), 0, 0, 0, n_hf_filtered, 0, 0

    # Focus LLM attention: φ top-N from the pool, else strongest lexical cands.
    focus_cap = int(scorecard_focus_cap)
    if focus_cap > 0 and len(eligible) > focus_cap:
        if phi_scores:
            eligible = build_scorecard_focus(
                eligible,
                t_u=t_u,
                preference_facts=facts,
                item_meta=item_meta,
                review_snippets=snips_for_gate,
                stage1_ranks=ranks,
                focus_cap=focus_cap,
                near_miss_lo=hybrid_rank_lo,
                near_miss_hi=hybrid_rank_hi,
                phi_scores=phi_scores,
            )
        else:
            eligible = select_lexical_argmax_top_k(
                eligible,
                promote_k=focus_cap,
                t_u=t_u,
                preference_facts=facts,
                item_meta=item_meta,
                review_snippets=snips_for_gate,
                stage1_ranks=ranks,
                min_overlap=0,
            )

    name_b, cats_b, rev_b, snips_b = card_budget_for_pool(
        len(eligible),
        max_name=card_max_name,
        max_cats=card_max_cats,
        max_review_chars=card_max_review_chars,
        review_snippets=matched or None,
        t_u=t_u,
    )
    displacee_cards = build_candidate_cards(
        displacee,
        scores,
        item_meta,
        stage1_ranks=stage1_ranks,
        review_snippets=matched or None,
        max_name=card_max_name,
        max_cats=card_max_cats,
        max_review_chars=card_max_review_chars,
        t_u=t_u,
        phi_scores=phi_scores,
        co_scores=co_scores,
    )
    candidate_cards = build_candidate_cards(
        eligible,
        scores,
        item_meta,
        stage1_ranks=stage1_ranks,
        review_snippets=snips_b,
        max_name=name_b,
        max_cats=cats_b,
        max_review_chars=rev_b,
        t_u=t_u,
        phi_scores=phi_scores,
        co_scores=co_scores,
    )
    # Always annotate ov= when hybrid-first / focus path so V3 can ground.
    overlap_grounded = bool(hybrid_first_enabled or focus_cap > 0)
    if overlap_grounded:
        displacee_cards = annotate_cards_with_overlap(
            displacee_cards,
            displacee,
            t_u=t_u,
            preference_facts=facts,
            displacee_ids=displacee,
            item_meta=item_meta,
            review_snippets=snips_for_gate,
        )
        candidate_cards = annotate_cards_with_overlap(
            candidate_cards,
            eligible,
            t_u=t_u,
            preference_facts=facts,
            displacee_ids=displacee,
            item_meta=item_meta,
            review_snippets=snips_for_gate,
        )
    picks = reason_pick_promotions(
        llm,
        t_u=t_u,
        preference_facts=facts,
        candidate_cards=candidate_cards,
        displacee_cards=displacee_cards,
        promote_k=k_out,
        eligible_ids=eligible,
        c_u=c_u_str,
        depth=depth,
        overlap_grounded=overlap_grounded and not bool(phi_scores),
        phi_grounded=bool(phi_scores),
    )
    n_blocked = 0
    if hybrid_gate_enabled and picks:
        picks, n_blocked = filter_picks_hybrid_gate(
            picks,
            displacee_ids=displacee,
            t_u=t_u,
            preference_facts=facts,
            item_meta=item_meta,
            review_snippets=snips_for_gate,
            stage1_ranks=stage1_ranks,
            overlap_delta=hybrid_overlap_delta,
            overlap_delta_out_of_band=hybrid_overlap_delta_out_of_band,
            rank_lo=hybrid_rank_lo,
            rank_hi=hybrid_rank_hi,
            min_overlap=hybrid_min_overlap,
        )
    return (
        promote_swap(
            base_order,
            picks,
            promote_k=k_out,
            protect_n=prot,
        ),
        len(picks),
        n_blocked,
        0,
        n_hf_filtered,
        0,
        0,
    )


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
    alignment_confidence: float | None = None,
    stage1_ranks: dict[str, int] | None = None,
    review_snippets: dict[str, list[str]] | None = None,
    rerank_mode: RerankMode = "listwise",
    promote_k: int = 10,
    protect_n: int = 5,
    card_max_name: int = 80,
    card_max_cats: int = 5,
    card_max_review_chars: int = 100,
    reason_then_pick: bool = False,
    narrow_cap: int = 12,
    reason_depth: str = "deep",
    hybrid_gate_enabled: bool = True,
    hybrid_overlap_delta: int = 1,
    hybrid_overlap_delta_out_of_band: int = 2,
    hybrid_rank_lo: int = 11,
    hybrid_rank_hi: int = 40,
    hybrid_first_enabled: bool = False,
    hybrid_min_overlap: int = 0,
    scorecard_focus_cap: int = 0,
    lexical_first_enabled: bool = False,
    lexical_first_rank_lo: int = 11,
    lexical_first_rank_hi: int = 20,
    lexical_first_overlap_delta: int = 1,
    pick_mode: str = "scorecard",
    swap_stats: dict[str, int] | None = None,
    phi_scores: dict[str, float] | None = None,
    co_scores: dict[str, float] | None = None,
    listwise_w_phi: float = 1.0,
    listwise_w_tu: float = 0.0,
    listwise_w_co: float = 0.0,
    listwise_w_llm: float = 0.0,
) -> list[str]:
    """Rerank pool via LLM; on failure or missing client return numeric_fallback."""
    if not pool:
        return []
    mode = (
        rerank_mode
        if rerank_mode
        in ("listwise", "top_k_promote", "promote_preserve", "promote_swap")
        else "listwise"
    )
    pick = (
        pick_mode
        if pick_mode in ("scorecard", "lexical_argmax", "argmax_llm_override")
        else "scorecard"
    )
    # Lexical-argmax / override-fallback need no LLM when client missing.
    if llm is None and not (
        mode == "promote_swap"
        and reason_then_pick
        and pick in ("lexical_argmax", "argmax_llm_override")
    ):
        return list(numeric_fallback)

    # 3+4+1 package (default on Option B promote_swap).
    if mode == "promote_swap" and reason_then_pick:
        try:
            (
                order,
                n_picks,
                n_hybrid_blocked,
                n_lex,
                n_hf,
                n_argmax,
                n_override,
            ) = _reason_then_pick_swap(
                llm,  # may be None for lexical_argmax
                t_u=t_u,
                pool=pool,
                scores=scores,
                numeric_fallback=numeric_fallback,
                item_meta=item_meta,
                stage1_ranks=stage1_ranks,
                review_snippets=review_snippets,
                promote_k=promote_k,
                protect_n=protect_n,
                narrow_cap=narrow_cap,
                card_max_name=card_max_name,
                card_max_cats=card_max_cats,
                card_max_review_chars=card_max_review_chars,
                alignment_confidence=alignment_confidence,
                reason_depth=reason_depth,
                hybrid_gate_enabled=hybrid_gate_enabled,
                hybrid_overlap_delta=hybrid_overlap_delta,
                hybrid_overlap_delta_out_of_band=hybrid_overlap_delta_out_of_band,
                hybrid_rank_lo=hybrid_rank_lo,
                hybrid_rank_hi=hybrid_rank_hi,
                hybrid_first_enabled=hybrid_first_enabled,
                hybrid_min_overlap=hybrid_min_overlap,
                scorecard_focus_cap=scorecard_focus_cap,
                lexical_first_enabled=lexical_first_enabled,
                lexical_first_rank_lo=lexical_first_rank_lo,
                lexical_first_rank_hi=lexical_first_rank_hi,
                lexical_first_overlap_delta=lexical_first_overlap_delta,
                pick_mode=pick,
                phi_scores=phi_scores,
                co_scores=co_scores,
            )
            if swap_stats is not None:
                if n_lex:
                    swap_stats["n_stage2_lexical_first"] = (
                        int(swap_stats.get("n_stage2_lexical_first", 0)) + int(n_lex)
                    )
                if n_argmax:
                    swap_stats["n_stage2_lexical_argmax"] = (
                        int(swap_stats.get("n_stage2_lexical_argmax", 0))
                        + int(n_argmax)
                    )
                if n_override:
                    swap_stats["n_stage2_llm_override"] = (
                        int(swap_stats.get("n_stage2_llm_override", 0))
                        + int(n_override)
                    )
                if n_hf:
                    swap_stats["n_stage2_hybrid_first_filtered"] = (
                        int(swap_stats.get("n_stage2_hybrid_first_filtered", 0))
                        + int(n_hf)
                    )
                if n_hybrid_blocked:
                    swap_stats["n_stage2_hybrid_blocked"] = (
                        int(swap_stats.get("n_stage2_hybrid_blocked", 0))
                        + int(n_hybrid_blocked)
                    )
                if n_picks > 0:
                    swap_stats["n_stage2_swaps"] = (
                        int(swap_stats.get("n_stage2_swaps", 0)) + 1
                    )
                else:
                    swap_stats["n_stage2_empty_picks"] = (
                        int(swap_stats.get("n_stage2_empty_picks", 0)) + 1
                    )
            return order
        except Exception as exc:  # noqa: BLE001 — fail closed to Stage-1 order
            logger.warning("stage2_reason_then_pick_fallback=true reason=%s", exc)
            if swap_stats is not None:
                swap_stats["n_stage2_empty_picks"] = (
                    int(swap_stats.get("n_stage2_empty_picks", 0)) + 1
                )
            return list(numeric_fallback)

    if llm is None:
        return list(numeric_fallback)

    # Optional T_u-matched snippets even on single-call paths.
    card_snips = review_snippets
    if review_snippets and t_u.strip():
        card_snips = match_snippets_to_tu(
            review_snippets,
            t_u,
            item_ids=list(pool),
            max_chars=card_max_review_chars,
        )

    reviewed_str = build_reviewed_items_str(
        reviewed_items, item_meta, max_name=card_max_name
    )
    lookup_hints = build_lookup_hints(
        reviewed_items,
        set(pool),
        lookup,
        id_only=id_only,
        item_meta=item_meta,
        max_name=min(40, card_max_name),
    )
    name_b, cats_b, rev_b, card_snips = card_budget_for_pool(
        len(pool),
        max_name=card_max_name,
        max_cats=card_max_cats,
        max_review_chars=card_max_review_chars,
        review_snippets=card_snips,
        t_u=t_u,
    )
    cards = build_candidate_cards(
        pool,
        scores,
        item_meta,
        stage1_ranks=stage1_ranks,
        review_snippets=card_snips,
        max_name=name_b,
        max_cats=cats_b,
        max_review_chars=rev_b,
        t_u=t_u,
        phi_scores=phi_scores,
        co_scores=co_scores,
    )
    c_u_str = (
        f"{float(alignment_confidence):.4f}"
        if alignment_confidence is not None
        else "n/a"
    )
    suffix = ""
    promote_modes = ("top_k_promote", "promote_preserve", "promote_swap")
    if mode in promote_modes:
        k_out = min(max(int(promote_k), 1), len(pool))
        prot = max(0, min(int(protect_n), len(pool)))
        if mode == "promote_swap":
            head_n = min(prot + k_out, len(pool))
            prompt = format_prompt(
                STAGE2_RERANK_PROMOTE_SWAP_V1,
                T_u=t_u.strip() or "(empty)",
                c_u=c_u_str,
                reviewed_items=reviewed_str,
                lookup_hints=lookup_hints,
                candidate_cards=cards,
                promote_k=str(k_out),
                protect_n=str(prot),
                protect_n_plus_1=str(prot + 1),
                head_n=str(head_n),
            )
        elif mode == "promote_preserve":
            prompt = format_prompt(
                STAGE2_RERANK_PROMOTE_PRESERVE_V1,
                T_u=t_u.strip() or "(empty)",
                c_u=c_u_str,
                reviewed_items=reviewed_str,
                lookup_hints=lookup_hints,
                candidate_cards=cards,
                promote_k=str(k_out),
                protect_n=str(prot),
            )
        else:
            prompt = format_prompt(
                STAGE2_RERANK_PROMOTE_V1,
                T_u=t_u.strip() or "(empty)",
                c_u=c_u_str,
                reviewed_items=reviewed_str,
                lookup_hints=lookup_hints,
                candidate_cards=cards,
                promote_k=str(k_out),
            )
        suffix = RANKING_PROMOTE_JSON_SUFFIX
        token_n = k_out
        token_cap = 1024
    elif phi_scores:
        prompt = format_prompt(
            STAGE2_RERANK_V7_PHI,
            T_u=t_u.strip() or "(empty)",
            c_u=c_u_str,
            reviewed_items=reviewed_str,
            lookup_hints=lookup_hints,
            candidate_cards=cards,
            w_phi=f"{float(listwise_w_phi):.2f}",
            w_tu=f"{float(listwise_w_tu):.2f}",
            w_co=f"{float(listwise_w_co):.2f}",
            w_llm=f"{float(listwise_w_llm):.2f}",
        )
        token_n = len(pool)
        token_cap = 4096
    elif alignment_confidence is not None:
        prompt = format_prompt(
            STAGE2_RERANK_PAPER_V1,
            T_u=t_u.strip() or "(empty)",
            c_u=c_u_str,
            reviewed_items=reviewed_str,
            lookup_hints=lookup_hints,
            candidate_cards=cards,
        )
        token_n = len(pool)
        token_cap = 512
    else:
        prompt = format_prompt(
            STAGE2_RERANK_V1,
            T_u=t_u.strip() or "(empty)",
            reviewed_items=reviewed_str,
            lookup_hints=lookup_hints,
            candidate_cards=cards,
        )
        token_n = len(pool)
        token_cap = 512

    max_tokens = ranking_num_predict or ranking_max_tokens(
        [token_n], cap=token_cap
    )
    try:
        ranked = llm.invoke_ranking_json(
            prompt,
            pool_ids=pool,
            max_tokens=max_tokens,
            suffix=suffix,
        )
    except LLMError as exc:
        logger.warning("stage2_rerank_fallback=true reason=%s", exc)
        return list(numeric_fallback)

    base_order = numeric_fallback if numeric_fallback else pool
    if mode == "listwise" and swap_stats is not None:
        if list(ranked) != list(pool):
            swap_stats["n_stage2_swaps"] = int(swap_stats.get("n_stage2_swaps", 0)) + 1
            swap_stats["n_stage2_llm_override"] = (
                int(swap_stats.get("n_stage2_llm_override", 0)) + 1
            )
        else:
            swap_stats["n_stage2_empty_picks"] = (
                int(swap_stats.get("n_stage2_empty_picks", 0)) + 1
            )
    if mode == "promote_swap":
        return promote_swap(
            base_order,
            ranked,
            promote_k=min(max(int(promote_k), 1), len(pool)),
            protect_n=max(0, min(int(protect_n), len(base_order))),
        )
    if mode == "promote_preserve":
        return promote_preserving_head(
            base_order,
            ranked,
            promote_k=min(max(int(promote_k), 1), len(pool)),
            protect_n=max(0, min(int(protect_n), len(base_order))),
        )
    if mode == "top_k_promote":
        return promote_then_fill(
            base_order,
            ranked,
            promote_k=min(max(int(promote_k), 1), len(pool)),
        )
    return ranked
