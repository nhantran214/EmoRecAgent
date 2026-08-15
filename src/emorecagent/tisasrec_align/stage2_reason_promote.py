"""Stage-2 helpers: T_u lexical overlap, shortlist narrowing, reason-then-pick.

Implements the 3+4+1 package on top of ``promote_swap``:
  3) narrow outside-head candidates by title/cats/snippet overlap with T_u
  4) pick the review snippet that best matches T_u (not the first review)
  1) two-call: extract preference facts → then pick swap ids
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from ..llm.client import LLMError
from ..llm.prompts import (
    STAGE2_EXTRACT_PREFS_V1,
    STAGE2_EXTRACT_PREFS_V2,
    STAGE2_REASON_PICK_V1,
    STAGE2_REASON_PICK_V2,
    STAGE2_REASON_PICK_V3,
    STAGE2_REASON_PICK_V4_OVERRIDE,
    STAGE2_REASON_PICK_V5_PHI,
    format_prompt,
)
from .item_metadata import ItemMeta

if TYPE_CHECKING:
    from ..llm.client import LLMClient

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-\+]{1,}", re.I)

# Common English / beauty stopwords — keep domain terms like serum, dry, etc.
_STOP = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "this",
        "that",
        "from",
        "have",
        "has",
        "was",
        "were",
        "are",
        "is",
        "been",
        "being",
        "but",
        "not",
        "you",
        "your",
        "my",
        "me",
        "we",
        "our",
        "they",
        "their",
        "its",
        "a",
        "an",
        "of",
        "in",
        "on",
        "to",
        "as",
        "at",
        "by",
        "or",
        "if",
        "so",
        "it",
        "be",
        "do",
        "does",
        "did",
        "will",
        "would",
        "can",
        "could",
        "should",
        "very",
        "just",
        "also",
        "into",
        "over",
        "after",
        "than",
        "then",
        "too",
        "all",
        "any",
        "more",
        "most",
        "some",
        "such",
        "no",
        "nor",
        "only",
        "own",
        "same",
        "other",
        "about",
        "up",
        "out",
        "off",
        "again",
        "further",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "what",
        "which",
        "who",
        "whom",
        "i",
        "he",
        "she",
        "him",
        "her",
        "them",
        "product",
        "products",
        "item",
        "items",
        "like",
        "likes",
        "love",
        "loves",
        "want",
        "wants",
        "need",
        "needs",
        "prefer",
        "prefers",
        "use",
        "uses",
        "using",
        "used",
        "good",
        "great",
        "really",
        "amazon",
    }
)


def preference_tokens(text: str) -> set[str]:
    """Lowercase alphanumeric tokens from T_u / titles, minus stopwords."""
    out: set[str] = set()
    for tok in _TOKEN_RE.findall(str(text or "").lower()):
        if tok in _STOP or len(tok) < 2:
            continue
        out.add(tok)
    return out


def token_overlap_score(query_tokens: set[str], doc_text: str) -> float:
    """Jaccard-like hit count: |Q ∩ D| / max(|Q|, 1)."""
    if not query_tokens:
        return 0.0
    doc = preference_tokens(doc_text)
    if not doc:
        return 0.0
    return float(len(query_tokens & doc)) / float(max(len(query_tokens), 1))


def select_tu_matched_snippet(
    candidates: list[str],
    t_u: str,
    *,
    max_chars: int = 100,
) -> str | None:
    """Pick the review text with highest T_u overlap; truncate for the card."""
    if not candidates:
        return None
    q = preference_tokens(t_u)
    best = ""
    best_score = -1.0
    for raw in candidates:
        text = str(raw or "").strip()
        if not text:
            continue
        score = token_overlap_score(q, text) if q else 0.0
        # Prefer any overlap; else keep first as weak fallback after loop.
        if score > best_score:
            best_score = score
            best = text
    if not best:
        best = str(candidates[0] or "").strip()
    if not best:
        return None
    if best_score <= 0.0 and q:
        # No lexical match — still return best-effort first snippet tagged later.
        pass
    width = max(16, int(max_chars))
    if len(best) > width:
        best = best[: width - 1] + "…"
    return best


def match_snippets_to_tu(
    review_snippets: dict[str, list[str]] | None,
    t_u: str,
    *,
    item_ids: list[str] | None = None,
    max_chars: int = 100,
) -> dict[str, list[str]]:
    """Per-item: choose the T_u-best snippet (single-element list for cards)."""
    if not review_snippets:
        return {}
    ids = item_ids if item_ids is not None else list(review_snippets)
    out: dict[str, list[str]] = {}
    for item in ids:
        snip = select_tu_matched_snippet(
            review_snippets.get(item) or [],
            t_u,
            max_chars=max_chars,
        )
        if snip:
            out[item] = [snip]
    return out


def item_overlap_text(
    item_id: str,
    item_meta: dict[str, ItemMeta] | None,
    review_snippets: dict[str, list[str]] | None,
) -> str:
    parts: list[str] = []
    meta = (item_meta or {}).get(item_id)
    if meta is not None:
        if meta.name:
            parts.append(meta.name)
        if meta.categories:
            parts.append(meta.categories)
    for snip in (review_snippets or {}).get(item_id) or []:
        parts.append(snip)
    return " ".join(parts)


def preference_query_tokens(
    t_u: str,
    preference_facts: dict[str, object] | None = None,
) -> set[str]:
    """Tokens used for hybrid lexical gate: T_u ∪ structured pref phrases."""
    q = set(preference_tokens(t_u))
    if not preference_facts:
        return q
    for key in (
        "must_have",
        "nice_to_have",
        "product_types",
        "ingredients",
        "brands",
        "keywords",
        "use_cases",
        "decision_rule",
    ):
        val = preference_facts.get(key)
        if val is None:
            continue
        if isinstance(val, str):
            q |= preference_tokens(val)
            continue
        if isinstance(val, (list, tuple)):
            for item in val:
                q |= preference_tokens(str(item))
    return q


def item_token_hit_count(
    query_tokens: set[str],
    item_id: str,
    item_meta: dict[str, ItemMeta] | None,
    review_snippets: dict[str, list[str]] | None,
) -> int:
    """Absolute |Q ∩ tokens(item)| for hybrid margin checks."""
    if not query_tokens:
        return 0
    text = item_overlap_text(item_id, item_meta, review_snippets)
    return len(query_tokens & preference_tokens(text))


def select_lexical_first_promotion(
    candidate_ids: list[str],
    *,
    displacee_ids: list[str],
    t_u: str,
    item_meta: dict[str, ItemMeta] | None,
    review_snippets: dict[str, list[str]] | None,
    stage1_ranks: dict[str, int] | None,
    overlap_delta: int = 1,
    require_positive_overlap: bool = True,
) -> str | None:
    """Pick the best T_u-overlap candidate that beats the displacee (no LLM).

    ``candidate_ids`` should already be restricted to the near-miss π¹ band and
    outside the frozen head. Returns ``None`` when no candidate clears the margin.
    """
    if not candidate_ids:
        return None
    q = preference_tokens(t_u)
    if not q:
        return None
    ranks = stage1_ranks or {}
    disp_hits = 0
    if displacee_ids:
        disp_hits = max(
            item_token_hit_count(q, d, item_meta, review_snippets)
            for d in displacee_ids
        )
    need = max(0, int(overlap_delta))
    best_id: str | None = None
    best_key: tuple[int, int, str] | None = None
    for item in candidate_ids:
        hits = item_token_hit_count(q, item, item_meta, review_snippets)
        if require_positive_overlap and hits <= 0:
            continue
        if hits < disp_hits + need:
            continue
        # Prefer more overlap; then better (smaller) π¹ rank; then id.
        key = (-hits, int(ranks.get(item, 10**9)), item)
        if best_key is None or key < best_key:
            best_key = key
            best_id = item
    return best_id


def select_lexical_argmax(
    candidate_ids: list[str],
    *,
    t_u: str,
    preference_facts: dict[str, object] | None = None,
    item_meta: dict[str, ItemMeta] | None,
    review_snippets: dict[str, list[str]] | None,
    stage1_ranks: dict[str, int] | None,
) -> str | None:
    """Pick the max T_u-overlap candidate (no margin check; caller filters).

    Tie-break: better (smaller) π¹ rank, then id. Used by ``lexical_argmax`` pick
    mode after hybrid-first has already constrained the eligible set.
    """
    picks = select_lexical_argmax_top_k(
        candidate_ids,
        promote_k=1,
        t_u=t_u,
        preference_facts=preference_facts,
        item_meta=item_meta,
        review_snippets=review_snippets,
        stage1_ranks=stage1_ranks,
    )
    return picks[0] if picks else None


def build_listwise_window(
    phi_order: list[str],
    *,
    t_u: str,
    item_meta: dict[str, ItemMeta] | None,
    review_snippets: dict[str, list[str]] | None,
    stage1_ranks: dict[str, int] | None,
    window_cap: int,
    overlap_inject: int = 0,
) -> list[str]:
    """φ prefix plus T_u-overlap items from the tail (listwise LLM window).

    ``overlap_inject`` slots are reserved for the highest-overlap items among
    ``phi_order[window_cap - inject:]`` so a text match that φ ranked past the
    prefix can still enter the LLM's ranking job (and therefore top-10).
    """
    if not phi_order:
        return []
    cap = max(1, min(int(window_cap), len(phi_order)))
    inject = max(0, min(int(overlap_inject), cap - 1))
    if inject <= 0:
        return list(phi_order[:cap])
    head_n = cap - inject
    head = list(phi_order[:head_n])
    rest = list(phi_order[head_n:])
    champs = select_lexical_argmax_top_k(
        rest,
        promote_k=inject,
        t_u=t_u,
        item_meta=item_meta,
        review_snippets=review_snippets,
        stage1_ranks=stage1_ranks,
        min_overlap=1,
    )
    out: list[str] = list(head)
    seen = set(out)
    for item in champs:
        if item in seen:
            continue
        out.append(item)
        seen.add(item)
        if len(out) >= cap:
            return out
    for item in rest:
        if item in seen:
            continue
        out.append(item)
        seen.add(item)
        if len(out) >= cap:
            break
    return out


def tu_channel_scores(
    item_ids: list[str],
    *,
    t_u: str,
    item_meta: dict[str, ItemMeta] | None,
    review_snippets: dict[str, list[str]] | None,
) -> dict[str, float]:
    """T_u / ABSA overlap in [0, 1] (hit fraction of query tokens)."""
    q = preference_tokens(t_u)
    if not q:
        return {item: 0.0 for item in item_ids}
    denom = float(len(q))
    return {
        item: float(item_token_hit_count(q, item, item_meta, review_snippets)) / denom
        for item in item_ids
    }


def _zscore_map(raw: dict[str, float], ids: list[str]) -> dict[str, float]:
    vals = [float(raw.get(item, 0.0)) for item in ids]
    if len(vals) < 2:
        return {item: 0.0 for item in ids}
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    sd = var**0.5
    if sd < 1e-12:
        return {item: 0.0 for item in ids}
    return {item: (float(raw.get(item, 0.0)) - mean) / sd for item in ids}


def build_weighted_window(
    item_ids: list[str],
    *,
    phi_scores: dict[str, float],
    tu_scores: dict[str, float],
    co_scores: dict[str, float],
    cap: int,
    w_phi: float = 1.0,
    w_tu: float = 0.0,
    w_co: float = 0.0,
) -> list[str]:
    """Top-``cap`` items by ``w_phi·z(φ) + w_tu·z(T_u) + w_co·z(co)``."""
    if not item_ids:
        return []
    cap_n = max(1, min(int(cap), len(item_ids)))
    z_phi = _zscore_map(phi_scores, item_ids)
    z_tu = _zscore_map(tu_scores, item_ids)
    z_co = _zscore_map(co_scores, item_ids)
    wp, wt, wc = float(w_phi), float(w_tu), float(w_co)
    return sorted(
        item_ids,
        key=lambda x: (
            -(wp * z_phi[x] + wt * z_tu[x] + wc * z_co[x]),
            -float(phi_scores.get(x, 0.0)),
            x,
        ),
    )[:cap_n]


def blend_window_ranks(
    window: list[str],
    *,
    llm_order: list[str],
    phi_scores: dict[str, float],
    tu_scores: dict[str, float],
    co_scores: dict[str, float],
    w_phi: float = 1.0,
    w_llm: float = 0.0,
    w_tu: float = 0.0,
    w_co: float = 0.0,
) -> list[str]:
    """Mix φ / LLM-rank / T_u / co inside a listwise window. φ stays heaviest."""
    if not window:
        return []
    n = len(window)
    win_set = set(window)
    llm_rank: dict[str, int] = {}
    rank = 1
    for item in llm_order:
        if item in win_set and item not in llm_rank:
            llm_rank[item] = rank
            rank += 1
    for item in window:
        if item not in llm_rank:
            llm_rank[item] = rank
            rank += 1
    llm_score = {
        item: float(n + 1 - llm_rank[item]) / float(n) for item in window
    }
    z_phi = _zscore_map(phi_scores, window)
    z_llm = _zscore_map(llm_score, window)
    z_tu = _zscore_map(tu_scores, window)
    z_co = _zscore_map(co_scores, window)
    wp, wl, wt, wc = float(w_phi), float(w_llm), float(w_tu), float(w_co)
    return sorted(
        window,
        key=lambda x: (
            -(wp * z_phi[x] + wl * z_llm[x] + wt * z_tu[x] + wc * z_co[x]),
            -float(phi_scores.get(x, 0.0)),
            x,
        ),
    )


def select_lexical_argmax_top_k(
    candidate_ids: list[str],
    *,
    promote_k: int,
    t_u: str,
    preference_facts: dict[str, object] | None = None,
    item_meta: dict[str, ItemMeta] | None,
    review_snippets: dict[str, list[str]] | None,
    stage1_ranks: dict[str, int] | None,
    min_overlap: int = 0,
) -> list[str]:
    """Return up to ``promote_k`` candidates by descending T_u overlap.

    Tie-break: better (smaller) π¹ rank, then id. Caller should already apply
    hybrid-first / margin filters. ``min_overlap`` drops cands with fewer hits.
    """
    if not candidate_ids or promote_k <= 0:
        return []
    ranks = stage1_ranks or {}
    q = preference_query_tokens(t_u, preference_facts)
    floor = max(0, int(min_overlap))
    scored: list[tuple[int, int, str]] = []
    for item in candidate_ids:
        hits = item_token_hit_count(q, item, item_meta, review_snippets) if q else 0
        if hits < floor:
            continue
        scored.append((-hits, int(ranks.get(item, 10**9)), item))
    scored.sort()
    k = min(int(promote_k), len(scored))
    return [item for _, _, item in scored[:k]]


def hybrid_lexical_allows(
    candidate_id: str,
    *,
    displacee_ids: list[str],
    t_u: str,
    preference_facts: dict[str, object] | None,
    item_meta: dict[str, ItemMeta] | None,
    review_snippets: dict[str, list[str]] | None,
    stage1_ranks: dict[str, int] | None,
    overlap_delta: int = 1,
    overlap_delta_out_of_band: int = 2,
    rank_lo: int = 11,
    rank_hi: int = 40,
    min_overlap: int = 0,
) -> bool:
    """Hard lexical gate: cand must beat displacee overlap by a rank-aware margin.

    Near-miss band ``[rank_lo, rank_hi]`` (1-indexed π¹) uses ``overlap_delta``;
    outside the band requires ``overlap_delta_out_of_band`` (stricter).
    ``min_overlap`` is an absolute floor on cand token hits (cuts weak champions).
    """
    q = preference_query_tokens(t_u, preference_facts)
    if not q:
        return False
    cand_hits = item_token_hit_count(
        q, candidate_id, item_meta, review_snippets
    )
    if cand_hits < max(0, int(min_overlap)):
        return False
    if not displacee_ids:
        return cand_hits >= max(1, int(overlap_delta), int(min_overlap))
    disp_hits = max(
        item_token_hit_count(q, d, item_meta, review_snippets)
        for d in displacee_ids
    )
    rank = int((stage1_ranks or {}).get(candidate_id, 10**9))
    in_band = int(rank_lo) <= rank <= int(rank_hi)
    need = int(overlap_delta) if in_band else int(overlap_delta_out_of_band)
    need = max(0, need)
    return cand_hits >= disp_hits + need


def filter_picks_hybrid_gate(
    picks: list[str],
    *,
    displacee_ids: list[str],
    t_u: str,
    preference_facts: dict[str, object] | None,
    item_meta: dict[str, ItemMeta] | None,
    review_snippets: dict[str, list[str]] | None,
    stage1_ranks: dict[str, int] | None,
    overlap_delta: int = 1,
    overlap_delta_out_of_band: int = 2,
    rank_lo: int = 11,
    rank_hi: int = 40,
    min_overlap: int = 0,
) -> tuple[list[str], int]:
    """Keep picks that pass :func:`hybrid_lexical_allows`.

    Returns ``(kept, n_blocked)``.
    """
    kept: list[str] = []
    blocked = 0
    for item in picks:
        ok = hybrid_lexical_allows(
            item,
            displacee_ids=displacee_ids,
            t_u=t_u,
            preference_facts=preference_facts,
            item_meta=item_meta,
            review_snippets=review_snippets,
            stage1_ranks=stage1_ranks,
            overlap_delta=overlap_delta,
            overlap_delta_out_of_band=overlap_delta_out_of_band,
            rank_lo=rank_lo,
            rank_hi=rank_hi,
            min_overlap=min_overlap,
        )
        if ok:
            kept.append(item)
        else:
            blocked += 1
    return kept, blocked


def filter_eligible_hybrid_first(
    eligible_ids: list[str],
    *,
    displacee_ids: list[str],
    t_u: str,
    preference_facts: dict[str, object] | None,
    item_meta: dict[str, ItemMeta] | None,
    review_snippets: dict[str, list[str]] | None,
    stage1_ranks: dict[str, int] | None,
    overlap_delta: int = 1,
    overlap_delta_out_of_band: int = 2,
    rank_lo: int = 11,
    rank_hi: int = 40,
    min_overlap: int = 0,
) -> tuple[list[str], int]:
    """Pre-LLM shortlist: keep only cands that beat displacee lexical margin.

    Same criterion as :func:`hybrid_lexical_allows` / post-pick gate, applied to
    the eligible set before scorecard so the LLM never sees weak overlap cands.
    Returns ``(kept, n_filtered)``.
    """
    return filter_picks_hybrid_gate(
        eligible_ids,
        displacee_ids=displacee_ids,
        t_u=t_u,
        preference_facts=preference_facts,
        item_meta=item_meta,
        review_snippets=review_snippets,
        stage1_ranks=stage1_ranks,
        overlap_delta=overlap_delta,
        overlap_delta_out_of_band=overlap_delta_out_of_band,
        rank_lo=rank_lo,
        rank_hi=rank_hi,
        min_overlap=min_overlap,
    )


def annotate_cards_with_overlap(
    cards: str,
    item_ids: list[str],
    *,
    t_u: str,
    preference_facts: dict[str, object] | None,
    displacee_ids: list[str],
    item_meta: dict[str, ItemMeta] | None,
    review_snippets: dict[str, list[str]] | None,
) -> str:
    """Append ``ov=cand/disp`` hit counts so the scorecard sees the lexical margin."""
    if not cards or not item_ids:
        return cards
    q = preference_query_tokens(t_u, preference_facts)
    if not q:
        return cards
    disp_hits = 0
    if displacee_ids:
        disp_hits = max(
            item_token_hit_count(q, d, item_meta, review_snippets)
            for d in displacee_ids
        )
    lines = cards.splitlines()
    out: list[str] = []
    for line, item in zip(lines, item_ids):
        cand_hits = item_token_hit_count(q, item, item_meta, review_snippets)
        out.append(f"{line} | ov={cand_hits}/{disp_hits}")
    # If card line count drifts, return original rather than mis-align.
    if len(out) != len(item_ids):
        return cards
    return "\n".join(out)


def narrow_llm_shortlist(
    pool: list[str],
    *,
    t_u: str,
    stage1_ranks: dict[str, int] | None,
    item_meta: dict[str, ItemMeta] | None,
    review_snippets: dict[str, list[str]] | None,
    protect_n: int,
    promote_k: int,
    narrow_cap: int,
) -> list[str]:
    """Keep head displacee slots + top-``narrow_cap`` outside-head by T_u overlap.

    Head window is ``protect_n + promote_k`` (same as ``promote_swap``). Outside
    that window, rank by lexical overlap with T_u (π¹ as tie-break). If all
    overlaps are zero, keep Stage-1 order for the outside set.
    """
    if not pool or narrow_cap <= 0:
        return list(pool)
    prot = max(0, min(int(protect_n), len(pool)))
    k = min(max(int(promote_k), 0), max(0, len(pool) - prot))
    head_end = prot + k
    head = list(pool[:head_end])
    outside = list(pool[head_end:])
    if not outside:
        return head
    q = preference_tokens(t_u)
    ranks = stage1_ranks or {}

    def sort_key(item: str) -> tuple[float, int, str]:
        text = item_overlap_text(item, item_meta, review_snippets)
        score = token_overlap_score(q, text) if q else 0.0
        # Higher overlap first; then better (smaller) π¹ rank; then id.
        return (-score, int(ranks.get(item, 10**9)), item)

    ranked_out = sorted(outside, key=sort_key)
    kept = ranked_out[: min(int(narrow_cap), len(ranked_out))]
    # Preserve relative Stage-1 order among kept for stable cards.
    kept_set = set(kept)
    kept_s1 = [item for item in outside if item in kept_set]
    return head + kept_s1


_PREF_LIST_KEYS = (
    "brands",
    "product_types",
    "ingredients",
    "use_cases",
    "must_have",
    "nice_to_have",
    "avoid",
    "keywords",
)
_PREF_KEYS = _PREF_LIST_KEYS  # alias for salvage

# Capture "key": [ ... ] even when inner quotes are messy (non-greedy to next ]).
_PREF_LIST_RE = re.compile(
    r'"(brands|product_types|ingredients|use_cases|must_have|nice_to_have|'
    r'avoid|keywords)"\s*:\s*\[(.*?)\]',
    re.I | re.S,
)
_QUOTED_STR_RE = re.compile(r'"((?:\\.|[^"\\])*)"')
_FIT_DELTA_MIN = 2
_CHAT_ROLE_PREFIX_RE = re.compile(r"^(?:user|assistant|system)\s*\n+", re.I)


def _strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _strip_chat_role_prefix(text: str) -> str:
    """Drop leading chat-template role lines when JSON follows (``user\\n{...``)."""
    t = (text or "").lstrip()
    while True:
        match = _CHAT_ROLE_PREFIX_RE.match(t)
        if not match:
            return t
        rest = t[match.end() :].lstrip()
        if not rest:
            return t
        if rest[:1] in "{[" or _CHAT_ROLE_PREFIX_RE.match(rest):
            t = rest
            continue
        return t


def _extract_json_object_span(text: str) -> str:
    """Return the first JSON object, or the unclosed prefix if truncated."""
    text = _strip_chat_role_prefix(text)
    start = 0 if text.startswith("{") else text.find("{")
    if start < 0:
        return text
    blob = text[start:]
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(blob):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
            if depth == 0:
                return blob[: i + 1]
    return blob


def _cap_runaway_json_strings(text: str, *, max_len: int = 160) -> str:
    """Truncate huge ``evidence`` / ``rationale`` values that blow the token budget."""

    def _repl(match: re.Match) -> str:
        key, val = match.group(1), match.group(2)
        if len(val) <= max_len:
            return match.group(0)
        return f'"{key}":"{val[:max_len]}"'

    return re.sub(
        r'"(evidence|rationale)"\s*:\s*"([^"]*)"',
        _repl,
        text,
        flags=re.I,
    )


def _close_truncated_json(text: str) -> str:
    """Append missing quotes / braces so a cut-off scorecard can json.loads."""
    s = (text or "").rstrip()
    if not s.startswith("{"):
        return s
    in_str = False
    escape = False
    for ch in s:
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
    if in_str:
        s += '"'
    stack: list[str] = []
    in_str = False
    escape = False
    for ch in s:
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            want = "}" if ch == "}" else "]"
            if stack[-1] == want:
                stack.pop()
    while stack:
        s += stack.pop()
    return s


def _relax_json_text(text: str) -> str:
    """Cheap fixes for common LLM JSON glitches before json.loads."""
    # Normalize curly / smart quotes that break JSON string delimiters.
    text = (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    # Trailing commas before } or ].
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def _unescape_json_str(value: str) -> str:
    try:
        return json.loads(f'"{value}"')
    except json.JSONDecodeError:
        return value.replace('\\"', '"').replace("\\n", " ").strip()


def _salvage_pref_lists(text: str) -> dict[str, list[str]]:
    """Pull string lists per preference key when full JSON parse fails."""
    out: dict[str, list[str]] = {k: [] for k in _PREF_KEYS}
    for match in _PREF_LIST_RE.finditer(text):
        key = match.group(1).lower()
        if key not in out:
            continue
        blob = match.group(2)
        vals: list[str] = []
        seen: set[str] = set()
        for sm in _QUOTED_STR_RE.finditer(blob):
            s = _unescape_json_str(sm.group(1)).strip()
            if not s or s.lower() in seen:
                continue
            seen.add(s.lower())
            vals.append(s)
            if len(vals) >= 12:
                break
        if vals:
            out[key] = vals
    return out


def _parse_json_object(raw: str) -> dict:
    text = _extract_json_object_span(_strip_code_fences(raw))
    relaxed = _relax_json_text(text)
    closed = _close_truncated_json(relaxed)
    capped = _cap_runaway_json_strings(closed)
    candidates = (text, relaxed, closed, capped)
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
        raise ValueError("expected JSON object")
    # Last resort: reconstruct preference-shaped dict from quoted list fragments.
    salvaged = _salvage_pref_lists(text)
    if any(salvaged.values()):
        return salvaged
    raise json.JSONDecodeError("could not parse JSON object", text, 0)


def _clean_pref_phrase(value: object) -> str:
    s = str(value or "").strip()
    s = re.sub(r'["\'`]', "", s)
    s = re.sub(r"\s+", " ", s).strip(" ,;")
    return s


def _empty_preference_facts(*, deep: bool) -> dict[str, object]:
    base: dict[str, object] = {
        "brands": [],
        "product_types": [],
        "ingredients": [],
        "use_cases": [],
        "avoid": [],
        "keywords": [],
    }
    if deep:
        base.update(
            {
                "must_have": [],
                "nice_to_have": [],
                "decision_rule": "",
            }
        )
    return base


def extract_preference_facts(
    llm: LLMClient,
    t_u: str,
    *,
    depth: str = "shallow",
) -> dict[str, object]:
    """Call 1: structured preference facts from T_u (empty lists on failure)."""
    deep = str(depth).lower() == "deep"
    empty = _empty_preference_facts(deep=deep)
    if deep:
        prompt = format_prompt(
            STAGE2_EXTRACT_PREFS_V2,
            T_u=t_u.strip() or "(empty)",
        )
        suffix = (
            "\n\nOutput ONE compact JSON object only (no markdown, no prose). "
            'Example: {"must_have":["serum"],"nice_to_have":["fragrance free"],'
            '"avoid":["alcohol"],"brands":[],"product_types":["serum"],'
            '"ingredients":["vitamin C"],"keywords":["serum","dry"],'
            '"decision_rule":"swap when candidate matches serum and vitamin C"}. '
            "Max 4 short ASCII phrases per list."
        )
        max_tokens = 384
    else:
        prompt = format_prompt(
            STAGE2_EXTRACT_PREFS_V1,
            T_u=t_u.strip() or "(empty)",
        )
        suffix = (
            "\n\nOutput ONE compact JSON object only (no markdown, no prose). "
            'Example shape: {"brands":[],"product_types":["serum"],'
            '"ingredients":["vitamin C"],"use_cases":["dry skin"],'
            '"avoid":[],"keywords":["serum","dry"]}. '
            "Max 4 short ASCII phrases per list."
        )
        max_tokens = 256
    try:
        raw = llm.invoke_text(prompt + suffix, max_tokens=max_tokens)
        data = _parse_json_object(raw)
    except (LLMError, json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("stage2_extract_prefs_fallback=true reason=%s", exc)
        toks = sorted(preference_tokens(t_u))
        empty["keywords"] = toks[:12]
        if deep:
            empty["decision_rule"] = (
                "prefer candidates matching keywords from T_u over displacee"
            )
        return empty
    out = dict(empty)
    list_keys = [k for k in empty if k != "decision_rule"]
    for key in list_keys:
        vals = data.get(key) or []
        if isinstance(vals, str):
            vals = [vals]
        cleaned: list[str] = []
        seen: set[str] = set()
        for v in vals:
            s = _clean_pref_phrase(v)
            if not s or s.lower() in seen:
                continue
            seen.add(s.lower())
            cleaned.append(s)
        out[key] = cleaned[:4]
    if deep:
        rule = _clean_pref_phrase(data.get("decision_rule") or "")
        # Cap decision_rule length for prompt hygiene.
        if len(rule) > 160:
            rule = rule[:159] + "…"
        out["decision_rule"] = rule
    list_nonempty = any(out.get(k) for k in list_keys)
    if not list_nonempty:
        out["keywords"] = sorted(preference_tokens(t_u))[:12]
        if deep and not out.get("decision_rule"):
            out["decision_rule"] = (
                "prefer candidates matching keywords from T_u over displacee"
            )
    return out


def format_preference_facts(facts: dict[str, object]) -> str:
    lines: list[str] = []
    for key in (
        "must_have",
        "nice_to_have",
        "brands",
        "product_types",
        "ingredients",
        "use_cases",
        "avoid",
        "keywords",
        "decision_rule",
    ):
        if key not in facts:
            continue
        val = facts.get(key)
        if key == "decision_rule":
            text = str(val or "").strip() or "(none)"
            lines.append(f"{key}: {text}")
            continue
        vals = val or []
        if isinstance(vals, str):
            vals = [vals]
        lines.append(
            f"{key}: {', '.join(str(x) for x in vals) if vals else '(none)'}"
        )
    return "\n".join(lines)


def _coerce_fit(value: object) -> int | None:
    try:
        score = int(round(float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return max(0, min(5, score))


def _resolve_item_id(raw_id: str, eligible_ids: list[str] | set[str]) -> str | None:
    """Map a possibly truncated id onto a unique eligible id (Amazon ASINs)."""
    s = str(raw_id or "").strip()
    if not s:
        return None
    eligible = list(eligible_ids)
    el_set = set(eligible)
    if s in el_set:
        return s
    # Truncation mid-string: unique prefix match (len≥4 to avoid chaos).
    if len(s) >= 4:
        hits = [e for e in eligible if e.startswith(s)]
        if len(hits) == 1:
            return hits[0]
    return None


def _parse_score_row_blob(
    blob: str,
    *,
    eligible_ids: list[str] | set[str] | None = None,
) -> dict[str, object] | None:
    """Extract id/fit/beats/evidence from a single {...} fragment."""
    id_m = re.search(r'"id"\s*:\s*"([^"]*)"', blob)
    if not id_m:
        return None
    raw_id = id_m.group(1).strip()
    if eligible_ids is not None:
        resolved = _resolve_item_id(raw_id, eligible_ids)
        if resolved is None:
            return None
        item_id = resolved
    else:
        if not raw_id:
            return None
        item_id = raw_id
    fit_m = re.search(r'"fit"\s*:\s*([0-9]+(?:\.[0-9]+)?)', blob)
    beats_m = re.search(r'"beats_displacee"\s*:\s*(true|false)', blob, re.I)
    evid_m = re.search(r'"evidence"\s*:\s*"((?:\\.|[^"\\])*)"', blob)
    row: dict[str, object] = {"id": item_id}
    if fit_m:
        row["fit"] = _coerce_fit(fit_m.group(1))
    if beats_m:
        row["beats_displacee"] = beats_m.group(1).lower() == "true"
    if evid_m:
        row["evidence"] = _unescape_json_str(evid_m.group(1))
    # Need at least fit to be useful for the local gate.
    if row.get("fit") is None:
        return None
    return row


def _salvage_scorecard(
    raw: str,
    *,
    eligible_ids: list[str],
) -> dict[str, object] | None:
    """Best-effort reconstruct scorecard JSON from truncated / broken LLM text."""
    text = _relax_json_text(_strip_code_fences(_strip_chat_role_prefix(raw or "")))
    if not text.strip():
        return None
    eligible = set(eligible_ids)

    displacee: dict[str, object] = {}
    disp_m = re.search(
        r'"displacee"\s*:\s*\{(.*?)\}(?=\s*,\s*"(?:candidates|ranked_item_ids)"|\s*\})',
        text,
        re.I | re.S,
    )
    if disp_m:
        # Displacee is a head id — not in eligible shortlist.
        row = _parse_score_row_blob(disp_m.group(1), eligible_ids=None)
        if row is not None:
            displacee = row

    candidates: list[dict[str, object]] = []
    seen: set[str] = set()
    # Prefer objects inside the candidates array when present.
    cand_region = text
    arr_m = re.search(r'"candidates"\s*:\s*\[(.*)', text, re.I | re.S)
    if arr_m:
        cand_region = arr_m.group(1)
        # Stop before ranked_item_ids if visible.
        stop = re.search(r'\]\s*,\s*"ranked_item_ids"', cand_region, re.I)
        if stop:
            cand_region = cand_region[: stop.start()]
    for obj_m in re.finditer(r"\{([^{}]{5,4000})\}", cand_region):
        row = _parse_score_row_blob(obj_m.group(1), eligible_ids=eligible)
        if row is None:
            continue
        cid = str(row.get("id") or "").strip()
        if not cid or cid not in eligible or cid in seen:
            continue
        # Skip re-capturing displacee object if it leaked into region.
        if displacee and cid == str(displacee.get("id") or ""):
            continue
        seen.add(cid)
        # Default beats/evidence when truncated mid-object.
        row.setdefault("beats_displacee", False)
        row.setdefault("evidence", "")
        candidates.append(row)

    ranked: list[str] = []
    ranked_m = re.search(
        r'"ranked_item_ids"\s*:\s*\[(.*?)\]',
        text,
        re.I | re.S,
    )
    if ranked_m:
        for sm in _QUOTED_STR_RE.finditer(ranked_m.group(1)):
            s = _resolve_item_id(_unescape_json_str(sm.group(1)), eligible)
            if s and s not in ranked:
                ranked.append(s)
    if not ranked:
        from ..llm.schemas import extract_ranked_ids_from_partial_json

        for s in extract_ranked_ids_from_partial_json(text):
            resolved = _resolve_item_id(s, eligible)
            if resolved and resolved not in ranked:
                ranked.append(resolved)

    # If model omitted ranked_item_ids but marked beats_displacee, infer order.
    if not ranked:
        disp_fit = _coerce_fit(displacee.get("fit")) if displacee else None
        scored: list[tuple[int, str]] = []
        for row in candidates:
            if not bool(row.get("beats_displacee")):
                continue
            if not _clean_pref_phrase(row.get("evidence") or ""):
                continue
            cf = _coerce_fit(row.get("fit"))
            if cf is None or disp_fit is None:
                continue
            if cf < disp_fit + _FIT_DELTA_MIN:
                continue
            scored.append((-cf, str(row["id"])))
        scored.sort()
        ranked = [cid for _, cid in scored]

    if not displacee and not candidates:
        return None
    return {
        "displacee": displacee,
        "candidates": candidates,
        "ranked_item_ids": ranked,
        "rationale": "",
    }


def _parse_scorecard_object(
    raw: str,
    *,
    eligible_ids: list[str],
) -> dict[str, object]:
    """Parse scorecard JSON; salvage partial scorecards when json.loads fails."""
    try:
        data = _parse_json_object(raw)
        if isinstance(data, dict) and (
            "displacee" in data or "candidates" in data or "ranked_item_ids" in data
        ):
            return data
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    salvaged = _salvage_scorecard(raw, eligible_ids=eligible_ids)
    if salvaged is not None:
        logger.info("stage2_scorecard_salvage=true")
        return salvaged
    raise json.JSONDecodeError("could not parse scorecard", raw or "", 0)


def _scorecard_allowed_ids(
    data: dict,
    *,
    eligible_ids: list[str],
    promote_k: int,
) -> list[str]:
    """Keep ranked_item_ids that pass local fit+2 / beats_displacee gate."""
    eligible = set(eligible_ids)
    disp = data.get("displacee") or {}
    if not isinstance(disp, dict):
        disp = {}
    disp_fit = _coerce_fit(disp.get("fit"))
    by_id: dict[str, dict] = {}
    raw_cands = data.get("candidates") or []
    if isinstance(raw_cands, dict):
        raw_cands = [raw_cands]
    for row in raw_cands:
        if not isinstance(row, dict):
            continue
        cid = str(row.get("id") or "").strip()
        if not cid or cid not in eligible:
            continue
        by_id[cid] = row

    ranked = data.get("ranked_item_ids") or []
    if isinstance(ranked, str):
        ranked = [ranked]
    out: list[str] = []
    seen: set[str] = set()
    for item in ranked:
        s = str(item).strip()
        if not s or s not in eligible or s in seen:
            continue
        row = by_id.get(s)
        if row is None:
            # Deep path requires a scorecard row; drop naked ids.
            continue
        cand_fit = _coerce_fit(row.get("fit"))
        beats = bool(row.get("beats_displacee"))
        evidence = _clean_pref_phrase(row.get("evidence") or "")
        if not beats or not evidence:
            continue
        if disp_fit is None or cand_fit is None:
            continue
        if cand_fit < disp_fit + _FIT_DELTA_MIN:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= int(promote_k):
            break
    return out


def _looks_like_prompt_echo(raw: str) -> bool:
    """True when the model echoes the Stage-2 pick prompt instead of answering."""
    if _looks_like_scorecard_json(raw):
        return False
    text = (raw or "").lstrip()
    if not text:
        return False
    # Chat-template continuation: model emits the next ``user`` turn + prompt.
    if re.match(r"(?i)^user\s*\n", text):
        return True
    if "You are the Guarded Reranking Agent" in text:
        return True
    if "Eligible candidate cards" in text and "Preference facts" in text:
        return True
    if "Reasoning procedure (do this carefully" in text:
        return True
    return False


def _looks_like_scorecard_json(raw: str) -> bool:
    """True when the reply contains a real scorecard JSON object (not a prompt echo)."""
    text = _strip_chat_role_prefix(raw or "")
    if not text or "{" not in text:
        return False
    # Full prompt reprint — placeholders, not a filled scorecard.
    if "You are the Guarded Reranking Agent" in text:
        return False
    if "Eligible candidate cards" in text and "Preference facts" in text:
        return False
    if "Reasoning procedure (do this carefully" in text:
        return False
    # Require a scored displacee block — prompt examples alone are not enough.
    if re.search(
        r'"displacee"\s*:\s*\{.{0,4000}?"fit"\s*:',
        text,
        re.I | re.S,
    ):
        return True
    if re.search(r'"ranked_item_ids"\s*:\s*\[', text, re.I):
        return True
    return False


def _looks_like_card_regurgitation(raw: str) -> bool:
    """True when the model echoes candidate-card lines instead of JSON."""
    text = raw or ""
    if _looks_like_scorecard_json(text) or _looks_like_prompt_echo(text):
        return False
    # Card lines look like: ``B0... | S=1.23 | name=...``
    if re.search(r"\bB0[A-Z0-9]{6,}\s*\|\s*S=", text):
        return True
    if "| name=" in text or "| cats=" in text or "| rev=" in text:
        return True
    return False


def _scorecard_repair_prompt(*, eligible_ids: list[str], promote_k: int) -> str:
    """Short repair prompt with few-shot shape; no product cards to echo."""
    example = (
        '{"displacee":{"id":"ID0","fit":2,"evidence":"weak match"},'
        '"candidates":[{"id":"ID1","fit":5,"beats_displacee":true,'
        '"evidence":"must have serum"}],'
        '"ranked_item_ids":["ID1"],'
        '"rationale":"ID1 beats displacee on must_have"}'
    )
    return (
        "Return ONLY one single-line JSON scorecard. No markdown, no prose, "
        "no prompt reprint, no product cards.\n"
        f"Eligible ids (use only these): {', '.join(eligible_ids)}\n"
        f"At most {promote_k} ids in ranked_item_ids (or []).\n"
        f"Example shape: {example}\n"
        "Now output your JSON object starting with { :"
    )


def build_scorecard_focus(
    eligible_ids: list[str],
    *,
    t_u: str,
    preference_facts: dict[str, object] | None,
    item_meta: dict[str, ItemMeta] | None,
    review_snippets: dict[str, list[str]] | None,
    stage1_ranks: dict[str, int] | None,
    focus_cap: int,
    near_miss_lo: int = 11,
    near_miss_hi: int = 20,
    phi_scores: dict[str, float] | None = None,
) -> list[str]:
    """Focus set for LLM: top-φ from the pool, else near-miss + overlap fill.

    When ``phi_scores`` is set, keep the ``focus_cap`` highest-φ eligible
    items (ties keep Stage-1 order). That is the φ filter over π¹[:K].
    Without φ, near-miss π¹∈[lo,hi] plus top overlap fillers (legacy).
    """
    if not eligible_ids:
        return []
    ranks = stage1_ranks or {}
    cap = max(0, int(focus_cap))
    if phi_scores:
        ordered = sorted(
            eligible_ids,
            key=lambda x: (
                -float(phi_scores.get(x, 0.0)),
                int(ranks.get(x, 10**9)),
                x,
            ),
        )
        return ordered[:cap] if cap > 0 else ordered
    near = [
        item
        for item in eligible_ids
        if int(near_miss_lo) <= int(ranks.get(item, 10**9)) <= int(near_miss_hi)
    ]
    near.sort(key=lambda x: (int(ranks.get(x, 10**9)), x))
    cap = max(0, int(focus_cap))
    fill_n = max(cap, len(near))
    by_ov = select_lexical_argmax_top_k(
        eligible_ids,
        promote_k=max(fill_n, 1),
        t_u=t_u,
        preference_facts=preference_facts,
        item_meta=item_meta,
        review_snippets=review_snippets,
        stage1_ranks=ranks,
        min_overlap=0,
    )
    out: list[str] = []
    seen: set[str] = set()
    for item in near + by_ov:
        if item in seen:
            continue
        out.append(item)
        seen.add(item)
        # Keep every near-miss; otherwise stop at focus_cap.
        if item not in near and cap > 0 and len(out) >= cap:
            break
    if cap > 0 and len(near) <= cap:
        return out[:cap] if out else out
    return out


def reason_pick_promotions(
    llm: LLMClient,
    *,
    t_u: str,
    preference_facts: dict[str, object],
    candidate_cards: str,
    displacee_cards: str,
    promote_k: int,
    eligible_ids: list[str],
    c_u: str = "n/a",
    depth: str = "shallow",
    overlap_grounded: bool = False,
    quality_override: bool = False,
    phi_grounded: bool = False,
) -> list[str]:
    """Call 2: pick ≤promote_k eligible ids; empty list means no swap."""
    if not eligible_ids or promote_k <= 0:
        return []
    deep = str(depth).lower() == "deep"
    k_out = min(int(promote_k), len(eligible_ids))
    if deep:
        if quality_override:
            template = STAGE2_REASON_PICK_V4_OVERRIDE
        elif phi_grounded:
            template = STAGE2_REASON_PICK_V5_PHI
        elif overlap_grounded:
            template = STAGE2_REASON_PICK_V3
        else:
            template = STAGE2_REASON_PICK_V2
        prompt = format_prompt(
            template,
            T_u=t_u.strip() or "(empty)",
            c_u=c_u,
            preference_facts=format_preference_facts(preference_facts),
            displacee_cards=displacee_cards or "(none)",
            candidate_cards=candidate_cards or "(none)",
            eligible_ids=", ".join(eligible_ids),
            promote_k=str(k_out),
        )
        if quality_override:
            suffix = (
                "\n\nIMPORTANT: Do NOT repeat the instructions or candidate cards. "
                "Output ONE SINGLE-LINE JSON object only. "
                "ov= is advisory — prefer card evidence / must_have over low ov. "
                f"Max {k_out} ids in ranked_item_ids (or []). "
                "Swap when fit>=displacee+2 with concrete evidence. "
                "Your entire reply must start with { and end with }."
            )
        elif phi_grounded:
            suffix = (
                "\n\nIMPORTANT: Do NOT repeat the instructions or candidate cards. "
                "Output ONE SINGLE-LINE JSON object only. "
                "phi is the primary potential signal; ABSA facts and co= are support. "
                "DEFAULT abstain: ranked_item_ids=[] unless phi(cand)>phi(displacee) "
                "AND fit>=displacee+2 AND evidence cites title/cats/rev/co. "
                f"Max {k_out} ids. "
                "Your entire reply must start with { and end with }."
            )
        elif overlap_grounded:
            suffix = (
                "\n\nIMPORTANT: Do NOT repeat the instructions or candidate cards. "
                "Output ONE SINGLE-LINE JSON object only. "
                "DEFAULT abstain: ranked_item_ids=[] unless a cand has higher ov "
                "than the displacee AND fit>=displacee+2 AND evidence cites a "
                f"must_have token from title/cats/rev. Max {k_out} ids. "
                "Evidence: <=6 ASCII words, no quotes/commas/apostrophes. "
                "Your entire reply must start with { and end with }."
            )
        else:
            suffix = (
                "\n\nIMPORTANT: Do NOT repeat the instructions or candidate cards. "
                "Output ONE SINGLE-LINE JSON object only (no markdown, no "
                "pretty-print, no newlines). Include displacee, only promising "
                "candidates (omit clear non-matches), ranked_item_ids "
                f"(max {k_out} or []), and a short rationale. "
                "Evidence: <=6 ASCII words, no quotes/commas/apostrophes. "
                "Swap only when fit(candidate) >= fit(displacee)+2 and "
                "beats_displacee is true with concrete evidence. "
                "Your entire reply must start with { and end with }."
            )
        # Headroom for up to ~30 compact candidate rows if the model over-lists.
        max_tokens = 4096
    else:
        prompt = format_prompt(
            STAGE2_REASON_PICK_V1,
            T_u=t_u.strip() or "(empty)",
            c_u=c_u,
            preference_facts=format_preference_facts(preference_facts),
            displacee_cards=displacee_cards or "(none)",
            candidate_cards=candidate_cards or "(none)",
            promote_k=str(k_out),
        )
        suffix = (
            "\n\nOutput ONE compact JSON line: "
            '{"ranked_item_ids":["id1",...],"rationale":"..."} '
            "with up to the requested ids from the eligible candidates only. "
            "Return [] if none clearly beat the displacee."
        )
        max_tokens = 384
    try:
        raw = llm.invoke_text(prompt + suffix, max_tokens=max_tokens)
    except LLMError as exc:
        logger.warning("stage2_reason_pick_fallback=true reason=%s", exc)
        return []

    if deep:
        # One repair retry when the model echoes prompt/cards or omits JSON.
        if not _looks_like_scorecard_json(raw):
            if _looks_like_prompt_echo(raw):
                why = "prompt_echo"
            elif _looks_like_card_regurgitation(raw):
                why = "card_regurgitation"
            else:
                why = "missing_scorecard_json"
            logger.warning(
                "stage2_reason_pick_retry=true reason=%s raw_head=%r",
                why,
                (raw or "")[:160],
            )
            try:
                raw = llm.invoke_text(
                    _scorecard_repair_prompt(
                        eligible_ids=eligible_ids, promote_k=k_out
                    ),
                    max_tokens=1024,
                )
            except LLMError as exc:
                logger.warning(
                    "stage2_reason_pick_fallback=true reason=retry_failed err=%s",
                    exc,
                )
                return []
            if _looks_like_prompt_echo(raw) or not _looks_like_scorecard_json(raw):
                logger.warning(
                    "stage2_reason_pick_fallback=true reason=retry_still_invalid "
                    "raw_head=%r",
                    (raw or "")[:160],
                )
                return []
        try:
            data = _parse_scorecard_object(raw, eligible_ids=eligible_ids)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning(
                "stage2_reason_pick_fallback=true reason=unparseable_scorecard "
                "err=%s raw_head=%r",
                exc,
                (raw or "")[:160],
            )
            return []
        return _scorecard_allowed_ids(
            data, eligible_ids=eligible_ids, promote_k=k_out
        )

    ranked: list[str] = []
    try:
        data = _parse_json_object(raw)
        ranked = data.get("ranked_item_ids") or []
        if isinstance(ranked, str):
            ranked = [ranked]
    except (json.JSONDecodeError, ValueError, TypeError):
        from ..llm.schemas import extract_ranked_ids_from_partial_json

        ranked = extract_ranked_ids_from_partial_json(raw)
        if not ranked:
            logger.warning(
                "stage2_reason_pick_fallback=true reason=unparseable_pick_json"
            )
            return []
    eligible = set(eligible_ids)
    seen: set[str] = set()
    out: list[str] = []
    for item in ranked:
        s = str(item).strip()
        if not s or s not in eligible or s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= k_out:
            break
    return out
