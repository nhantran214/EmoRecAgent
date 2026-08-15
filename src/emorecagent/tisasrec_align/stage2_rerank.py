"""Stage 2 pool boost, merge, and guardrail helpers."""

from __future__ import annotations


def build_pool(stage1_ranked: list[str], k: int) -> list[str]:
    if k <= 0:
        return []
    return list(stage1_ranked[:k])


def apply_cross_user_boosts(
    pool: list[str],
    scores: dict[str, float],
    boost_weights: dict[str, float],
    boost_scale: float,
) -> list[str]:
    """Re-sort pool by Stage 1 score + optional cross-user boost."""
    if not pool:
        return []

    def sort_key(item: str) -> tuple[float, str]:
        base = float(scores.get(item, 0.0))
        boost = float(boost_weights.get(item, 0.0)) * boost_scale
        return (-(base + boost), item)

    return sorted(pool, key=sort_key)


def merge_ranking(
    reranked_pool: list[str],
    stage1_ranked: list[str],
    pool_k: int,
) -> list[str]:
    """Place reranked pool at head; append Stage 1 tail excluding pool duplicates."""
    pool_set = set(reranked_pool)
    tail = [item for item in stage1_ranked[pool_k:] if item not in pool_set]
    seen: set[str] = set()
    out: list[str] = []
    for item in list(reranked_pool) + tail:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def reorder_within_head(
    stage1_ranked: list[str],
    reranked_pool: list[str],
    head_n: int,
) -> list[str]:
    """Permute only Stage-1's top ``head_n`` items using the reranked order.

    Membership of every prefix at ``k >= head_n`` is identical to
    ``stage1_ranked`` (the head is a permutation of the same items and the tail
    is untouched), so hr@k / recall@k for ``k >= head_n`` cannot regress. Items
    inside the head keep the relative order induced by ``reranked_pool``; any
    head item missing from ``reranked_pool`` retains its Stage-1 order after the
    ranked ones.
    """
    if head_n <= 0 or not stage1_ranked:
        return list(stage1_ranked)
    head = stage1_ranked[:head_n]
    rank_pos = {item: i for i, item in enumerate(reranked_pool)}
    ordered_head = sorted(
        head,
        key=lambda item: (rank_pos.get(item, len(reranked_pool)), head.index(item)),
    )
    return ordered_head + stage1_ranked[head_n:]


def blend_rank_orders(
    stage1_order: list[str],
    llm_order: list[str],
    *,
    beta: float,
) -> list[str]:
    """Convex blend of two permutations of the same item set.

    ``score(i) = (1-β)·rank_s1(i) + β·rank_llm(i)`` (0-based ranks); lower is
    better. ``beta=0`` keeps Stage-1; ``beta=1`` keeps LLM order. Items missing
    from ``llm_order`` keep a large LLM rank so Stage-1 breaks ties.
    """
    if not stage1_order:
        return []
    b = min(max(float(beta), 0.0), 1.0)
    if b <= 1e-9:
        return list(stage1_order)
    if b >= 1.0 - 1e-9:
        # Preserve Stage-1 membership; apply LLM relative order.
        return reorder_within_head(stage1_order, llm_order, len(stage1_order))
    n = len(stage1_order)
    s1_pos = {item: i for i, item in enumerate(stage1_order)}
    llm_pos = {item: i for i, item in enumerate(llm_order)}
    return sorted(
        stage1_order,
        key=lambda item: (
            (1.0 - b) * s1_pos[item] + b * float(llm_pos.get(item, n)),
            s1_pos[item],
        ),
    )


def promote_then_fill(
    pool_order: list[str],
    llm_ranked: list[str],
    *,
    promote_k: int,
) -> list[str]:
    """Place LLM's first ``promote_k`` picks at the head; fill rest by ``pool_order``.

    Easier task for 7B than a full listwise permute: the model only needs to
    surface the best matches to T_u.
    """
    if not pool_order:
        return []
    k = min(max(int(promote_k), 0), len(pool_order))
    if k <= 0:
        return list(pool_order)
    pool_set = set(pool_order)
    seen: set[str] = set()
    head: list[str] = []
    for item in llm_ranked:
        if item not in pool_set or item in seen:
            continue
        head.append(item)
        seen.add(item)
        if len(head) >= k:
            break
    tail = [item for item in pool_order if item not in seen]
    return head + tail


def promote_preserving_head(
    pool_order: list[str],
    llm_ranked: list[str],
    *,
    promote_k: int,
    protect_n: int,
) -> list[str]:
    """Insert LLM promotions *after* a frozen Stage-1 head (A1).

    ``pool_order[:protect_n]`` keep membership and relative order in the top
    ``protect_n`` slots. Up to ``promote_k`` LLM picks that are *outside* that
    head are inserted immediately after it; the remainder follows ``pool_order``.
    Unlike ``promote_then_fill``, π¹ top-N cannot be demoted out of the head.

    Warning: insert-shift pushes ``pool_order[protect_n:protect_n+k]`` past the
    head — can drop gold from ranks protect_n+1..head out of top-10. Prefer
    ``promote_swap`` when optimizing hr@10.
    """
    if not pool_order:
        return []
    n = max(0, min(int(protect_n), len(pool_order)))
    protected = list(pool_order[:n])
    prot_set = set(protected)
    pool_set = set(pool_order)
    k = min(max(int(promote_k), 0), max(0, len(pool_order) - n))
    promos: list[str] = []
    seen: set[str] = set(prot_set)
    if k > 0:
        for item in llm_ranked:
            if item not in pool_set or item in seen or item in prot_set:
                continue
            promos.append(item)
            seen.add(item)
            if len(promos) >= k:
                break
    rest = [item for item in pool_order if item not in seen]
    return protected + promos + rest


def promote_swap(
    pool_order: list[str],
    llm_ranked: list[str],
    *,
    promote_k: int,
    protect_n: int,
) -> list[str]:
    """Swap LLM picks into the tail of the head — no insert-shift (hr@10-safe).

    Freezes ``pool_order[:protect_n]``. Up to ``promote_k`` picks from *outside*
    the current head window ``pool_order[:protect_n+promote_k]`` replace the
    last ``promote_k`` head slots. Displaced items move just below the head.
    With ``protect_n=9``, ``promote_k=1`` only slot 10 can leave top-10.
    """
    if not pool_order:
        return []
    n = len(pool_order)
    prot = max(0, min(int(protect_n), n))
    k = min(max(int(promote_k), 0), max(0, n - prot))
    if k <= 0:
        return list(pool_order)
    head_end = prot + k
    head_set = set(pool_order[:head_end])
    pool_set = set(pool_order)
    promos: list[str] = []
    seen: set[str] = set()
    for item in llm_ranked:
        if item not in pool_set or item in seen or item in head_set:
            continue
        promos.append(item)
        seen.add(item)
        if len(promos) >= k:
            break
    if not promos:
        return list(pool_order)
    # Fewer promos than k → swap only the last len(promos) head slots.
    k_eff = len(promos)
    prot_eff = head_end - k_eff
    protected = list(pool_order[:prot_eff])
    displaced = list(pool_order[prot_eff:head_end])
    rest = [item for item in pool_order[head_end:] if item not in seen]
    return protected + promos + displaced + rest


def check_guardrail(
    stage1_ranked: list[str],
    merged_ranked: list[str],
    *,
    top_n: int = 5,
    max_drop_rank: int = 10,
) -> bool:
    """Return True if merged ranking passes position-stability guardrail."""
    if not stage1_ranked or top_n <= 0:
        return True
    pos = {item: i for i, item in enumerate(merged_ranked)}
    for item in stage1_ranked[:top_n]:
        new_rank = pos.get(item)
        if new_rank is None:
            return False
        if new_rank + 1 > max_drop_rank:
            return False
    return True
