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
    head_set = set(head)
    rank_pos = {item: i for i, item in enumerate(reranked_pool)}
    ordered_head = sorted(
        head,
        key=lambda item: (rank_pos.get(item, len(reranked_pool)), head.index(item)),
    )
    return ordered_head + stage1_ranked[head_n:]


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
