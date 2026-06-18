"""Build an in-memory recommendation context from train data + ABSA cache."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..absa.cache import AbsaCache
from ..data.types import Interaction
from ..data.review_index import build_review_index_from_scope
from ..kg.loaders import (
    aggregate_item_sentiment_from_cache,
    load_absa_cache,
    load_interactions,
)
from ..kg.memory import InMemoryKG
from ..scoring.cf_base import CFBase


@dataclass
class RecommendContext:
    kg: InMemoryKG
    cf: CFBase
    lambda_decay: float
    alpha: float
    helpful_cap: int
    affective_rescaled: bool = True
    max_reflection_iters: int = 2
    pool_size: int = 200
    top_k_aspects: int = 5
    use_reflection: bool = True
    use_dynamic_weights: bool = True
    use_aspect_term: bool = True


def _train_cutoff(train: list[Interaction]) -> int:
    if not train:
        return 0
    return max(it.timestamp for it in train)


def build_recommend_context(
    train: list[Interaction],
    *,
    seed: int = 42,
    cf_backend: str = "svd",
    cf_factors: int = 64,
    alpha: float = 0.5,
    lambda_decay: float = 0.01,
    helpful_cap: int = 10,
    affective_rescaled: bool = True,
    absa_cache_path: str | None = None,
    review_path: str | None = None,
    max_reflection_iters: int = 2,
    pool_size: int = 200,
    top_k_aspects: int = 5,
    use_reflection: bool = True,
    use_dynamic_weights: bool = True,
    use_aspect_term: bool = True,
) -> RecommendContext:
    """Hydrate an in-memory KG + CF base from the train split."""
    kg = InMemoryKG()
    load_interactions(kg, train)
    cutoff = _train_cutoff(train)

    helpful_by_key = {
        (it.user_id, it.item, it.timestamp): it.helpful_vote for it in train
    }

    cache: AbsaCache | None = None
    if absa_cache_path and Path(absa_cache_path).exists():
        cache = AbsaCache(absa_cache_path)
        if review_path and Path(review_path).exists():
            review_index = build_review_index_from_scope(train, review_path)
            load_absa_cache(kg, cache, review_index)

            item_reviews: dict[str, list[tuple[str, int]]] = {}
            for rid, (uid, item, ts) in review_index.items():
                if ts > cutoff:
                    continue
                item_reviews.setdefault(item, []).append(
                    (rid, helpful_by_key.get((uid, item, ts), 0))
                )
            aggregate_item_sentiment_from_cache(
                kg,
                cache,
                item_reviews,
                helpful_cap=helpful_cap,
                cutoff_ts=cutoff,
            )
        cache.close()

    cf = CFBase(backend=cf_backend, factors=cf_factors, seed=seed).fit(train)

    effective_alpha = alpha if use_aspect_term else 1.0

    return RecommendContext(
        kg=kg,
        cf=cf,
        lambda_decay=lambda_decay,
        alpha=effective_alpha,
        helpful_cap=helpful_cap,
        affective_rescaled=affective_rescaled,
        max_reflection_iters=max_reflection_iters if use_reflection else 0,
        pool_size=pool_size,
        top_k_aspects=top_k_aspects,
        use_reflection=use_reflection,
        use_dynamic_weights=use_dynamic_weights,
        use_aspect_term=use_aspect_term,
    )
