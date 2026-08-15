"""Build the Stage-1 scorer for ``emorecagent_align`` (ERA or RecBole)."""

from __future__ import annotations

from ..config import Config
from ..data.types import Interaction
from .full_rank_recommender import AlignFullRankRecommender
from .recbole_backend import RecBoleStage1Recommender
from .stage1_protocol import Stage1Scorer


def build_stage1_recommender(
    config: Config,
    train: list[Interaction],
    *,
    seed: int = 42,
    force_stage1_only: bool | None = None,
) -> Stage1Scorer:
    """Return ERA or RecBole Stage-1 backend from ``tisasrec_align.stage1_backend``.

    ``force_stage1_only`` overrides config when Stage-2 wraps a frozen Stage-1
    tower (rerank always scores with Stage-1-only semantics).
    """
    ta = config.tisasrec_align
    if force_stage1_only is not None and force_stage1_only != ta.stage1_only:
        ta = ta.model_copy(update={"stage1_only": force_stage1_only})
        config = config.model_copy(update={"tisasrec_align": ta})

    backend = getattr(ta, "stage1_backend", "era")
    if backend == "recbole":
        # Option B: RecBole TiSASRec CE for all five paper benchmarks.
        allowed = {
            "Beauty_and_Personal_Care",
            "Sports_and_Outdoors",
            "Toys_and_Games",
            "Yelp",
            "Yelp_AC",
        }
        if config.data.category not in allowed:
            raise ValueError(
                "tisasrec_align.stage1_backend='recbole' is only supported for "
                f"{sorted(allowed)} (got category={config.data.category!r})"
            )
        return RecBoleStage1Recommender.from_config(config, train, seed=seed)
    if backend != "era":
        raise ValueError(
            f"unknown tisasrec_align.stage1_backend={backend!r}; "
            "expected 'era' or 'recbole'"
        )
    return AlignFullRankRecommender.from_config(config, train, seed=seed)
