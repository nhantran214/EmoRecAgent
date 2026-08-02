"""HetTiSASRec retriever adapter for the shared eval harness (no agents)."""

from __future__ import annotations

from ..baselines.base import Recommender
from ..config import Config
from ..data.types import Interaction
from .retriever import HetTiSASRecRetriever, build_hettisasrec_retriever


class HetTiSASRecEvalRecommender(Recommender):
    """Score/rank test candidates with a trained HetTiSASRec checkpoint."""

    name = "hettisasrec"

    def __init__(self, retriever: HetTiSASRecRetriever) -> None:
        self._retriever = retriever

    @classmethod
    def from_config(
        cls,
        config: Config,
        train: list[Interaction],
        *,
        seed: int = 42,
    ) -> HetTiSASRecEvalRecommender:
        return cls(
            build_hettisasrec_retriever(
                config, seed=seed, train_interactions=train
            )
        )

    @property
    def retriever(self) -> HetTiSASRecRetriever:
        return self._retriever

    def fit(self, interactions: list[Interaction]) -> HetTiSASRecEvalRecommender:
        self._retriever.fit(interactions)
        return self

    def score(self, user_id: str, candidates: list[str]) -> dict[str, float]:
        return self._retriever.score(user_id, candidates)

    def prepare_user_query(self, user_id: str, timestamp_ms: int) -> None:
        self._retriever.prepare_user_query(user_id, timestamp_ms)
