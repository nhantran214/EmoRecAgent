"""HGT-based candidate retriever implementing the CFScorer protocol."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..absa.normalize import normalize_aspect
from ..config import Config
from ..data.types import Interaction
from .aspect_vocab import AspectVocab, load_aspect_vocab
from .embeddings import EmbeddingStore


class HGTRetriever:
    """Loads frozen HGT embeddings; scores and retrieves items via dot product."""

    backend = "hgt"

    def __init__(
        self,
        store: EmbeddingStore,
        *,
        aspect_vocab: AspectVocab | None = None,
        pool_size: int = 50,
        seed: int = 42,
    ) -> None:
        self._store = store
        self._aspect_vocab = aspect_vocab
        self.pool_size = pool_size
        self.seed = seed
        self._user_idx = store.user_index()
        self._item_idx = store.item_index()
        self._aspect_idx = store.aspect_index()
        self._items = store.item_ids

    @classmethod
    def from_config(cls, config: Config, *, seed: int = 42) -> "HGTRetriever":
        hgt = config.hgt
        emb_dir = Path(hgt.embeddings_dir)
        if not emb_dir.exists():
            raise FileNotFoundError(
                f"HGT embeddings not found: {emb_dir}. Run `make train-hgt` first."
            )
        store = EmbeddingStore.load(emb_dir)
        vocab_path = Path(hgt.aspect_vocab_path)
        aspect_vocab = load_aspect_vocab(vocab_path) if vocab_path.exists() else None
        return cls(
            store,
            aspect_vocab=aspect_vocab,
            pool_size=hgt.pool_size,
            seed=seed,
        )

    def fit(self, interactions: list[Interaction]) -> "HGTRetriever":
        del interactions
        return self

    def _adjusted_user_vec(
        self, user_id: str, gammas: dict[str, float] | None
    ) -> np.ndarray | None:
        ui = self._user_idx.get(user_id)
        if ui is None:
            return None
        vec = self._store.user_embeddings[ui].astype(np.float64, copy=True)
        if not gammas:
            return vec
        for aspect, gamma in gammas.items():
            if abs(gamma) < 1e-12:
                continue
            key = normalize_aspect(aspect)
            ai = self._aspect_idx.get(key)
            if ai is None and self._aspect_vocab is not None:
                other = self._aspect_vocab.aspects[self._aspect_vocab.other_id]
                ai = self._aspect_idx.get(other)
            if ai is None:
                continue
            vec += gamma * self._store.aspect_embeddings[ai]
        return vec

    def _raw_scores(
        self,
        user_id: str,
        item_ids: list[str],
        gammas: dict[str, float] | None = None,
    ) -> dict[str, float]:
        uvec = self._adjusted_user_vec(user_id, gammas)
        if uvec is None:
            return {i: 0.0 for i in item_ids}
        out: dict[str, float] = {}
        for item_id in item_ids:
            ii = self._item_idx.get(item_id)
            if ii is None:
                out[item_id] = 0.0
                continue
            out[item_id] = float(uvec @ self._store.item_embeddings[ii])
        return out

    def score(
        self,
        user_id: str,
        item_ids: list[str],
        *,
        gammas: dict[str, float] | None = None,
    ) -> dict[str, float]:
        raw = self._raw_scores(user_id, item_ids, gammas)
        if not raw:
            return {}
        lo, hi = min(raw.values()), max(raw.values())
        if hi - lo < 1e-12:
            return {i: 0.0 for i in item_ids}
        return {i: (v - lo) / (hi - lo) for i, v in raw.items()}

    def retrieve(
        self,
        user_id: str,
        k: int,
        gammas: dict[str, float] | None,
        candidates: list[str],
        *,
        exclude: set[str] | None = None,
    ) -> list[str]:
        exclude = exclude or set()
        pool = [c for c in candidates if c not in exclude]
        if len(pool) <= k:
            return list(pool)
        raw = self._raw_scores(user_id, pool, gammas)
        ranked = sorted(raw.items(), key=lambda kv: (-kv[1], kv[0]))
        return [item for item, _ in ranked[:k]]
