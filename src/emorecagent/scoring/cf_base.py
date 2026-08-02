"""Collaborative-filtering base score S_base(u, i).

Computed classically (not by the LLM) from the train-split interactions, which
are read from the split files — never pulled through Cypher. Two backends:
truncated-SVD matrix factorization (default) and cosine ItemKNN. Scores are
min-max normalized to [0, 1] over the candidate set so they blend cleanly with
the [0, 1] affective term.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from ..data.types import Interaction


class CFBase:
    def __init__(self, backend: str = "svd", factors: int = 64, seed: int = 42) -> None:
        if backend not in ("svd", "itemknn"):
            raise ValueError(f"Unknown CF backend: {backend}")
        self.backend = backend
        self.factors = factors
        self.seed = seed
        self._user_idx: dict[str, int] = {}
        self._item_idx: dict[str, int] = {}
        self._items: list[str] = []
        self._matrix: csr_matrix | None = None
        self._user_factors: np.ndarray | None = None
        self._item_factors: np.ndarray | None = None
        self._item_sim: np.ndarray | None = None

    def fit(self, interactions: list[Interaction]) -> "CFBase":
        users = sorted({it.user_id for it in interactions})
        items = sorted({it.item for it in interactions})
        self._user_idx = {u: i for i, u in enumerate(users)}
        self._item_idx = {it: i for i, it in enumerate(items)}
        self._items = items

        rows = [self._user_idx[it.user_id] for it in interactions]
        cols = [self._item_idx[it.item] for it in interactions]
        vals = [1.0] * len(interactions)
        mat = csr_matrix(
            (vals, (rows, cols)), shape=(len(users), len(items)), dtype=np.float64
        )
        # collapse duplicate (u,i) entries to 1.0
        mat.data[:] = 1.0
        self._matrix = mat

        if self.backend == "svd":
            n_comp = max(1, min(self.factors, min(mat.shape) - 1))
            svd = TruncatedSVD(n_components=n_comp, random_state=self.seed)
            self._user_factors = svd.fit_transform(mat)
            self._item_factors = svd.components_.T
        else:  # itemknn
            self._item_sim = cosine_similarity(mat.T)
        return self

    def _raw_score(self, user_id: str, item_id: str) -> float:
        ui = self._user_idx.get(user_id)
        ii = self._item_idx.get(item_id)
        if ui is None or ii is None:
            return 0.0
        if self.backend == "svd":
            assert self._user_factors is not None and self._item_factors is not None
            return float(self._user_factors[ui] @ self._item_factors[ii])
        assert self._item_sim is not None and self._matrix is not None
        user_row = self._matrix.getrow(ui).toarray().ravel()
        return float(self._item_sim[ii] @ user_row)

    def score(self, user_id: str, item_ids: list[str]) -> dict[str, float]:
        """Min-max normalized S_base over the given candidate items (in [0, 1])."""
        raw = {i: self._raw_score(user_id, i) for i in item_ids}
        if not raw:
            return {}
        lo, hi = min(raw.values()), max(raw.values())
        if hi - lo < 1e-12:
            return {i: 0.0 for i in item_ids}
        return {i: (v - lo) / (hi - lo) for i, v in raw.items()}
