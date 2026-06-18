"""Top-K ranking metrics using internationally standard definitions.

Validated against hand-computed golden vectors in ``tests/eval/``.
"""

from __future__ import annotations

import math

METRIC_NAMES = ("recall", "ndcg", "hr", "mrr")
HR_AVG_KS: tuple[int, ...] = (1, 3, 5)
AVG_HR_KEY = "avg_hr@1,3,5"


def _require_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


def _first_hit_rank(ranked: list[str], relevant: set[str], k: int) -> int | None:
    """1-indexed rank of the first relevant item within the top-k, else None."""
    _require_k(k)
    for pos, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            return pos
    return None


def hr_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    r"""HR@K: :math:`\mathbb{1}[\exists\, i \le K : \text{rank}_i \in \mathcal{R}]`."""
    return 1.0 if _first_hit_rank(ranked, relevant, k) is not None else 0.0


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    r"""Recall@K: :math:`|\mathcal{R} \cap \text{top-}K| / |\mathcal{R}|`."""
    _require_k(k)
    if not relevant:
        return 0.0
    hits = sum(1 for item in ranked[:k] if item in relevant)
    return hits / len(relevant)


def _binary_gain(rel: int) -> float:
    r"""NDCG gain: :math:`2^{rel} - 1` (Järvelin & Kekäläinen, 2002)."""
    return float((2**rel) - 1)


def _dcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    r"""DCG@K: :math:`\sum_{i=1}^{K} \frac{2^{rel_i}-1}{\log_2(i+1)}`."""
    _require_k(k)
    dcg = 0.0
    for pos, item in enumerate(ranked[:k], start=1):
        rel = 1 if item in relevant else 0
        dcg += _binary_gain(rel) / math.log2(pos + 1)
    return dcg


def _idcg_at_k(relevant: set[str], k: int) -> float:
    """Ideal DCG@K for ``relevant``."""
    _require_k(k)
    if not relevant:
        return 0.0
    ideal_hits = min(len(relevant), k)
    return sum(_binary_gain(1) / math.log2(pos + 1) for pos in range(1, ideal_hits + 1))


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    r"""NDCG@K: :math:`\text{DCG@K} / \text{IDCG@K}`."""
    idcg = _idcg_at_k(relevant, k)
    if idcg <= 0:
        return 0.0
    return _dcg_at_k(ranked, relevant, k) / idcg


def mrr_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """MRR@K: reciprocal rank of first relevant in top-K (else 0)."""
    rank = _first_hit_rank(ranked, relevant, k)
    return 1.0 / rank if rank is not None else 0.0


def avg_hr_at_k_list(
    ranked: list[str],
    relevant: set[str],
    ks: tuple[int, ...] = HR_AVG_KS,
) -> float:
    r"""AvgHR@1,3,5: :math:`\frac{1}{|K|}\sum_{k \in K} \text{HR@}k`."""
    if not ks:
        raise ValueError("ks must be non-empty")
    for k in ks:
        _require_k(k)
    return sum(hr_at_k(ranked, relevant, k) for k in ks) / len(ks)


def evaluate_ranking(
    ranked: list[str], relevant: set[str], k: int
) -> dict[str, float]:
    """Recall, NDCG, HR, MRR at a single k (``k >= 1``)."""
    _require_k(k)
    return {
        "recall": recall_at_k(ranked, relevant, k),
        "ndcg": ndcg_at_k(ranked, relevant, k),
        "hr": hr_at_k(ranked, relevant, k),
        "mrr": mrr_at_k(ranked, relevant, k),
    }


def evaluate_hr_avg(
    ranked: list[str],
    relevant: set[str],
    ks: tuple[int, ...] = HR_AVG_KS,
) -> dict[str, float]:
    """HR@k for each k in ``ks`` plus ``avg_hr@1,3,5``."""
    out: dict[str, float] = {}
    for k in ks:
        out[f"hr@{k}"] = hr_at_k(ranked, relevant, k)
    out[AVG_HR_KEY] = avg_hr_at_k_list(ranked, relevant, ks)
    return out
