"""Full-catalog valid ranking metrics for TiSASRec Stage 1 training."""

from __future__ import annotations

import math
import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from ..eval.metrics import HR_AVG_KS, evaluate_hr_avg, evaluate_ranking
from .model import TiSASRecModel
from .sequence_data import UserBatchEvalCase, ValidEvalCase

logger = logging.getLogger(__name__)

EARLY_STOP_METRICS: dict[str, str] = {
    "valid_pool_recall@50": "pool_recall",
    "valid_mrr": "link_mrr",
    "valid_link_recall@10": "link_recall_at_10",
    "valid_link_recall@20": "link_recall_at_20",
    "valid_link_hr@10": "link_hr_at_10",
    "valid_link_ndcg@10": "link_ndcg_at_10",
    "valid_avg_hr@1,3,5": "avg_hr_at_1_3_5",
}


@dataclass(frozen=True, slots=True)
class ValidMetrics:
    link_mrr: float
    link_mrr_at_10: float
    link_mrr_at_20: float
    link_recall_at_10: float
    link_recall_at_20: float
    link_ndcg_at_10: float
    link_ndcg_at_20: float
    link_hr_at_1: float
    link_hr_at_3: float
    link_hr_at_5: float
    link_hr_at_10: float
    link_hr_at_20: float
    avg_hr_at_1_3_5: float
    pool_recall: float
    n_pairs_eval: int
    n_valid_pairs_total: int
    pool_size: int

    @classmethod
    def empty(cls, *, pool_size: int = 50) -> ValidMetrics:
        return cls(
            link_mrr=0.0,
            link_mrr_at_10=0.0,
            link_mrr_at_20=0.0,
            link_recall_at_10=0.0,
            link_recall_at_20=0.0,
            link_ndcg_at_10=0.0,
            link_ndcg_at_20=0.0,
            link_hr_at_1=0.0,
            link_hr_at_3=0.0,
            link_hr_at_5=0.0,
            link_hr_at_10=0.0,
            link_hr_at_20=0.0,
            avg_hr_at_1_3_5=0.0,
            pool_recall=0.0,
            n_pairs_eval=0,
            n_valid_pairs_total=0,
            pool_size=pool_size,
        )

    def early_stop_value(self, metric_name: str) -> float:
        key = EARLY_STOP_METRICS.get(metric_name)
        if key is None:
            raise ValueError(
                f"unknown early_stop_metric={metric_name!r}; "
                f"supported: {sorted(EARLY_STOP_METRICS)}"
            )
        return float(getattr(self, key))

    def format_line(self) -> str:
        return (
            f"hr@10={self.link_hr_at_10:.4f} recall@10={self.link_recall_at_10:.4f} "
            f"mrr@10={self.link_mrr_at_10:.4f} ndcg@10={self.link_ndcg_at_10:.4f} "
            f"recall@20={self.link_recall_at_20:.4f} mrr@20={self.link_mrr_at_20:.4f}"
        )


def normalize_early_stop_metric(name: str) -> str:
    if name in EARLY_STOP_METRICS:
        return name
    raise ValueError(
        f"unknown early_stop_metric={name!r}; supported: {sorted(EARLY_STOP_METRICS)}"
    )


def build_user_seen(train_pairs: list[tuple[int, int]]) -> dict[int, set[int]]:
    seen: dict[int, set[int]] = defaultdict(set)
    for u, i in train_pairs:
        seen[u].add(i)
    return seen


def subsample_cases(
    cases: list[ValidEvalCase],
    *,
    max_pairs: int,
    seed: int,
) -> list[ValidEvalCase]:
    if len(cases) <= max_pairs:
        return list(cases)
    rng = random.Random(seed)
    idx = list(range(len(cases)))
    rng.shuffle(idx)
    return [cases[i] for i in idx[:max_pairs]]


def _build_user_context(
    history: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    *,
    maxlen: int,
    time_span: int,
) -> tuple[np.ndarray, np.ndarray]:
    from ..sequential.seq_utils import compute_repos

    seq = np.zeros([maxlen], dtype=np.int32)
    time_seq = np.zeros([maxlen], dtype=np.int32)
    idx = maxlen - 1
    for item_local, norm_ts in reversed(history):
        seq[idx] = item_local
        time_seq[idx] = norm_ts
        idx -= 1
        if idx == -1:
            break
    return seq, compute_repos(time_seq, time_span)


def subsample_user_batch_cases(
    cases: list[UserBatchEvalCase],
    *,
    max_users: int,
    seed: int,
) -> list[UserBatchEvalCase]:
    if len(cases) <= max_users:
        return list(cases)
    rng = random.Random(seed)
    idx = list(range(len(cases)))
    rng.shuffle(idx)
    return [cases[i] for i in idx[:max_users]]


def _ranked_locals_from_scores(
    row_scores: torch.Tensor,
    *,
    forbidden: set[int],
) -> list[int]:
    """Stable descending local item ids (1-based), skipping forbidden."""
    neg_inf = torch.finfo(row_scores.dtype).min
    masked = row_scores.clone()
    for loc in forbidden:
        if 1 <= loc <= masked.shape[0]:
            masked[loc - 1] = neg_inf
    order = torch.argsort(masked, descending=True, stable=True)
    return [int(loc) + 1 for loc in order.tolist()]


def _user_batch_metrics(
    ranked_locals: list[int],
    relevant_locals: set[int],
    *,
    pool_size: int,
    idx_to_item: dict[int, str],
) -> tuple[float, float, float, float, float, float, float, float, float, float, float, float, float]:
    ranked = [
        idx_to_item[i] for i in ranked_locals if i in idx_to_item
    ]
    relevant = {idx_to_item[i] for i in relevant_locals if i in idx_to_item}
    if not ranked or not relevant:
        return (0.0,) * 13

    r10 = evaluate_ranking(ranked, relevant, 10)
    r20 = evaluate_ranking(ranked, relevant, 20)
    hr_block = evaluate_hr_avg(ranked, relevant)
    pool_hit = 0.0
    for loc in relevant_locals:
        try:
            rank = ranked_locals.index(loc) + 1
        except ValueError:
            continue
        if rank <= pool_size:
            pool_hit = 1.0
            break
    return (
        r10["mrr"],
        r20["mrr"],
        r10["recall"],
        r20["recall"],
        r10["ndcg"],
        r20["ndcg"],
        hr_block["hr@1"],
        hr_block["hr@3"],
        hr_block["hr@5"],
        r10["hr"],
        r20["hr"],
        hr_block["avg_hr@1,3,5"],
        pool_hit,
    )


@torch.no_grad()
def evaluate_user_batch_cases(
    model: TiSASRecModel,
    cases: list[UserBatchEvalCase],
    item_ids: list[str],
    train_pairs: list[tuple[int, int]],
    *,
    device: torch.device,
    pool_size: int = 50,
    max_users: int | None = None,
    seed: int = 42,
    mask_train_seen: bool = True,
    maxlen: int = 50,
    time_span: int = 256,
    eval_batch_size: int = 64,
) -> ValidMetrics:
    """Full-catalog user-batch eval (baseline Protocol B, ``user_mean`` aggregation)."""
    if not cases:
        return ValidMetrics.empty(pool_size=pool_size)

    cap = max_users if max_users is not None else len(cases)
    sample = subsample_user_batch_cases(cases, max_users=cap, seed=seed)
    n_target_rows = sum(c.n_target_rows for c in cases)

    model.eval()
    user_seen = build_user_seen(train_pairs)
    n_items = len(item_ids)
    idx_to_item = {i + 1: item_ids[i] for i in range(n_items)}

    item_table = model.all_item_embeddings()
    catalog = item_table[1 : n_items + 1]

    mrr10: list[float] = []
    mrr20: list[float] = []
    recall10: list[float] = []
    recall20: list[float] = []
    ndcg10: list[float] = []
    ndcg20: list[float] = []
    hr1: list[float] = []
    hr3: list[float] = []
    hr5: list[float] = []
    hr10: list[float] = []
    hr20: list[float] = []
    avg_hr: list[float] = []
    pool_hits: list[float] = []

    for start in range(0, len(sample), max(1, eval_batch_size)):
        batch = sample[start : start + eval_batch_size]
        seqs = []
        time_mats = []
        for case in batch:
            seq, time_mat = _build_user_context(
                case.history, maxlen=maxlen, time_span=time_span
            )
            seqs.append(seq)
            time_mats.append(time_mat)

        seq_t = torch.as_tensor(np.stack(seqs), dtype=torch.long, device=device)
        time_t = torch.as_tensor(np.stack(time_mats), dtype=torch.long, device=device)
        hu = model.user_repr(seq_t, time_t, item_table=item_table)
        scores = hu @ catalog.T

        for row, case in enumerate(batch):
            forbidden = user_seen.get(case.user_local, set()) if mask_train_seen else set()
            ranked_locals = _ranked_locals_from_scores(
                scores[row], forbidden=forbidden
            )
            row_metrics = _user_batch_metrics(
                ranked_locals,
                set(case.relevant_locals),
                pool_size=pool_size,
                idx_to_item=idx_to_item,
            )
            mrr10.append(row_metrics[0])
            mrr20.append(row_metrics[1])
            recall10.append(row_metrics[2])
            recall20.append(row_metrics[3])
            ndcg10.append(row_metrics[4])
            ndcg20.append(row_metrics[5])
            hr1.append(row_metrics[6])
            hr3.append(row_metrics[7])
            hr5.append(row_metrics[8])
            hr10.append(row_metrics[9])
            hr20.append(row_metrics[10])
            avg_hr.append(row_metrics[11])
            pool_hits.append(row_metrics[12])

    n = len(sample)
    mean_mrr10 = float(np.mean(mrr10)) if n else 0.0
    return ValidMetrics(
        link_mrr=mean_mrr10,
        link_mrr_at_10=mean_mrr10,
        link_mrr_at_20=float(np.mean(mrr20)) if n else 0.0,
        link_recall_at_10=float(np.mean(recall10)) if n else 0.0,
        link_recall_at_20=float(np.mean(recall20)) if n else 0.0,
        link_ndcg_at_10=float(np.mean(ndcg10)) if n else 0.0,
        link_ndcg_at_20=float(np.mean(ndcg20)) if n else 0.0,
        link_hr_at_1=float(np.mean(hr1)) if n else 0.0,
        link_hr_at_3=float(np.mean(hr3)) if n else 0.0,
        link_hr_at_5=float(np.mean(hr5)) if n else 0.0,
        link_hr_at_10=float(np.mean(hr10)) if n else 0.0,
        link_hr_at_20=float(np.mean(hr20)) if n else 0.0,
        avg_hr_at_1_3_5=float(np.mean(avg_hr)) if n else 0.0,
        pool_recall=float(np.mean(pool_hits)) if n else 0.0,
        n_pairs_eval=n,
        n_valid_pairs_total=n_target_rows,
        pool_size=pool_size,
    )


def popularity_pool_recall_user_batch(
    cases: list[UserBatchEvalCase],
    train_pairs: list[tuple[int, int]],
    *,
    pool_size: int = 50,
    max_users: int = 2048,
    seed: int = 42,
    mask_train_seen: bool = True,
) -> float:
    if not cases:
        return 0.0
    user_seen = build_user_seen(train_pairs)
    item_counts = Counter(i for _, i in train_pairs)
    popularity = sorted(item_counts.keys(), key=lambda i: (-item_counts[i], i))
    sample = subsample_user_batch_cases(cases, max_users=max_users, seed=seed)

    hits = 0.0
    for case in sample:
        forbidden = user_seen.get(case.user_local, set()) if mask_train_seen else set()
        rank = 0
        found_ranks: dict[int, int] = {}
        for i in popularity:
            if i in forbidden:
                continue
            rank += 1
            if i in case.relevant_locals:
                found_ranks[i] = rank
        if not found_ranks:
            continue
        best_rank = min(found_ranks.values())
        hits += 1.0 if best_rank <= pool_size else 0.0
    return hits / len(sample) if sample else 0.0


def popularity_pool_recall(
    cases: list[ValidEvalCase],
    train_pairs: list[tuple[int, int]],
    item_ids: list[str],
    *,
    pool_size: int = 50,
    max_pairs: int = 2048,
    seed: int = 42,
    mask_train_seen: bool = True,
) -> float:
    if not cases:
        return 0.0
    user_seen = build_user_seen(train_pairs)
    item_counts = Counter(i for _, i in train_pairs)
    popularity = sorted(item_counts.keys(), key=lambda i: (-item_counts[i], i))
    pop_set = set(popularity)
    sample = subsample_cases(cases, max_pairs=max_pairs, seed=seed)
    n_items = len(item_ids)
    tail_items = [i for i in range(1, n_items + 1) if i not in pop_set]

    hits = 0.0
    for case in sample:
        forbidden = user_seen.get(case.user_local, set()) if mask_train_seen else set()
        if mask_train_seen and case.gold_local in forbidden:
            continue

        rank = 0
        found = False
        for i in popularity:
            if i in forbidden:
                continue
            rank += 1
            if i == case.gold_local:
                found = True
                break
        if not found:
            for i in tail_items:
                if i in forbidden:
                    continue
                rank += 1
                if i == case.gold_local:
                    found = True
                    break
        if not found:
            continue
        hits += 1.0 if rank <= pool_size else 0.0
    return hits / len(sample) if sample else 0.0


def _stable_descending_rank(row_scores: torch.Tensor, gold_idx: int) -> int:
    """1-indexed rank matching ``torch.argsort(..., descending=True, stable=True)``."""
    gold_score = row_scores[gold_idx]
    higher = (row_scores > gold_score).sum()
    item_idx = torch.arange(row_scores.shape[0], device=row_scores.device)
    ties_before = ((row_scores == gold_score) & (item_idx < gold_idx)).sum()
    return int(higher + ties_before) + 1


def _single_relevant_metrics(
    rank: int,
    *,
    pool_size: int,
) -> tuple[float, float, float, float, float, float, float, float, float, float, float, float, float]:
    """Metrics for one relevant item at ``rank`` (full-catalog link metrics)."""
    hr1 = 1.0 if rank <= 1 else 0.0
    hr3 = 1.0 if rank <= 3 else 0.0
    hr5 = 1.0 if rank <= 5 else 0.0
    hr10 = 1.0 if rank <= 10 else 0.0
    hr20 = 1.0 if rank <= 20 else 0.0
    mrr10 = (1.0 / rank) if rank <= 10 else 0.0
    mrr20 = (1.0 / rank) if rank <= 20 else 0.0
    recall10 = hr10
    recall20 = hr20
    ndcg10 = (1.0 / math.log2(rank + 1)) if rank <= 10 else 0.0
    ndcg20 = (1.0 / math.log2(rank + 1)) if rank <= 20 else 0.0
    avg_hr = (hr1 + hr3 + hr5) / len(HR_AVG_KS)
    pool_hit = 1.0 if rank <= pool_size else 0.0
    return (
        mrr10,
        mrr20,
        recall10,
        recall20,
        ndcg10,
        ndcg20,
        hr1,
        hr3,
        hr5,
        hr10,
        hr20,
        avg_hr,
        pool_hit,
    )


@torch.no_grad()
def evaluate_valid_cases(
    model: TiSASRecModel,
    cases: list[ValidEvalCase],
    item_ids: list[str],
    train_pairs: list[tuple[int, int]],
    *,
    device: torch.device,
    pool_size: int = 50,
    max_pairs: int = 512,
    seed: int = 42,
    mask_train_seen: bool = True,
    maxlen: int = 50,
    time_span: int = 256,
    eval_batch_size: int = 64,
) -> ValidMetrics:
    if not cases:
        return ValidMetrics.empty(pool_size=pool_size)

    model.eval()
    user_seen = build_user_seen(train_pairs)
    sample = subsample_cases(cases, max_pairs=max_pairs, seed=seed)
    n_items = len(item_ids)

    item_table = model.all_item_embeddings()
    catalog = item_table[1 : n_items + 1]

    mrr10: list[float] = []
    mrr20: list[float] = []
    recall10: list[float] = []
    recall20: list[float] = []
    ndcg10: list[float] = []
    ndcg20: list[float] = []
    hr1: list[float] = []
    hr3: list[float] = []
    hr5: list[float] = []
    hr10: list[float] = []
    hr20: list[float] = []
    avg_hr: list[float] = []
    pool_hits: list[float] = []

    neg_inf = torch.finfo(catalog.dtype).min

    for start in range(0, len(sample), max(1, eval_batch_size)):
        batch = sample[start : start + eval_batch_size]
        seqs = []
        time_mats = []
        for case in batch:
            seq, time_mat = _build_user_context(
                case.history, maxlen=maxlen, time_span=time_span
            )
            seqs.append(seq)
            time_mats.append(time_mat)

        seq_t = torch.as_tensor(np.stack(seqs), dtype=torch.long, device=device)
        time_t = torch.as_tensor(np.stack(time_mats), dtype=torch.long, device=device)
        hu = model.user_repr(seq_t, time_t, item_table=item_table)
        scores = hu @ catalog.T

        for row, case in enumerate(batch):
            if mask_train_seen:
                forbidden = user_seen.get(case.user_local, set())
                if case.gold_local in forbidden:
                    pool_hits.append(0.0)
                    mrr10.append(0.0)
                    mrr20.append(0.0)
                    recall10.append(0.0)
                    recall20.append(0.0)
                    ndcg10.append(0.0)
                    ndcg20.append(0.0)
                    hr1.append(0.0)
                    hr3.append(0.0)
                    hr5.append(0.0)
                    hr10.append(0.0)
                    hr20.append(0.0)
                    avg_hr.append(0.0)
                    continue
                for loc in forbidden:
                    scores[row, loc - 1] = neg_inf

            gold_idx = case.gold_local - 1
            rank = _stable_descending_rank(scores[row], gold_idx)
            row_metrics = _single_relevant_metrics(rank, pool_size=pool_size)
            mrr10.append(row_metrics[0])
            mrr20.append(row_metrics[1])
            recall10.append(row_metrics[2])
            recall20.append(row_metrics[3])
            ndcg10.append(row_metrics[4])
            ndcg20.append(row_metrics[5])
            hr1.append(row_metrics[6])
            hr3.append(row_metrics[7])
            hr5.append(row_metrics[8])
            hr10.append(row_metrics[9])
            hr20.append(row_metrics[10])
            avg_hr.append(row_metrics[11])
            pool_hits.append(row_metrics[12])

    n = len(sample)
    mean_mrr10 = float(np.mean(mrr10)) if n else 0.0
    return ValidMetrics(
        link_mrr=mean_mrr10,
        link_mrr_at_10=mean_mrr10,
        link_mrr_at_20=float(np.mean(mrr20)) if n else 0.0,
        link_recall_at_10=float(np.mean(recall10)) if n else 0.0,
        link_recall_at_20=float(np.mean(recall20)) if n else 0.0,
        link_ndcg_at_10=float(np.mean(ndcg10)) if n else 0.0,
        link_ndcg_at_20=float(np.mean(ndcg20)) if n else 0.0,
        link_hr_at_1=float(np.mean(hr1)) if n else 0.0,
        link_hr_at_3=float(np.mean(hr3)) if n else 0.0,
        link_hr_at_5=float(np.mean(hr5)) if n else 0.0,
        link_hr_at_10=float(np.mean(hr10)) if n else 0.0,
        link_hr_at_20=float(np.mean(hr20)) if n else 0.0,
        avg_hr_at_1_3_5=float(np.mean(avg_hr)) if n else 0.0,
        pool_recall=float(np.mean(pool_hits)) if n else 0.0,
        n_pairs_eval=n,
        n_valid_pairs_total=len(cases),
        pool_size=pool_size,
    )
