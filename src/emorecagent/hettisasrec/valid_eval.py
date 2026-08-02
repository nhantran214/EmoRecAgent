"""Full-catalog valid ranking metrics for HetTiSASRec training."""

from __future__ import annotations

import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from ..eval.metrics import HR_AVG_KS, evaluate_hr_avg, evaluate_ranking
from .model import HetTiSASRecModel
from .sequence_data import ValidEvalCase

logger = logging.getLogger(__name__)

EARLY_STOP_METRICS: dict[str, str] = {
    "valid_pool_recall@50": "pool_recall",
    "valid_mrr": "link_mrr",
    "valid_link_recall@20": "link_recall_at_20",
}


@dataclass(frozen=True, slots=True)
class ValidMetrics:
    link_mrr: float
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
            f"pool@{self.pool_size}={self.pool_recall:.4f} "
            f"mrr={self.link_mrr:.4f} recall@20={self.link_recall_at_20:.4f} "
            f"ndcg@10={self.link_ndcg_at_10:.4f}"
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
    """Sanity baseline: rank by train interaction frequency."""
    if not cases:
        return 0.0
    user_seen = build_user_seen(train_pairs)
    item_counts = Counter(i for _, i in train_pairs)
    popularity = sorted(item_counts.keys(), key=lambda i: (-item_counts[i], i))
    sample = subsample_cases(cases, max_pairs=max_pairs, seed=seed)
    n_items = len(item_ids)
    all_locals = list(range(1, n_items + 1))
    hits = 0.0
    for case in sample:
        if mask_train_seen:
            forbidden = user_seen.get(case.user_local, set())
            candidates = [i for i in all_locals if i not in forbidden]
        else:
            candidates = all_locals
        if case.gold_local not in candidates:
            continue
        ranked = [i for i in popularity if i in set(candidates)]
        ranked.extend(i for i in candidates if i not in set(ranked))
        try:
            rank = ranked.index(case.gold_local) + 1
        except ValueError:
            rank = len(candidates) + 1
        hits += 1.0 if rank <= pool_size else 0.0
    return hits / len(sample) if sample else 0.0


@torch.no_grad()
def evaluate_valid_cases(
    model: HetTiSASRecModel,
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

    mrr_vals: list[float] = []
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
                    mrr_vals.append(0.0)
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
            order = torch.argsort(scores[row], descending=True, stable=True)
            rank = int((order == gold_idx).nonzero(as_tuple=True)[0].item()) + 1

            ranked = [item_ids[int(i)] for i in order.cpu().tolist()]
            gold_id = item_ids[gold_idx]
            relevant = {gold_id}

            mrr_vals.append(1.0 / rank)
            r20 = evaluate_ranking(ranked, relevant, 20)
            recall20.append(r20["recall"])
            ndcg10.append(evaluate_ranking(ranked, relevant, 10)["ndcg"])
            ndcg20.append(r20["ndcg"])
            hr_block = evaluate_hr_avg(ranked, relevant, HR_AVG_KS)
            hr1.append(hr_block["hr@1"])
            hr3.append(hr_block["hr@3"])
            hr5.append(hr_block["hr@5"])
            hr10.append(evaluate_ranking(ranked, relevant, 10)["hr"])
            hr20.append(r20["hr"])
            avg_hr.append(hr_block["avg_hr@1,3,5"])
            pool_hits.append(1.0 if rank <= pool_size else 0.0)

    n = len(sample)
    return ValidMetrics(
        link_mrr=float(np.mean(mrr_vals)) if n else 0.0,
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


# Backward-compatible alias
evaluate_valid_pairs = evaluate_valid_cases
