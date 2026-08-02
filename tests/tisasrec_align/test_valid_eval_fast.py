"""Parity tests for rank-based valid eval fast path."""

from __future__ import annotations

import torch

from emorecagent.eval.metrics import evaluate_hr_avg, evaluate_ranking
from emorecagent.tisasrec_align.valid_eval import (
    _single_relevant_metrics,
    _stable_descending_rank,
)


def _rank_via_argsort(scores: torch.Tensor, gold_idx: int) -> int:
    order = torch.argsort(scores, descending=True, stable=True)
    return int((order == gold_idx).nonzero(as_tuple=True)[0].item()) + 1


def test_stable_rank_matches_argsort():
    scores = torch.tensor([0.2, 0.9, 0.5, 0.9, 0.1])
    for gold_idx in range(scores.shape[0]):
        assert _stable_descending_rank(scores, gold_idx) == _rank_via_argsort(
            scores, gold_idx
        )


def test_single_relevant_metrics_match_evaluate_ranking():
    ranked = [f"i{j}" for j in range(30)]
    gold = "i7"
    relevant = {gold}
    rank = ranked.index(gold) + 1
    fast = _single_relevant_metrics(rank, pool_size=50)
    r10 = evaluate_ranking(ranked, relevant, 10)
    r20 = evaluate_ranking(ranked, relevant, 20)
    hr_block = evaluate_hr_avg(ranked, relevant)
    assert fast[0] == r10["mrr"]
    assert fast[1] == r20["mrr"]
    assert fast[2] == r10["recall"]
    assert fast[3] == r20["recall"]
    assert fast[4] == r10["ndcg"]
    assert fast[5] == r20["ndcg"]
    assert fast[6] == hr_block["hr@1"]
    assert fast[7] == hr_block["hr@3"]
    assert fast[8] == hr_block["hr@5"]
    assert fast[9] == r10["hr"]
    assert fast[10] == r20["hr"]
    assert fast[11] == hr_block["avg_hr@1,3,5"]
    assert fast[12] == (1.0 if rank <= 50 else 0.0)
