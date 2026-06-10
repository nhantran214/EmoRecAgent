"""ABSA quality metrics against a hand-labeled gold set (U4 / R13)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..llm.schemas import AbsaTriple, TripleSet


@dataclass(frozen=True, slots=True)
class TripleKey:
    aspect: str
    sentiment: str

    @classmethod
    def from_triple(cls, t: AbsaTriple) -> "TripleKey":
        return cls(aspect=t.aspect.lower(), sentiment=t.sentiment)


def _keys(triples: list[AbsaTriple]) -> set[TripleKey]:
    return {TripleKey.from_triple(t) for t in triples}


@dataclass(frozen=True, slots=True)
class TripleScores:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def triple_f1(predicted: list[AbsaTriple], gold: list[AbsaTriple]) -> TripleScores:
    pred = _keys(predicted)
    ref = _keys(gold)
    tp = len(pred & ref)
    fp = len(pred - ref)
    fn = len(ref - pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return TripleScores(
        precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn
    )


def load_gold(path: str | Path) -> dict[str, list[AbsaTriple]]:
    """Load gold labels: one JSON object per line with review_id + triples."""
    out: dict[str, list[AbsaTriple]] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = row["review_id"]
            triples = TripleSet.model_validate({"triples": row.get("triples", [])})
            out[rid] = triples.triples
    return out


def evaluate_predictions(
    predictions: dict[str, list[AbsaTriple]],
    gold: dict[str, list[AbsaTriple]],
) -> TripleScores:
    """Micro-averaged P/R/F1 over all gold review ids."""
    tp = fp = fn = 0
    for rid, gold_triples in gold.items():
        pred_triples = predictions.get(rid, [])
        s = triple_f1(pred_triples, gold_triples)
        tp += s.tp
        fp += s.fp
        fn += s.fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return TripleScores(
        precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn
    )
