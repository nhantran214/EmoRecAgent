"""ABSA quality metrics against a hand-labeled gold set."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..llm.schemas import AbsaTriple, TripleSet
from .normalize import normalize_aspect

SENTIMENTS = ("positive", "negative", "neutral")


@dataclass(frozen=True, slots=True)
class TripleKey:
    aspect: str
    sentiment: str

    @classmethod
    def from_triple(cls, t: AbsaTriple) -> "TripleKey":
        return cls(aspect=normalize_aspect(t.aspect), sentiment=t.sentiment)


def _normalize_triples(triples: list[AbsaTriple]) -> list[AbsaTriple]:
    return [
        AbsaTriple(
            aspect=normalize_aspect(t.aspect),
            opinion=t.opinion,
            sentiment=t.sentiment,
            confidence=t.confidence,
        )
        for t in triples
    ]


def _keys(triples: list[AbsaTriple]) -> set[TripleKey]:
    return {TripleKey.from_triple(t) for t in _normalize_triples(triples)}


@dataclass(frozen=True, slots=True)
class TripleScores:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def triple_f1(predicted: list[AbsaTriple], gold: list[AbsaTriple]) -> TripleScores:
    """Set-based P/R/F1 over ``(aspect, sentiment)`` keys."""
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


def macro_f1_review(
    predictions: dict[str, list[AbsaTriple]],
    gold: dict[str, list[AbsaTriple]],
) -> TripleScores:
    """Unweighted mean of per-review F1 over scored gold reviews."""
    per_review: list[TripleScores] = []
    for rid, gold_triples in gold.items():
        if rid not in predictions:
            continue
        pred_triples = predictions[rid]
        if not gold_triples and not pred_triples:
            continue
        per_review.append(triple_f1(pred_triples, gold_triples))
    if not per_review:
        return TripleScores(0.0, 0.0, 0.0, 0, 0, 0)
    n = len(per_review)
    return TripleScores(
        precision=sum(s.precision for s in per_review) / n,
        recall=sum(s.recall for s in per_review) / n,
        f1=sum(s.f1 for s in per_review) / n,
        tp=sum(s.tp for s in per_review),
        fp=sum(s.fp for s in per_review),
        fn=sum(s.fn for s in per_review),
    )


def per_aspect_scores(
    predictions: dict[str, list[AbsaTriple]],
    gold: dict[str, list[AbsaTriple]],
) -> dict[str, TripleScores]:
    """Pooled micro P/R/F1 per canonical aspect across scored reviews."""
    scored_ids = [rid for rid in gold if rid in predictions]
    aspects: set[str] = set()
    for rid in scored_ids:
        for t in _normalize_triples(gold[rid]) + _normalize_triples(predictions[rid]):
            aspects.add(t.aspect)

    out: dict[str, TripleScores] = {}
    for aspect in sorted(aspects):
        tp = fp = fn = 0
        for rid in scored_ids:
            pred = {k for k in _keys(predictions[rid]) if k.aspect == aspect}
            ref = {k for k in _keys(gold[rid]) if k.aspect == aspect}
            tp += len(pred & ref)
            fp += len(pred - ref)
            fn += len(ref - pred)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        out[aspect] = TripleScores(
            precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn
        )
    return out


def macro_f1_aspect(
    per_aspect: dict[str, TripleScores],
    *,
    min_support: int = 5,
    gold: dict[str, list[AbsaTriple]] | None = None,
    predictions: dict[str, list[AbsaTriple]] | None = None,
) -> float | None:
    """Mean per-aspect F1 over aspects with gold support >= min_support."""
    support: dict[str, int] = {}
    if gold is not None and predictions is not None:
        for rid in gold:
            if rid not in predictions:
                continue
            for k in _keys(gold[rid]):
                support[k.aspect] = support.get(k.aspect, 0) + 1
    eligible = [
        s.f1
        for aspect, s in per_aspect.items()
        if support.get(aspect, 0) >= min_support
    ]
    if not eligible:
        return None
    return sum(eligible) / len(eligible)


def per_sentiment_scores(
    predictions: dict[str, list[AbsaTriple]],
    gold: dict[str, list[AbsaTriple]],
    *,
    min_sentiment_support: int = 10,
) -> dict[str, dict[str, Any]]:
    """Micro and macro F1 per sentiment class."""
    scored_ids = [rid for rid in gold if rid in predictions]
    out: dict[str, dict[str, Any]] = {}

    for sentiment in SENTIMENTS:
        tp = fp = fn = 0
        macro_f1s: list[float] = []
        support_gold = 0

        for rid in scored_ids:
            pred = {k for k in _keys(predictions[rid]) if k.sentiment == sentiment}
            ref = {k for k in _keys(gold[rid]) if k.sentiment == sentiment}
            support_gold += len(ref)
            if not ref and not pred:
                continue
            tp += len(pred & ref)
            fp += len(pred - ref)
            fn += len(ref - pred)
            s = triple_f1(
                [
                    AbsaTriple(aspect=k.aspect, opinion="", sentiment=k.sentiment)
                    for k in pred
                ],
                [
                    AbsaTriple(aspect=k.aspect, opinion="", sentiment=k.sentiment)
                    for k in ref
                ],
            )
            macro_f1s.append(s.f1)

        micro_p = tp / (tp + fp) if (tp + fp) else 0.0
        micro_r = tp / (tp + fn) if (tp + fn) else 0.0
        micro_f1 = (
            2 * micro_p * micro_r / (micro_p + micro_r)
            if (micro_p + micro_r)
            else 0.0
        )
        macro_f1 = sum(macro_f1s) / len(macro_f1s) if macro_f1s else 0.0
        out[sentiment] = {
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "support_gold": support_gold,
            "low_support": support_gold < min_sentiment_support,
        }
    return out


def classify_review_errors(
    predicted: list[AbsaTriple],
    gold: list[AbsaTriple],
) -> set[str]:
    """Per-review error tags (missed aspect, spurious aspect, sentiment flip, etc.)."""
    pred = _keys(predicted)
    ref = _keys(gold)
    tags: set[str] = set()
    if pred == ref:
        tags.add("perfect")
    if ref - pred:
        tags.add("missed_aspect")
    if pred - ref:
        tags.add("spurious_aspect")
    pred_by_aspect = {k.aspect: k.sentiment for k in pred}
    ref_by_aspect = {k.aspect: k.sentiment for k in ref}
    for aspect in set(pred_by_aspect) & set(ref_by_aspect):
        if pred_by_aspect[aspect] != ref_by_aspect[aspect]:
            tags.add("sentiment_flip")
    return tags


def error_summary(
    predictions: dict[str, list[AbsaTriple]],
    gold: dict[str, list[AbsaTriple]],
) -> dict[str, int]:
    counts: dict[str, int] = {
        "perfect": 0,
        "missed_aspect": 0,
        "spurious_aspect": 0,
        "sentiment_flip": 0,
    }
    for rid, gold_triples in gold.items():
        if rid not in predictions:
            continue
        for tag in classify_review_errors(predictions[rid], gold_triples):
            counts[tag] = counts.get(tag, 0) + 1
    return counts


def jaccard_keys(
    predicted: list[AbsaTriple],
    gold: list[AbsaTriple],
) -> float:
    """Jaccard similarity on ``(aspect, sentiment)`` key sets."""
    pred = _keys(predicted)
    ref = _keys(gold)
    union = pred | ref
    if not union:
        return 1.0
    return len(pred & ref) / len(union)


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
    """Micro-averaged P/R/F1 over scored gold review ids."""
    tp = fp = fn = 0
    for rid, gold_triples in gold.items():
        if rid not in predictions:
            continue
        s = triple_f1(predictions[rid], gold_triples)
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


@dataclass
class AbsaQualityReport:
    n_gold_reviews: int
    n_scored_reviews: int
    coverage: float
    micro: TripleScores
    macro_review: TripleScores
    macro_aspect_f1: float | None
    macro_aspect_n: int
    min_aspect_support: int
    per_aspect: dict[str, dict[str, Any]] = field(default_factory=dict)
    per_sentiment: dict[str, dict[str, Any]] = field(default_factory=dict)
    error_summary: dict[str, int] = field(default_factory=dict)
    labeling_qa: dict[str, Any] = field(default_factory=dict)
    insufficient_support: bool = False

    def to_json(self) -> dict[str, Any]:
        def _scores(s: TripleScores) -> dict[str, Any]:
            return asdict(s)

        per_aspect_out = {}
        for aspect, scores in sorted(
            self.per_aspect.items(),
            key=lambda x: x[1].get("support_gold", 0),
            reverse=True,
        ):
            per_aspect_out[aspect] = scores

        return {
            "n_gold_reviews": self.n_gold_reviews,
            "n_scored_reviews": self.n_scored_reviews,
            "coverage": self.coverage,
            "micro": _scores(self.micro),
            "macro_review": {
                **_scores(self.macro_review),
                "n_reviews": self.n_scored_reviews,
            },
            "macro_aspect": {
                "f1": self.macro_aspect_f1,
                "n_aspects": self.macro_aspect_n,
                "min_support": self.min_aspect_support,
                "insufficient_support": self.insufficient_support,
            },
            "per_aspect": per_aspect_out,
            "per_sentiment": self.per_sentiment,
            "error_summary": self.error_summary,
            "labeling_qa": self.labeling_qa,
        }


def build_absa_quality_report(
    predictions: dict[str, list[AbsaTriple]],
    gold: dict[str, list[AbsaTriple]],
    *,
    min_support: int = 5,
    min_sentiment_support: int = 10,
    labeling_qa: dict[str, Any] | None = None,
) -> AbsaQualityReport:
    scored_ids = [rid for rid in gold if rid in predictions]
    macro_scored = [
        rid for rid in scored_ids if gold[rid] or predictions[rid]
    ]
    n_gold = len(gold)
    n_scored = len(macro_scored)
    coverage = len(scored_ids) / n_gold if n_gold else 0.0

    micro = evaluate_predictions(predictions, gold)
    macro_review = macro_f1_review(predictions, gold)
    per_aspect = per_aspect_scores(predictions, gold)

    aspect_support: dict[str, int] = {}
    for rid in macro_scored:
        for k in _keys(gold[rid]):
            aspect_support[k.aspect] = aspect_support.get(k.aspect, 0) + 1

    macro_aspect = macro_f1_aspect(
        per_aspect,
        min_support=min_support,
        gold=gold,
        predictions=predictions,
    )
    eligible_aspects = [
        a for a, c in aspect_support.items() if c >= min_support
    ]

    per_aspect_json: dict[str, dict[str, Any]] = {}
    for aspect, scores in per_aspect.items():
        per_aspect_json[aspect] = {
            **asdict(scores),
            "support_gold": aspect_support.get(aspect, 0),
            "support_pred": sum(
                1
                for rid in macro_scored
                for k in _keys(predictions[rid])
                if k.aspect == aspect
            ),
        }

    return AbsaQualityReport(
        n_gold_reviews=n_gold,
        n_scored_reviews=n_scored,
        coverage=coverage,
        micro=micro,
        macro_review=macro_review,
        macro_aspect_f1=macro_aspect,
        macro_aspect_n=len(eligible_aspects),
        min_aspect_support=min_support,
        per_aspect=per_aspect_json,
        per_sentiment=per_sentiment_scores(
            predictions, gold, min_sentiment_support=min_sentiment_support
        ),
        error_summary=error_summary(predictions, gold),
        labeling_qa=labeling_qa or {},
        insufficient_support=macro_aspect is None,
    )
