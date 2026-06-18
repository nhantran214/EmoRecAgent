"""ABSA quality metric tests."""

from __future__ import annotations

from emorecagent.absa.quality import (
    build_absa_quality_report,
    classify_review_errors,
    jaccard_keys,
    macro_f1_review,
    triple_f1,
)
from emorecagent.llm.schemas import AbsaTriple


def _t(aspect: str, sentiment: str = "positive") -> AbsaTriple:
    return AbsaTriple(aspect=aspect, opinion="x", sentiment=sentiment)


def test_synonym_match_smell_scent() -> None:
    pred = [_t("scent", "positive")]
    gold = [_t("smell", "positive")]
    assert triple_f1(pred, gold).f1 == 1.0


def test_sentiment_flip_tag() -> None:
    tags = classify_review_errors(
        [_t("comfort", "positive")],
        [_t("comfort", "negative")],
    )
    assert "sentiment_flip" in tags


def test_both_empty_excluded_from_macro() -> None:
    gold = {"r1": [], "r2": [_t("scent")]}
    pred = {"r1": [], "r2": [_t("scent")]}
    macro = macro_f1_review(pred, gold)
    assert macro.f1 == 1.0


def test_macro_perfect_and_zero() -> None:
    gold = {"r1": [_t("scent")], "r2": [_t("comfort")]}
    pred = {"r1": [_t("scent")], "r2": []}
    macro = macro_f1_review(pred, gold)
    assert macro.f1 == 0.5


def test_macro_aspect_insufficient_support() -> None:
    gold = {"r1": [_t("rare")]}
    pred = {"r1": [_t("rare")]}
    report = build_absa_quality_report(pred, gold, min_support=5)
    assert report.macro_aspect_f1 is None
    assert report.insufficient_support


def test_jaccard_identical() -> None:
    triples = [_t("scent")]
    assert jaccard_keys(triples, triples) == 1.0
