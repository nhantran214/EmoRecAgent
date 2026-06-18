"""Tests for loose ABSA triple parsing."""

from __future__ import annotations

from emorecagent.llm.schemas import (
    TripleSet,
    coerce_triple_set,
    salvage_triple_set_from_error,
)


def test_coerce_drops_missing_sentiment_and_dedupes() -> None:
    raw = {
        "triples": [
            {"aspect": "price", "opinion": "too high", "sentiment": "negative"},
            {"aspect": "price", "opinion": "too high", "sentiment": "negative"},
            {"aspect": "availability", "opinion": "not in stores"},
        ]
    }
    out = coerce_triple_set(raw)
    assert len(out.triples) == 1
    assert out.triples[0].aspect == "price"


def test_coerce_caps_max_triples() -> None:
    rows = [
        {
            "aspect": f"aspect_{i}",
            "opinion": f"opinion {i}",
            "sentiment": "positive",
        }
        for i in range(50)
    ]
    out = coerce_triple_set({"triples": rows})
    assert len(out.triples) == 24


def test_salvage_from_parser_error_message() -> None:
    exc = Exception(
        'Failed to parse TripleSet from completion {"triples": ['
        '{"aspect": "quality", "opinion": "great", "sentiment": "positive"}, '
        '{"aspect": "price", "opinion": "high"}]}'
    )
    out = salvage_triple_set_from_error(exc)
    assert out is not None
    assert len(out.triples) == 1
    assert out.triples[0].aspect == "quality"


def test_coerce_triple_set_instance() -> None:
    from emorecagent.llm.schemas import AbsaTriple

    ts = TripleSet(
        triples=[
            AbsaTriple(aspect="scent", opinion="nice", sentiment="positive"),
            AbsaTriple(aspect="scent", opinion="nice", sentiment="positive"),
        ]
    )
    out = coerce_triple_set(ts)
    assert len(out.triples) == 1


def test_coerce_confidence_percent_scale() -> None:
    from emorecagent.llm.schemas import AbsaTriple, coerce_hybrid_verdict

    triple = AbsaTriple.model_validate(
        {
            "aspect": "packaging",
            "opinion": "broken tube",
            "sentiment": "negative",
            "confidence": 100,
        }
    )
    assert triple.confidence == 1.0

    verdict = coerce_hybrid_verdict(
        {
            "triples": [
                {
                    "aspect": "texture",
                    "opinion": "nice",
                    "sentiment": "positive",
                },
                {
                    "aspect": "packaging",
                    "opinion": "leaked",
                    "sentiment": "negative",
                    "confidence": 100,
                },
            ],
            "needs_repair": True,
            "missing_aspect_hints": ["texture"],
        }
    )
    assert verdict.needs_repair is True
    assert verdict.triples[-1].confidence == 1.0


def test_salvage_hybrid_verdict_from_percent_confidence() -> None:
    from emorecagent.llm.schemas import salvage_hybrid_verdict_from_error

    exc = Exception(
        'Failed to parse HybridAbsaVerdict from completion {"triples": ['
        '{"aspect": "packaging", "opinion": "leaked", "sentiment": "negative", '
        '"confidence": 100}], "needs_repair": true, "missing_aspect_hints": []}'
    )
    out = salvage_hybrid_verdict_from_error(exc)
    assert out is not None
    assert out.triples[0].confidence == 1.0
