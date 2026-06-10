"""U4 ABSA pipeline tests with FakeLLM."""

from __future__ import annotations

from emorecagent.absa.cache import AbsaCache
from emorecagent.absa.extractor import AbsaExtractor
from emorecagent.absa.judge import AbsaJudge
from emorecagent.absa.normalize import normalize_aspect
from emorecagent.absa.pipeline import AbsaPipeline, ReviewRecord
from emorecagent.absa.quality import triple_f1
from emorecagent.llm.client import FakeLLM, LLMClient
from emorecagent.llm.schemas import AbsaTriple, TripleSet

BATTERY_REVIEW = "The battery lasts long but charging is slow."


def _battery_extract() -> str:
    return TripleSet(
        triples=[
            AbsaTriple(aspect="battery", opinion="lasts long", sentiment="positive"),
            AbsaTriple(aspect="charging", opinion="slow", sentiment="negative"),
            AbsaTriple(aspect="screen", opinion="bright", sentiment="positive", confidence=0.2),
        ]
    ).model_dump_json()


def _battery_judge() -> str:
    # judge drops unsupported "screen" and low-confidence items
    return TripleSet(
        triples=[
            AbsaTriple(aspect="battery", opinion="lasts long", sentiment="positive", confidence=0.95),
            AbsaTriple(aspect="charging", opinion="slow", sentiment="negative", confidence=0.9),
        ]
    ).model_dump_json()


def _pipeline(fake: FakeLLM, cache: AbsaCache | None = None) -> AbsaPipeline:
    client = LLMClient(fake)
    return AbsaPipeline(
        AbsaExtractor(client),
        AbsaJudge(client, min_confidence=0.5),
        cache=cache,
    )


def test_battery_review_yields_expected_triples() -> None:
    fake = FakeLLM([_battery_extract(), _battery_judge()])
    out = _pipeline(fake).process(ReviewRecord("r1", BATTERY_REVIEW), use_cache=False)
    aspects = {(t.aspect, t.sentiment) for t in out.triples}
    assert ("battery", "positive") in aspects
    assert ("charging", "negative") in aspects
    assert all(t.aspect != "screen" for t in out.triples)


def test_normalize_maps_synonyms() -> None:
    assert normalize_aspect("Build") == "build quality"
    assert normalize_aspect("smell") == "scent"


def test_cache_hit_skips_llm_calls(tmp_path) -> None:
    cache = AbsaCache(tmp_path / "c.sqlite")
    cached = TripleSet(
        triples=[AbsaTriple(aspect="scent", opinion="ok", sentiment="neutral")]
    )
    cache.put("r99", cached)
    # FakeLLM with no responses — would return empty if called
    fake = FakeLLM([])
    out = _pipeline(fake, cache).process(ReviewRecord("r99", "ignored"), use_cache=True)
    assert out.triples[0].aspect == "scent"


def test_partial_run_resumes_without_duplicate(tmp_path) -> None:
    cache = AbsaCache(tmp_path / "c.sqlite")
    fake = FakeLLM([_battery_extract(), _battery_judge()])
    pipe = _pipeline(fake, cache)
    pipe.process(ReviewRecord("r1", BATTERY_REVIEW))
    # second run: cache hit, no extra fake responses consumed
    fake2 = FakeLLM([])
    out = _pipeline(fake2, cache).process(ReviewRecord("r1", BATTERY_REVIEW))
    assert len(out.triples) == 2


def test_quality_f1_hand_computed() -> None:
    gold = [
        AbsaTriple(aspect="battery", opinion="x", sentiment="positive"),
        AbsaTriple(aspect="charging", opinion="y", sentiment="negative"),
    ]
    pred = [
        AbsaTriple(aspect="battery", opinion="a", sentiment="positive"),
        AbsaTriple(aspect="scent", opinion="b", sentiment="positive"),
    ]
    s = triple_f1(pred, gold)
    assert s.tp == 1 and s.fp == 1 and s.fn == 1
    assert s.precision == 0.5 and s.recall == 0.5 and s.f1 == 0.5
