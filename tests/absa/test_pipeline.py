"""U4 ABSA pipeline tests with FakeLLM."""

from __future__ import annotations

import textwrap

from emorecagent.absa.cache import AbsaCache
from emorecagent.absa.extractor import AbsaExtractor
from emorecagent.absa.judge import AbsaJudge
from emorecagent.absa.normalize import normalize_aspect
from emorecagent.absa.classical import MockClassicalAbsaTool
from emorecagent.absa.pipeline import (
    AbsaPipeline,
    LlmOnlyProcessor,
    ReviewRecord,
    build_mock_hybrid_pipeline,
)
from emorecagent.llm.schemas import HybridAbsaVerdict
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
        LlmOnlyProcessor(
            AbsaExtractor(client),
            AbsaJudge(client, min_confidence=0.5),
        ),
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


def test_hybrid_mock_pipeline_caches_second_call(tmp_path) -> None:
    cache = AbsaCache(tmp_path / "c.sqlite")
    tool = MockClassicalAbsaTool(
        [
            AbsaTriple(
                aspect="battery",
                opinion="",
                sentiment="positive",
                confidence=0.95,
            ),
        ]
    )
    verdict = HybridAbsaVerdict(
        triples=[
            AbsaTriple(
                aspect="battery",
                opinion="lasts",
                sentiment="positive",
                confidence=0.9,
            ),
        ],
        needs_repair=False,
    )
    fake = FakeLLM([verdict.model_dump_json()])
    pipe = build_mock_hybrid_pipeline(LLMClient(fake), tool, cache=cache)
    rec = ReviewRecord("h1", BATTERY_REVIEW)
    pipe.process(rec)
    fake2 = FakeLLM([])
    pipe2 = build_mock_hybrid_pipeline(LLMClient(fake2), tool, cache=cache)
    out = pipe2.process(rec)
    assert len(out.triples) == 1
    assert out.triples[0].aspect == "battery"


def test_classical_only_processor_filters_confidence() -> None:
    from emorecagent.absa.pipeline import ClassicalOnlyProcessor

    tool = MockClassicalAbsaTool(
        [
            AbsaTriple(
                aspect="battery",
                opinion="lasts long",
                sentiment="positive",
                confidence=0.95,
            ),
            AbsaTriple(
                aspect="noise",
                opinion="loud",
                sentiment="negative",
                confidence=0.2,
            ),
        ]
    )
    pipe = AbsaPipeline(
        ClassicalOnlyProcessor(tool, min_confidence=0.5),
        cache=None,
    )
    out = pipe.process(ReviewRecord("c1", BATTERY_REVIEW), use_cache=False)
    assert len(out.triples) == 1
    assert out.triples[0].aspect == "battery"


def test_classical_process_batch(tmp_path) -> None:
    from emorecagent.absa.pipeline import ClassicalOnlyProcessor

    tool = MockClassicalAbsaTool(
        [
            AbsaTriple(
                aspect="battery",
                opinion="lasts long",
                sentiment="positive",
                confidence=0.95,
            ),
        ]
    )
    cache = AbsaCache(tmp_path / "c.sqlite")
    pipe = AbsaPipeline(ClassicalOnlyProcessor(tool, min_confidence=0.5), cache=cache)
    recs = [
        ReviewRecord("b1", BATTERY_REVIEW),
        ReviewRecord("b2", BATTERY_REVIEW),
    ]
    outs = pipe.process_batch(recs, use_cache=True)
    assert len(outs) == 2
    assert all(len(o.triples) == 1 for o in outs)
    assert cache.contains("b1") and cache.contains("b2")


def test_build_absa_pipeline_classical_without_client(tmp_path) -> None:
    from emorecagent.absa.pipeline import build_absa_pipeline
    from emorecagent.config import load_config

    yaml = textwrap.dedent(
        f"""
        experiment:
          name: t
          seed: 1
        data:
          category: Beauty_and_Personal_Care
          review_path: r.jsonl
          meta_path: m.jsonl
          out_dir: {tmp_path}
          k_core: 5
          max_users: 10
          max_items: 10
          min_history: 2
          min_distinct_aspects: 1
        scoring:
          alpha: 0.5
          lambda_decay: 0.01
          affective_rescaled: true
          helpful_vote_cap: 10
        cf:
          backend: svd
          factors: 8
        absa:
          targets_path: t.jsonl
          cache_path: {tmp_path / "build.sqlite"}
          gold_path: g.jsonl
          backend: classical
          pipeline_version: classical-v1
        agents: {{}}
        eval: {{}}
        llm:
          model: m
        tgi:
          base_url: http://localhost:8080
        ablation:
          reflection: true
          dynamic_weights: true
          aspect_term: true
        tisasrec_align: {{}}
        """
    )
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml, encoding="utf-8")
    cfg = load_config(cfg_path)
    tool = MockClassicalAbsaTool(
        [
            AbsaTriple(
                aspect="battery",
                opinion="lasts long",
                sentiment="positive",
                confidence=0.95,
            ),
        ]
    )
    built = build_absa_pipeline(cfg, client=None, classical_tool=tool)
    out = built.process(ReviewRecord("c2", BATTERY_REVIEW), use_cache=False)
    assert len(out.triples) == 1


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
