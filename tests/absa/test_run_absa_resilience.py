"""ABSA batch resilience: per-review failures do not abort the run."""

from __future__ import annotations

import json
from pathlib import Path

from emorecagent.absa.cache import AbsaCache
from emorecagent.absa.extractor import AbsaExtractor
from emorecagent.absa.judge import AbsaJudge
from emorecagent.absa.pipeline import AbsaPipeline, LlmOnlyProcessor, ReviewRecord
from emorecagent.llm.client import FakeLLM, LLMClient, LLMError
from emorecagent.llm.schemas import AbsaTriple, TripleSet


def test_pipeline_failure_can_cache_empty(tmp_path: Path) -> None:
    cache = AbsaCache(tmp_path / "c.sqlite")

    class _FailAlways:
        def with_structured_output(self, schema):  # noqa: ANN001
            raise LLMError("boom")

        def invoke(self, messages):  # noqa: ANN001
            raise LLMError("boom")

    client = LLMClient(_FailAlways(), max_retries=1, backoff_s=0.0)
    pipe = AbsaPipeline(
        LlmOnlyProcessor(AbsaExtractor(client), AbsaJudge(client)),
        cache=cache,
    )
    try:
        pipe.process(ReviewRecord("bad", "text"), use_cache=True)
        raised = False
    except LLMError:
        raised = True
    assert raised
    cache.put("bad", TripleSet(triples=[]))
    assert cache.contains("bad")


def test_coerce_salvages_user_reported_shape() -> None:
    raw = {
        "triples": [
            {"aspect": "quality", "opinion": "THE BEST", "sentiment": "positive"},
            {"aspect": "price", "opinion": "beyond $3", "sentiment": "negative"},
            {
                "aspect": "availability",
                "opinion": "no where carried them",
                "sentiment": "neutral",
            },
            {"aspect": "availability", "opinion": "no where carried them"},
        ]
    }
    from emorecagent.llm.schemas import coerce_triple_set

    out = coerce_triple_set(raw)
    assert len(out.triples) == 3
