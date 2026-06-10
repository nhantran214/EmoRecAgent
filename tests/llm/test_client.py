"""U3 LLM client tests: FakeLLM, structured output, retries."""

from __future__ import annotations

import pytest

from emorecagent.llm.client import FakeLLM, LLMClient, LLMError
from emorecagent.llm.schemas import AbsaTriple, TripleSet


def test_fake_llm_returns_scripted_structured_json() -> None:
    triples = TripleSet(
        triples=[
            AbsaTriple(aspect="battery", opinion="long lasting", sentiment="positive")
        ]
    )
    client = LLMClient(FakeLLM([triples.model_dump_json()]))
    out = client.invoke_structured("extract", TripleSet)
    assert len(out.triples) == 1
    assert out.triples[0].aspect == "battery"


def test_structured_retry_then_success() -> None:
    bad = "not json"
    good = TripleSet(triples=[]).model_dump_json()
    fake = FakeLLM([bad, good])
    client = LLMClient(fake, max_retries=3, backoff_s=0.0)
    out = client.invoke_structured("x", TripleSet)
    assert out.triples == []


def test_structured_exhausted_retries_raises_llm_error() -> None:
    fake = FakeLLM(["{broken", "{also-broken"])
    client = LLMClient(fake, max_retries=2, backoff_s=0.0)
    with pytest.raises(LLMError, match="Structured output failed"):
        client.invoke_structured("x", TripleSet)


def test_invoke_text_retries_on_failure() -> None:
    class _Flaky:
        def __init__(self) -> None:
            self.n = 0

        def invoke(self, messages):  # noqa: ANN001
            from langchain_core.messages import AIMessage

            self.n += 1
            if self.n == 1:
                raise RuntimeError("transient")
            return AIMessage(content="hello")

        def with_structured_output(self, schema):  # noqa: ANN001
            raise NotImplementedError

    client = LLMClient(_Flaky(), max_retries=2, backoff_s=0.0)
    assert client.invoke_text("hi") == "hello"


def test_fake_llm_structured_direct() -> None:
    payload = TripleSet(
        triples=[AbsaTriple(aspect="scent", opinion="nice", sentiment="positive")]
    )
    fake = FakeLLM([payload])
    out = fake.with_structured_output(TripleSet).invoke([])
    assert out.triples[0].sentiment == "positive"
